"""
rag/embed.py

Step three of the vector pipeline: turn every chunk into a vector.

WHAT AN EMBEDDING ACTUALLY IS
-----------------------------
A list of numbers standing for a piece of text, arranged so that texts about
similar things end up pointing in similar directions. That is the whole idea.
Everything after this step operates on the numbers and never sees the text
again, so whatever meaning fails to survive this conversion is gone.

TWO KINDS, AND WHY THE DEFAULT IS THE TRANSPARENT ONE
-----------------------------------------------------
    sparse (TF-IDF)   one dimension per word in your document's vocabulary.
                      Thousands of dimensions, almost all zero. Every
                      dimension is a word you can look up, so a vector can be
                      read and checked by hand.

    dense (neural)    a few hundred dimensions from a trained model. No
                      dimension means anything on its own. Far better at
                      matching paraphrases, and completely opaque.

This module defaults to TF-IDF, computed from your document's own vocabulary,
in pure Python with no dependencies. That is not a toy: sparse retrieval is
real retrieval, and BM25 (a later step) is built on the same counts. It is
chosen because you can print a vector, point at dimension 41, and see that it
means "reimbursement" -- which you cannot do with a dense model, and which is
the difference between understanding this step and taking it on faith.

Swap in a dense backend when you want paraphrase matching. The interface is
four methods; see EmbeddingBackend at the bottom of this file.

THE LIMIT OF THE DEFAULT, STATED PLAINLY
----------------------------------------
TF-IDF matches words, not meaning. A query asking about "reimbursement" will
not match a chunk that only ever says "paid back". Dense embeddings exist
precisely because of that gap. Later steps in this project (hybrid fusion,
reranking) are largely about papering over it from both directions.
"""

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Vector = Dict[int, float]           # sparse: term index -> weight

# A short, ordinary stopword list. Deliberately not exhaustive -- IDF already
# pushes common words toward zero, so this is belt and braces rather than the
# main mechanism.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "for", "from", "had", "has", "have", "if", "in", "into", "is", "it", "its",
    "may", "must", "no", "not", "of", "on", "or", "s", "shall", "should",
    "such", "than", "that", "the", "their", "then", "there", "these", "this",
    "to", "was", "were", "when", "which", "who", "will", "with", "within",
    "would", "you", "your",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> List[str]:
    """
    Lowercase, split on non-alphanumerics, drop stopwords, strip a few common
    English suffixes.

    The stemming here is crude on purpose: "claims" and "claiming" both become
    "claim", but "claimant" does not, and no linguistics is applied. Real
    systems use a proper stemmer or a subword tokenizer. The point of doing it
    badly and visibly is that you can see WHY it matters -- every failure to
    normalise a word is a query that will not match a chunk containing it.
    """
    out = []
    for raw in TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS or len(raw) < 2:
            continue
        out.append(_stem(raw))
    return out


#  Words whose final "s" is part of the word, not a plural. Without this,
#  "access" becomes "acces" and "bonus" becomes "bonu" -- and worse, stemming
#  stops being idempotent: feeding "acces" back through strips another letter
#  and gives "acce", which matches nothing. Any term already in the vocabulary
#  must stem to itself, or a query built from vocabulary terms retrieves
#  nothing.
_KEEP_FINAL_S = ("ss", "us", "is", "as", "ys")


def _stem(word: str) -> str:
    if word[0].isdigit():
        return word                       # leave numbers alone
    # Note there is no "es" rule. Stripping "es" from "expenses" gives
    # "expens", which then loses another "s" on a second pass -- and worse,
    # "expense" stems to itself, so the singular and plural of the same word
    # end up in different dimensions. Removing just the "s" gives "expense"
    # for both, and it is already a fixed point. "boxes" becomes "boxe"
    # rather than "box", which is the price; a real stemmer handles it.
    for suffix in ("ements", "ement", "ings", "ing", "ies", "ers", "er",
                   "ed", "s"):
        if not word.endswith(suffix) or len(word) - len(suffix) < 4:
            continue
        if suffix == "s" and word.endswith(_KEEP_FINAL_S):
            continue
        stem = word[:-len(suffix)]
        if suffix == "ies":
            return stem + "y"
        return stem
    return word


# =============================================================================
#  VOCABULARY
# =============================================================================
class Vocabulary:
    """
    Every term worth having a dimension for, plus its inverse document
    frequency.

    IDF is the part that does the real work. A term appearing in every chunk
    tells you nothing about which chunk to pick, so it gets a weight near
    zero. A term appearing in two chunks out of two hundred is highly
    discriminating and gets a large one. This is why "the" does not dominate
    every vector despite being the most common word in the document.
    """

    def __init__(self, documents: Sequence[Sequence[str]],
                 min_df: int = 1, max_df_ratio: float = 1.0,
                 max_terms: Optional[int] = None):
        self.n_docs = len(documents)
        df: Dict[str, int] = {}
        tf: Dict[str, int] = {}
        for tokens in documents:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
            for term in tokens:
                tf[term] = tf.get(term, 0) + 1

        ceiling = max_df_ratio * self.n_docs
        kept = [(t, c) for t, c in df.items() if c >= min_df and c <= ceiling]
        kept.sort(key=lambda tc: (-tf[tc[0]] * self._idf(tc[1]), tc[0]))
        if max_terms:
            kept = kept[:max_terms]

        self.terms: List[str] = sorted(t for t, _ in kept)
        self.index: Dict[str, int] = {t: i for i, t in enumerate(self.terms)}
        self.df: Dict[str, int] = {t: df[t] for t in self.terms}
        self.tf: Dict[str, int] = {t: tf[t] for t in self.terms}
        self.idf: Dict[str, float] = {t: self._idf(df[t]) for t in self.terms}

    def _idf(self, doc_freq: int) -> float:
        # Smoothed, so a term in every document still gets a small positive
        # weight rather than exactly zero. Matches the usual library formula.
        return math.log((1 + self.n_docs) / (1 + doc_freq)) + 1.0

    def __len__(self) -> int:
        return len(self.terms)

    def most_informative(self, n: int = 12) -> List[Tuple[str, int, float]]:
        """
        Ranked by total usage x idf, not by idf alone.

        Sorting on idf alone surfaces every term that appears exactly once --
        page numbers, dates, section codes -- because they all tie at the
        maximum. Multiplying by how often the term is actually used favours
        words that are both reasonably common and still discriminating, which
        is what "informative" should mean.
        """
        rows = [(t, self.df[t], self.idf[t]) for t in self.terms]
        rows.sort(key=lambda r: (-self.tf[r[0]] * r[2], r[0]))
        return rows[:n]

    def most_common(self, n: int = 12) -> List[Tuple[str, int, float]]:
        rows = [(t, self.df[t], self.idf[t]) for t in self.terms]
        rows.sort(key=lambda r: (-r[1], r[0]))
        return rows[:n]


# =============================================================================
#  VECTOR MATHS
# =============================================================================
#  All of it, on sparse dictionaries. Six functions, no library, and every one
#  of them is arithmetic you could do on paper for a small vector.
# =============================================================================
def dot(a: Vector, b: Vector) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(weight * b.get(i, 0.0) for i, weight in a.items())


def magnitude(v: Vector) -> float:
    """Length of the arrow. Long vectors are long texts, not relevant ones."""
    return math.sqrt(sum(w * w for w in v.values()))


def l2_normalise(v: Vector) -> Vector:
    """
    Scale to length 1, so only DIRECTION survives.

    This is why a two-line chunk can outrank a two-page one. Without it, long
    texts win every comparison simply by containing more words, which measures
    verbosity rather than relevance.
    """
    length = magnitude(v)
    if length == 0:
        return {}
    return {i: w / length for i, w in v.items()}


def cosine(a: Vector, b: Vector) -> float:
    """
    The angle between two vectors, ignoring their lengths.

    For vectors already normalised to length 1, this is just the dot product
    -- which is why real systems normalise once at index time and then never
    divide again.
    """
    denominator = magnitude(a) * magnitude(b)
    return dot(a, b) / denominator if denominator else 0.0


def angle_degrees(a: Vector, b: Vector) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine(a, b)))))


def euclidean(a: Vector, b: Vector) -> float:
    """
    Straight-line distance. Sensitive to length, which is usually the wrong
    thing for text -- but for unit vectors it is a monotone function of cosine,
    so the two rank identically.
    """
    keys = set(a) | set(b)
    return math.sqrt(sum((a.get(i, 0.0) - b.get(i, 0.0)) ** 2 for i in keys))


# =============================================================================
#  BACKENDS
# =============================================================================
class TfidfEmbedder:
    """Sparse, inspectable, dependency-free. The default."""

    name = "tfidf"
    is_dense = False

    def __init__(self, sublinear_tf: bool = True, min_df: int = 1,
                 max_df_ratio: float = 1.0, max_terms: Optional[int] = None):
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.max_terms = max_terms
        self.vocab: Optional[Vocabulary] = None

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        docs = [tokenize(t) for t in texts]
        self.vocab = Vocabulary(docs, self.min_df, self.max_df_ratio,
                                self.max_terms)
        return self

    def counts(self, tokens: Iterable[str]) -> Dict[int, int]:
        assert self.vocab is not None, "call fit() first"
        out: Dict[int, int] = {}
        for term in tokens:
            i = self.vocab.index.get(term)
            if i is not None:
                out[i] = out.get(i, 0) + 1
        return out

    def _weight(self, raw_counts: Dict[int, int]) -> Vector:
        assert self.vocab is not None
        vec: Vector = {}
        for i, count in raw_counts.items():
            term = self.vocab.terms[i]
            tf = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            vec[i] = tf * self.vocab.idf[term]
        return vec

    def embed(self, text: str, normalise: bool = True) -> Vector:
        vec = self._weight(self.counts(tokenize(text)))
        return l2_normalise(vec) if normalise else vec

    def embed_query(self, text: str, normalise: bool = True) -> Vector:
        """
        Same vocabulary, same weights.

        Anything the query says that the document never says has no dimension
        to live in and is silently dropped. That is the sharpest limitation of
        this backend, and unknown_terms() exists so you can see it happen.
        """
        return self.embed(text, normalise)

    def unknown_terms(self, text: str) -> List[str]:
        assert self.vocab is not None
        return [t for t in dict.fromkeys(tokenize(text))
                if t not in self.vocab.index]

    def explain(self, vec: Vector, top: int = 10) -> List[Tuple[str, float]]:
        """The heaviest dimensions of a vector, as words."""
        assert self.vocab is not None
        rows = [(self.vocab.terms[i], w) for i, w in vec.items()]
        rows.sort(key=lambda r: -r[1])
        return rows[:top]

    @property
    def dimensions(self) -> int:
        return len(self.vocab) if self.vocab else 0


class SvdEmbedder(TfidfEmbedder):
    """
    TF-IDF squeezed into a few hundred dense dimensions.

    This is the bridge between the two worlds: it produces dense vectors like a
    neural model, but by rotating the sparse ones rather than by learning
    anything. Two chunks using different words for the same idea can end up
    close, because the rotation groups words that co-occur -- which is the
    thing plain TF-IDF cannot do.

    Needs numpy. If you do not have it, stay on tfidf; nothing else in the
    project requires this.
    """

    name = "svd"
    is_dense = True

    def __init__(self, n_components: int = 96, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self._components = None

    def fit(self, texts: Sequence[str]) -> "SvdEmbedder":
        import numpy as np
        super().fit(texts)
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=float)
        for row, text in enumerate(texts):
            for i, w in super().embed(text).items():
                matrix[row, i] = w
        k = min(self.n_components, min(matrix.shape) - 1) or 1
        _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
        self._components = vt[:k]
        return self

    def embed(self, text: str, normalise: bool = True) -> Vector:
        import numpy as np
        sparse = super().embed(text, normalise=False)
        dense_in = np.zeros(len(self.vocab))
        for i, w in sparse.items():
            dense_in[i] = w
        projected = self._components @ dense_in
        vec = {i: float(v) for i, v in enumerate(projected) if v}
        return l2_normalise(vec) if normalise else vec

    def explain(self, vec: Vector, top: int = 10):
        """
        Dense dimensions have no names. Each one is a blend of many words, so
        the honest thing to report is the words that weigh most heavily in
        that component -- an approximation, not a label.
        """
        assert self._components is not None
        rows = []
        for i, w in sorted(vec.items(), key=lambda r: -abs(r[1]))[:top]:
            comp = self._components[i]
            heaviest = comp.argsort()[-3:][::-1]
            words = "/".join(self.vocab.terms[j] for j in heaviest)
            rows.append((f"dim{i} ~ {words}", w))
        return rows

    @property
    def dimensions(self) -> int:
        return 0 if self._components is None else len(self._components)


BACKENDS = {"tfidf": TfidfEmbedder, "svd": SvdEmbedder}


# =============================================================================
#  ADDING A REAL DENSE BACKEND
# =============================================================================
#  Four methods and you are done. Untested here because this environment has
#  no key and no network to those services, so it is written out rather than
#  shipped:
#
#      class OpenAIEmbedder:
#          name, is_dense = "openai", True
#          def __init__(self, model="text-embedding-3-small"):
#              from openai import OpenAI
#              self.client, self.model = OpenAI(), model
#          def fit(self, texts):  return self          # nothing to learn
#          def embed(self, text, normalise=True):
#              raw = self.client.embeddings.create(
#                        model=self.model, input=text).data[0].embedding
#              vec = {i: v for i, v in enumerate(raw) if v}
#              return l2_normalise(vec) if normalise else vec
#          embed_query = embed
#          def unknown_terms(self, text):  return []   # no vocabulary to miss
#          def explain(self, vec, top=10): return []    # dimensions are opaque
#
#  Register it with BACKENDS["openai"] = OpenAIEmbedder and every later step
#  works unchanged, because they all speak sparse dictionaries.
# =============================================================================


# -----------------------------------------------------------------------------
#  THE EMBED NODE  (state -> updates)
# -----------------------------------------------------------------------------
def embed_node(state: dict) -> dict:
    chunks = state["chunks"]
    backend_name = state.get("embed_backend", "tfidf")
    backend_cls = BACKENDS[backend_name]

    kwargs = {}
    if backend_name == "svd":
        kwargs["n_components"] = state.get("embed_dimensions", 96)

    embedder = backend_cls(**kwargs)
    texts = [c.page_content for c in chunks]
    embedder.fit(texts)

    vectors = [embedder.embed(t) for t in texts]
    for chunk, vec in zip(chunks, vectors):
        chunk.metadata["vector_dims"] = len(vec)

    return {"embedder": embedder, "vectors": vectors,
            "vocabulary": getattr(embedder, "vocab", None),
            "embed_settings": {"backend": backend_name,
                               "dimensions": embedder.dimensions}}
