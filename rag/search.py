"""
rag/search.py

Step four: actually retrieve something.

Everything before this was preparation. This is the part that, given a
question, returns the chunks most likely to answer it. Two independent ways of
deciding "most likely", and they disagree more often than people expect.

    DENSE / COSINE      compares the query vector against every chunk vector.
                        Matches direction in meaning-space. Finds paraphrases
                        when the embedding is a neural one; with TF-IDF it
                        still matches on shared vocabulary, weighted by rarity.

    SPARSE / BM25       counts how often the query's words appear in each
                        chunk, damped so the tenth occurrence adds almost
                        nothing, and corrected so long chunks do not win by
                        length alone. No vectors, no model, no training.

BM25 is not a legacy fallback. It beats dense retrieval outright on exact
terms -- part numbers, statute references, error codes, names -- because those
are precisely the tokens an embedding model has no good representation for.
The right question is never which one to use; it is what to do when they
disagree, which is the next step.

WHY TOP-K IS THE MOST CONSEQUENTIAL NUMBER HERE
-----------------------------------------------
Retrieval returns a ranked list, and something must decide where to cut it.
Too small and the answer is left behind; too large and the model reads three
irrelevant chunks for every useful one. Most systems set K to 3 or 5 because
that is what the tutorial said. This module measures, on your document, where
the answer actually falls.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

from rag.embed import cosine, tokenize

Vector = Dict[int, float]


# =============================================================================
#  DENSE RETRIEVAL
# =============================================================================
def cosine_search(query_vec: Vector, vectors: Sequence[Vector],
                  top_k: Optional[int] = None) -> List[Tuple[int, float]]:
    """
    Compare the query against every chunk and sort. Brute force, exact.

    This is what an index replaces. At small scale it is also what an index
    should not replace: comparing against ten thousand vectors takes
    milliseconds, and approximate search would only add error.
    """
    scored = [(i, cosine(query_vec, v)) for i, v in enumerate(vectors)]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:top_k] if top_k else scored


# =============================================================================
#  BM25
# =============================================================================
class BM25:
    """
    Okapi BM25.

        score(q, d) = SUM over query terms t of
                          idf(t) * ( f(t,d) * (k1 + 1) )
                                 / ( f(t,d) + k1 * (1 - b + b * len(d)/avgdl) )

    Two ideas are doing the work, and both are corrections to naive counting:

    SATURATION (k1)
        A chunk mentioning "invoice" ten times is more about invoices than one
        mentioning it twice -- but not five times more. The k1 term makes each
        additional occurrence worth less than the last. Set k1 low and the
        second mention barely counts; set it high and BM25 approaches raw term
        frequency. Typical range 1.2 to 2.0.

    LENGTH NORMALISATION (b)
        A long chunk contains more of every word, so without correction the
        longest chunk wins every query. b scales the penalty by how far the
        chunk deviates from average length. b=1 fully normalises, b=0 ignores
        length entirely. Typical value 0.75.

    The IDF here differs from the TF-IDF one: it can go negative for terms
    appearing in more than half the corpus, which is why the +1 inside the log
    exists -- without it, a very common query term would actively subtract
    from a chunk's score.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = []
        self.doc_freq: Dict[str, int] = {}
        self.doc_len: List[int] = []
        self.avg_len: float = 0.0
        self.n_docs: int = 0

    def fit(self, texts: Sequence[str]) -> "BM25":
        self.doc_tokens = [tokenize(t) for t in texts]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.n_docs = len(texts)
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.doc_freq = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        self._counts = [self._count(tokens) for tokens in self.doc_tokens]
        return self

    @staticmethod
    def _count(tokens: Sequence[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for term in tokens:
            out[term] = out.get(term, 0) + 1
        return out

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        counts = self._counts[doc_index]
        length = self.doc_len[doc_index]
        norm = 1 - self.b + self.b * (length / self.avg_len if self.avg_len else 1)
        total = 0.0
        for term in set(query_tokens):
            freq = counts.get(term, 0)
            if not freq:
                continue
            total += self.idf(term) * (freq * (self.k1 + 1)) / (
                freq + self.k1 * norm)
        return total

    def search(self, query: str,
               top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        tokens = tokenize(query)
        scored = [(i, self.score(tokens, i)) for i in range(self.n_docs)]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k] if top_k else scored

    def explain(self, query: str, doc_index: int) -> List[Tuple[str, int, float, float]]:
        """Per-term contribution, so a score can be checked by hand."""
        counts = self._counts[doc_index]
        length = self.doc_len[doc_index]
        norm = 1 - self.b + self.b * (length / self.avg_len if self.avg_len else 1)
        rows = []
        for term in dict.fromkeys(tokenize(query)):
            freq = counts.get(term, 0)
            contribution = 0.0
            if freq:
                contribution = self.idf(term) * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * norm)
            rows.append((term, freq, self.idf(term), contribution))
        rows.sort(key=lambda r: -r[3])
        return rows


# =============================================================================
#  INDEX COST
# =============================================================================
#  Exact search compares the query against every vector. That is N x D
#  multiply-adds per query, and it is fine until suddenly it is not. These
#  functions compute where "not" begins for a given corpus rather than quoting
#  a rule of thumb.
# =============================================================================
def exact_search_cost(n_vectors: int, dimensions: int) -> Dict[str, float]:
    flops = n_vectors * dimensions
    return {"vectors": n_vectors, "dimensions": dimensions,
            "flops_per_query": flops,
            # A conservative single-core figure. Real throughput varies by an
            # order of magnitude with SIMD and memory layout, so treat this as
            # a scale indicator, not a benchmark.
            "seconds_per_query": flops / 2e9,
            "memory_mb": n_vectors * dimensions * 4 / 1e6}


INDEX_TYPES = [
    ("flat / exact", "none", "100%", "N x D per query",
     "Correct by construction. Right answer below a few hundred thousand vectors."),
    ("IVF", "cluster centroids", "90-98%", "N/nprobe x D",
     "Searches only the nearest clusters. Misses anything sitting just over a "
     "cluster border."),
    ("HNSW", "navigable graph", "95-99%", "log N x D",
     "Walks a graph of neighbours. Fast and accurate; memory hungry and slow "
     "to build."),
    ("PQ / quantisation", "compressed codes", "80-95%", "N x D but cheaper ops",
     "Stores approximations of vectors. Cuts memory by 8-32x and loses "
     "precision in exchange."),
]


def scale_table(dimensions: int,
                sizes: Sequence[int] = (1_000, 100_000, 10_000_000)) -> List[Dict]:
    return [exact_search_cost(n, dimensions) for n in sizes]


# =============================================================================
#  METADATA FILTERING
# =============================================================================
def filter_after(results: Sequence[Tuple[int, float]], keep: Sequence[bool],
                 top_k: int) -> List[Tuple[int, float]]:
    """
    Post-filtering: retrieve top_k, THEN drop what fails the filter.

    The trap is that you asked for k results and may receive fewer, or none.
    The filter is applied to a list that was already truncated, so anything
    matching your filter but ranked below k was never a candidate.
    """
    return [(i, s) for i, s in results[:top_k] if keep[i]]


def filter_before(results: Sequence[Tuple[int, float]], keep: Sequence[bool],
                  top_k: int) -> List[Tuple[int, float]]:
    """
    Pre-filtering: restrict the candidates, THEN take the top k.

    Always returns k results when k exist. Costs more, because the index
    cannot prune as aggressively -- and with some ANN structures it degrades
    to a near-exhaustive scan, which is why post-filtering keeps getting
    shipped despite the bug.
    """
    return [(i, s) for i, s in results if keep[i]][:top_k]


# -----------------------------------------------------------------------------
#  THE SEARCH NODES  (state -> updates)
# -----------------------------------------------------------------------------
def index_node(state: dict) -> dict:
    """Build the keyword index. The dense side needs no build step here."""
    bm25 = BM25(k1=state.get("bm25_k1", 1.5), b=state.get("bm25_b", 0.75))
    bm25.fit([c.page_content for c in state["chunks"]])
    return {"bm25": bm25,
            "index_cost": exact_search_cost(len(state["vectors"]),
                                            state["embed_settings"]["dimensions"])}


def retrieve_node(state: dict) -> dict:
    question = state["question"]
    top_k = state.get("top_k", 5)
    embedder = state["embedder"]

    query_vec = embedder.embed_query(question)
    dense = cosine_search(query_vec, state["vectors"])
    sparse = state["bm25"].search(question)

    chunks = state["chunks"]
    return {"query_vector": query_vec,
            "dense_ranking": dense,
            "sparse_ranking": sparse,
            "retrieved": [chunks[i] for i, _ in dense[:top_k]]}
