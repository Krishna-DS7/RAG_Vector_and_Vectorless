#!/usr/bin/env python3
"""
embed.py  --  turn chunks into vectors, and show what a vector actually is

    python3 embed.py report.pdf
    python3 embed.py                                  # built-in sample
    python3 embed.py doc.pdf --query "annual leave"
    python3 embed.py doc.pdf --backend svd            # dense, needs numpy
    python3 embed.py doc.pdf --size 600 --overlap 80
    python3 embed.py doc.pdf --terms 16               # wider term matrix
    python3 embed.py doc.pdf --out results/

This is the step where text stops being text. From here on every later stage
sees only numbers, so anything meaning-bearing that fails to survive the
conversion is gone and cannot be recovered by better ranking.

The default backend is TF-IDF over your document's own vocabulary: real
retrieval maths, no dependencies, and every dimension is a word you can point
at. Use --backend svd for dense vectors once you want to see the difference.
"""

import argparse
import json
import os
import sys

from pi_rag.parse import parse_node as tree_parse
from rag.chunk import chunk_node
from rag.embed import (
    BACKENDS,
    angle_degrees,
    cosine,
    dot,
    embed_node,
    euclidean,
    l2_normalise,
    magnitude,
    tokenize,
)
from rag.parse import parse_node as vector_parse
from shared.loader import load_node

BAR = "=" * 78


def rule(title):
    print(f"\n{BAR}\n{title}\n{BAR}")


def wrap(text, width=74, indent="  "):
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(indent + line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(indent + line)
    return "\n".join(out)


# =============================================================================
def section_vocabulary(embedder, chunks):
    rule("1.  VOCABULARY")
    vocab = embedder.vocab
    if vocab is None:
        print("\n  This backend has no vocabulary; its dimensions are learned.")
        return

    print(f"""
  chunks              {len(chunks)}
  distinct terms      {len(vocab)}   <- one dimension each

  Every chunk becomes a point in {len(vocab)}-dimensional space. Almost every
  coordinate is zero, because a chunk uses a few dozen of those terms.
""")
    print("  Terms that best tell chunks apart (used often, still selective):\n")
    print(f"    {'term':<20}{'in chunks':>11}{'idf weight':>13}")
    print("    " + "-" * 44)
    for term, df, idf in vocab.most_informative(8):
        print(f"    {term:<20}{df:>11}{idf:>13.3f}")

    print("\n  Most common terms (everywhere, so they identify nothing):\n")
    print(f"    {'term':<20}{'in chunks':>11}{'idf weight':>13}")
    print("    " + "-" * 44)
    for term, df, idf in vocab.most_common(5):
        print(f"    {term:<20}{df:>11}{idf:>13.3f}")

    print(wrap("\nThat second table is the mechanism. A term appearing in every "
               "chunk cannot help you choose between them, so its weight "
               "collapses toward zero. A term appearing twice in two hundred "
               "chunks is highly discriminating and gets a large one. Nobody "
               "wrote a stopword list to achieve this; the arithmetic does it."))


# =============================================================================
def section_term_matrix(embedder, chunks, n_terms):
    rule("2.  THE TERM MATRIX")
    vocab = embedder.vocab
    if vocab is None:
        print("\n  Not applicable to a dense backend.")
        return

    picked = [t for t, _, _ in vocab.most_informative(n_terms)]
    print(f"""
  Raw counts: how many times each term appears in each chunk. This is the
  whole input to the vectoriser -- word order, grammar and sentence structure
  are already gone at this point.

  Showing {len(picked)} of {len(vocab)} terms:
""")
    header = "".join(f"{t[:7]:>8}" for t in picked)
    print(f"    {'chunk':<8}{header}")
    print("    " + "-" * (8 + 8 * len(picked)))
    for chunk in chunks[:10]:
        counts = embedder.counts(tokenize(chunk.page_content))
        row = "".join(
            f"{counts.get(vocab.index[t], 0) or '.':>8}" for t in picked)
        print(f"    {chunk.metadata['chunk_id']:<8}{row}")
    if len(chunks) > 10:
        print(f"    ... and {len(chunks) - 10} more chunks")

    print(wrap("\nA dot means zero. Most of this grid is dots, which is what "
               "'sparse' means and why these vectors are stored as "
               "dictionaries rather than lists."))


# =============================================================================
def section_vector(embedder, chunks, vectors):
    rule("3.  ONE CHUNK, AS A VECTOR")
    target = max(range(len(chunks)), key=lambda i: len(vectors[i]))
    chunk, vec = chunks[target], vectors[target]

    print(f"""
  chunk {chunk.metadata['chunk_id']}, pages {chunk.metadata['start_page']}-{chunk.metadata['end_page']}, {len(chunk.page_content)} characters

  Opening: {' '.join(chunk.page_content.split())[:64]}...

  Non-zero dimensions: {len(vec)} of {embedder.dimensions}
""")
    print("  Its heaviest dimensions:\n")
    print(f"    {'dimension':<28}{'weight':>10}")
    print("    " + "-" * 39)
    for name, weight in embedder.explain(vec, 8):
        print(f"    {name:<28}{weight:>10.4f}")

    # Ask the backend for its own un-normalised vector. Reaching into the
    # tf-idf internals here would report the sparse length even when the
    # backend is dense, which is a different vector entirely.
    raw = embedder.embed(chunk.page_content, normalise=False)
    print(f"""
  Before normalising, this vector has length {magnitude(raw):.3f}.
  After normalising, it has length {magnitude(vec):.3f}.
""")
    print(wrap("Normalisation throws away length and keeps only direction. "
                   "That matters more than it sounds: without it, a long chunk "
                   "scores higher against everything simply by containing more "
                   "words. You would be ranking by verbosity. With it, a "
                   "two-line chunk and a two-page chunk compete on what they "
                   "are about."))


# =============================================================================
def section_vector_math(embedder, chunks, vectors):
    rule("4.  VECTOR MATHS")
    if len(vectors) < 2:
        print("\n  Only one chunk; nothing to compare.")
        return

    best = (0, 1, -1.0)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            score = cosine(vectors[i], vectors[j])
            if score > best[2]:
                best = (i, j, score)
    i, j, _ = best
    a, b = vectors[i], vectors[j]

    print(f"""
  The two most similar chunks in this document:

      {chunks[i].metadata['chunk_id']}   {' '.join(chunks[i].page_content.split())[:52]}...
      {chunks[j].metadata['chunk_id']}   {' '.join(chunks[j].page_content.split())[:52]}...

    dot product          {dot(a, b):.4f}
    length of first      {magnitude(a):.4f}
    length of second     {magnitude(b):.4f}
    cosine similarity    {cosine(a, b):.4f}
    angle between them   {angle_degrees(a, b):.1f} degrees
    euclidean distance   {euclidean(a, b):.4f}
""")
    print(wrap("Both vectors have length 1, so the dot product and the cosine "
               "are the same number. That is not a coincidence -- it is why "
               "production systems normalise once when indexing and then use "
               "the dot product forever after, skipping a division on every "
               "single comparison."))

    shared = sorted(set(a) & set(b), key=lambda k: -(a[k] * b[k]))[:5]
    if shared and embedder.vocab is not None:
        print("\n  What they have in common, by contribution to the score:\n")
        print(f"    {'term':<20}{'first':>9}{'second':>9}{'product':>10}")
        print("    " + "-" * 48)
        for k in shared:
            print(f"    {embedder.vocab.terms[k]:<20}{a[k]:>9.3f}"
                  f"{b[k]:>9.3f}{a[k] * b[k]:>10.4f}")
        print(wrap("\nThe score is just those products added up. Nothing is "
                   "hidden: similarity is a sum over shared words, weighted by "
                   "how rare each one is."))


# =============================================================================
def section_query(embedder, chunks, vectors, query, auto):
    rule("5.  A QUERY, AS A VECTOR")
    if auto:
        print(wrap(f"\nNo --query given, so this used a section title from your "
                   f"document as a realistic question."))
    print(f"\n  query: {query!r}\n")

    tokens = list(dict.fromkeys(tokenize(query)))
    unknown = embedder.unknown_terms(query)
    known = [t for t in tokens if t not in unknown]
    qvec = embedder.embed_query(query)

    print(f"  after tokenising    {tokens}")
    print(f"  in the vocabulary   {known}")
    print(f"  dropped, not in it  {unknown or '(none)'}")
    print(f"  non-zero dimensions {len(qvec)}\n")

    if unknown:
        print(wrap("Those dropped terms have no dimension to live in, so they "
                   "contribute nothing. The query is silently smaller than "
                   "what you typed. This is the sharpest limitation of a "
                   "sparse backend, and it is invisible at query time -- you "
                   "get results, they just were not scored on everything you "
                   "asked."))

    if not qvec:
        print(wrap("\nNothing in this query exists in the document's "
                   "vocabulary, so every score will be zero. A real system "
                   "must detect this and say so rather than returning its "
                   "least-bad guess."))
        return None

    scored = sorted(((cosine(qvec, v), i) for i, v in enumerate(vectors)),
                    reverse=True)
    print("\n  Closest chunks:\n")
    print(f"    {'rank':<6}{'chunk':<9}{'cosine':>9}{'pages':>10}   opening words")
    print("    " + "-" * 72)
    for rank, (score, i) in enumerate(scored[:5], 1):
        meta = chunks[i].metadata
        pages = (f"p{meta['start_page']}" if meta["start_page"] == meta["end_page"]
                 else f"p{meta['start_page']}-{meta['end_page']}")
        head = " ".join(chunks[i].page_content.split())[:28]
        print(f"    {rank:<6}{meta['chunk_id']:<9}{score:>9.4f}{pages:>10}   {head}")

    top = scored[0][0]
    pair_best = max(cosine(vectors[i], vectors[j])
                    for i in range(len(vectors))
                    for j in range(i + 1, len(vectors))) if len(vectors) > 1 else 0
    print(f"""
  best query-to-chunk score   {top:.4f}
  best chunk-to-chunk score   {pair_best:.4f}
""")
    if top < pair_best:
        print(wrap("The query scores lower than the closest pair of chunks, "
                   f"{top:.2f} against {pair_best:.2f}. A few query words are "
                   "being compared against a chunk containing hundreds, so "
                   "most of the chunk's dimensions have nothing to match "
                   "against and the score is diluted."))
    else:
        print(wrap(f"Here the query scores {top:.2f} against a best chunk pair "
                   f"of {pair_best:.2f} -- higher, because a short query "
                   "concentrates all its weight on a few terms, and those "
                   "terms happen to be the ones dominating the top chunk. "
                   "Short queries can score high or low depending entirely on "
                   "whether their few words are the rare ones."))
    print(wrap("\nEither way the lesson is the same: an absolute cosine value "
               "means nothing on its own. It depends on query length, chunk "
               "length and which terms happen to be rare in this document. "
               "Compare scores within a single query's results, never across "
               "queries, and never set a fixed relevance threshold -- it will "
               "be wrong for the next question."))
    return scored


# =============================================================================
def section_verdict(embedder, chunks, vectors, scored, settings):
    rule("6.  WHAT THIS STEP COST AND GAINED")
    zero = sum(1 for v in vectors if not v)
    avg_nonzero = sum(len(v) for v in vectors) / max(len(vectors), 1)

    print(f"""
  backend             {settings['backend']}
  dimensions          {settings['dimensions']}
  average non-zero    {avg_nonzero:.0f} per chunk
  empty vectors       {zero}
""")
    if zero:
        print(wrap(f"WARNING  {zero} chunks embedded to nothing. They are "
                   "unreachable: no query can ever retrieve them. Usually "
                   "boilerplate or a page of numbers."))

    if not embedder.is_dense:
        print(wrap("This backend matches WORDS. A question asking about "
                   "'reimbursement' will not match a chunk that only says "
                   "'paid back', because those are different dimensions with "
                   "nothing connecting them. Dense embeddings exist to close "
                   "that gap, at the cost of every dimension becoming "
                   "unreadable."))
        print(wrap("\nTry --backend svd to see the same document in dense "
                   "vectors, and notice that the closest-chunk list changes "
                   "while the maths does not."))
    else:
        print(wrap("Dense vectors match meaning better than words, and no "
                   "single dimension means anything you can name. The labels "
                   "above are approximations -- the heaviest words in each "
                   "component, not a definition of it. That opacity is the "
                   "real price of dense retrieval: when a result is wrong, "
                   "there is nothing to inspect."))

    print(wrap("\nFrom here the text is gone. Retrieval, ranking and filtering "
               "all operate on these numbers alone, so a meaning that failed "
               "to survive this step cannot be recovered later by a better "
               "ranker."))
    print(wrap("\nNext: similarity search over the whole set, and the index "
               "structures that make it fast enough to use."))


# =============================================================================
def pick_query(tree, chunks):
    """
    Choose a realistic demo query from the document itself.

    A section title is a good stand-in for a real question: it uses the
    document's own words about a real topic. If there is no outline, fall back
    to the heaviest terms of a middle chunk.
    """
    leaves = [n for n in tree.walk() if n.is_leaf and n.level > 0]
    if leaves:
        node = sorted(leaves, key=lambda n: n.size)[len(leaves) // 2]
        title = node.title
        if node.number and title.startswith(node.number):
            title = title[len(node.number):].strip()
        return title
    mid = chunks[len(chunks) // 2].page_content
    return " ".join(dict.fromkeys(tokenize(mid)))[:60] or "summary"


# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Embed a document's chunks and show what a vector is.")
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--query", default=None,
                    help="question to embed (default: a section title from the file)")
    ap.add_argument("--backend", choices=list(BACKENDS), default="tfidf")
    ap.add_argument("--dimensions", type=int, default=96,
                    help="components for --backend svd")
    ap.add_argument("--strategy", default="recursive",
                    choices=["fixed", "recursive", "structural"])
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--terms", type=int, default=10,
                    help="columns in the term matrix")
    ap.add_argument("--extractor", choices=["pypdf", "pdfplumber"], default=None)
    ap.add_argument("--out", default=None, metavar="DIR")
    args = ap.parse_args()

    if args.overlap >= args.size:
        sys.exit("overlap must be smaller than size")

    try:
        state = load_node({"path": args.path, "extractor": args.extractor})
    except FileNotFoundError:
        sys.exit(f"file not found: {args.path}")
    except ImportError as exc:
        sys.exit(str(exc))

    state = {**state, **vector_parse(state)}
    tree = tree_parse(state)["tree"]
    boundaries = [n.char_start for n in tree.walk() if n.level > 0]
    state = {**state, "chunk_strategy": args.strategy, "chunk_size": args.size,
             "chunk_overlap": args.overlap, "section_boundaries": boundaries}
    state = {**state, **chunk_node(state)}

    state = {**state, "embed_backend": args.backend,
             "embed_dimensions": args.dimensions}
    try:
        state = {**state, **embed_node(state)}
    except ImportError:
        sys.exit("the svd backend needs numpy:  pip install numpy")

    embedder = state["embedder"]
    chunks, vectors = state["chunks"], state["vectors"]

    auto = args.query is None
    query = args.query or pick_query(tree, chunks)

    print(BAR)
    print("EMBEDDING REPORT")
    print(BAR)
    print(f"\n  {state['pages'][0].metadata.get('source')}   "
          f"{len(chunks)} chunks, {args.strategy} at {args.size}/{args.overlap}")

    section_vocabulary(embedder, chunks)
    section_term_matrix(embedder, chunks, args.terms)
    section_vector(embedder, chunks, vectors)
    section_vector_math(embedder, chunks, vectors)
    scored = section_query(embedder, chunks, vectors, query, auto)
    section_verdict(embedder, chunks, vectors, scored, state["embed_settings"])

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        payload = []
        for chunk, vec in zip(chunks, vectors):
            payload.append({"chunk_id": chunk.metadata["chunk_id"],
                            "start_page": chunk.metadata["start_page"],
                            "end_page": chunk.metadata["end_page"],
                            "non_zero": len(vec),
                            "top_terms": embedder.explain(vec, 8)})
        with open(os.path.join(args.out, "vectors.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        if embedder.vocab is not None:
            with open(os.path.join(args.out, "vocabulary.json"), "w") as fh:
                json.dump({"terms": embedder.vocab.terms,
                           "idf": embedder.vocab.idf}, fh, indent=2)
        print(f"\n  written to {args.out}/\n      vectors.json"
              + ("\n      vocabulary.json" if embedder.vocab else ""))
    print()


if __name__ == "__main__":
    main()
