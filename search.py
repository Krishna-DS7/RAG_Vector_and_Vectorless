#!/usr/bin/env python3
"""
search.py  --  retrieve, and find out whether the retrieval is any good

    python3 search.py report.pdf --query "how long are records kept"
    python3 search.py                                   # built-in sample
    python3 search.py doc.pdf --top-k 3
    python3 search.py doc.pdf --k1 1.2 --b 0.5          # BM25 tuning
    python3 search.py doc.pdf --evaluate                # score it, no labelling
    python3 search.py doc.pdf --backend svd
    python3 search.py doc.pdf --out results/

This is where the pipeline becomes a retriever. Two independent methods rank
your chunks against a question, and the interesting part is not either ranking
on its own -- it is where they disagree, and how far down the list the right
answer actually sits.

--evaluate is the one to run. It builds a golden set from your document's own
section headings, so retrieval quality becomes a number on any structured PDF
without labelling anything by hand.
"""

import argparse
import json
import os
import sys

from pi_rag.parse import parse_node as tree_parse
from rag.chunk import chunk_node
from rag.embed import BACKENDS, embed_node, tokenize
from rag.parse import parse_node as vector_parse
from rag.search import (
    BM25,
    INDEX_TYPES,
    cosine_search,
    exact_search_cost,
    filter_after,
    filter_before,
    scale_table,
)
from shared import goldset
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


def _duration(seconds):
    """Readable at every scale, from a handful of vectors to ten million."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.1f} s"


def head(chunk, n=30):
    return " ".join(chunk.page_content.split())[:n]


def pages(chunk):
    meta = chunk.metadata
    return (f"p{meta['start_page']}" if meta["start_page"] == meta["end_page"]
            else f"p{meta['start_page']}-{meta['end_page']}")


# =============================================================================
def section_ranking(chunks, dense, sparse, query, top_k):
    rule("1.  THE RANKING")
    print(f"\n  query: {query!r}\n")
    print(f"  {'rank':<6}{'chunk':<9}{'cosine':>9}{'bm25':>9}{'pages':>9}"
          f"   opening words")
    print("  " + "-" * 74)
    bm_by_index = dict(sparse)
    for rank, (i, score) in enumerate(dense[:8], 1):
        mark = " <- cut" if rank == top_k else ""
        print(f"  {rank:<6}{chunks[i].metadata['chunk_id']:<9}{score:>9.4f}"
              f"{bm_by_index.get(i, 0):>9.3f}{pages(chunks[i]):>9}"
              f"   {head(chunks[i])}{mark}")
    print(wrap("\nSorted by cosine, with each chunk's BM25 score alongside. The "
               "two columns are not on the same scale and never will be -- "
               "cosine is bounded at 1, BM25 is unbounded and grows with query "
               "length. They cannot be added together, which is the problem "
               "hybrid fusion exists to solve."))


# =============================================================================
def section_topk(chunks, dense, relevant, top_k):
    rule("2.  WHERE THE ANSWER ACTUALLY SITS")
    if relevant is None:
        print(wrap("\nNo outline in this document, so there is no ground truth "
                   "for this query. Run --evaluate on a structured document to "
                   "see this measured."))
        return

    relevant_set = set(relevant)
    positions = [rank for rank, (i, _) in enumerate(dense, 1) if i in relevant_set]
    print(f"""
  chunks that genuinely answer this   {len(relevant)}
  their positions in the ranking      {positions or 'none found'}
""")
    print(f"  {'K':<5}{'recall':>10}{'precision':>12}   what you would send the model")
    print("  " + "-" * 70)
    for k in (1, 2, 3, 5, 10):
        if k > len(dense):
            break
        got = {i for i, _ in dense[:k]}
        hit = len(got & relevant_set)
        recall = hit / len(relevant)
        precision = hit / k
        note = f"{hit} of {k} chunks useful"
        print(f"  {k:<5}{recall:>9.0%}{precision:>12.0%}   {note}")

    if positions:
        need = max(positions)
        print(wrap(f"\nEvery answering chunk is inside the top {need}. Set K "
                   f"below that and part of the answer never reaches the "
                   "model, which then answers confidently from what it did "
                   "receive."))
    else:
        print(wrap("\nNo answering chunk appears anywhere in the ranking. "
                   "Retrieval failed outright for this query -- no value of K "
                   "helps."))
    print(wrap("\nThis is the trade behind K. Raising it protects recall and "
               "fills the context with chunks the model must read past; "
               "lowering it sharpens precision and risks dropping the half of "
               "the answer that ranked fourth."))


# =============================================================================
def section_bm25(chunks, bm25, sparse, query, k1, b):
    rule("3.  BM25, TAKEN APART")
    print(f"""
  k1 = {k1}   how fast repeated terms stop helping
  b  = {b}   how hard long chunks are penalised
  average chunk length {bm25.avg_len:.0f} tokens
""")
    top = sparse[0][0]
    print(f"  Why chunk {chunks[top].metadata['chunk_id']} scored "
          f"{sparse[0][1]:.3f}:\n")
    print(f"    {'term':<16}{'count':>7}{'idf':>9}{'contributes':>13}")
    print("    " + "-" * 46)
    for term, freq, idf, contribution in bm25.explain(query, top):
        print(f"    {term:<16}{freq:>7}{idf:>9.3f}{contribution:>13.4f}")
    print(wrap("\nThe score is that column added up. Every part of it is a "
               "count and a log; there is no model and nothing learned, which "
               "is exactly why BM25 cannot be fooled about whether a word is "
               "present -- and exactly why it cannot tell that 'paid back' "
               "means 'reimbursed'."))

    tokens = tokenize(query)
    if tokens:
        print("\n  What happens to the top ranking as the knobs move:\n")
        print(f"    {'setting':<28}{'top 3 chunks':<26}top score")
        print("    " + "-" * 64)
        for label, kk, bb in [(f"k1={k1}, b={b}  (current)", k1, b),
                              ("k1=0.5  little saturation", 0.5, b),
                              ("k1=3.0  near raw counts", 3.0, b),
                              ("b=0.0   ignore length", k1, 0.0),
                              ("b=1.0   full length norm", k1, 1.0)]:
            variant = BM25(k1=kk, b=bb).fit([c.page_content for c in chunks])
            ranked = variant.search(query, 3)
            ids = " ".join(chunks[i].metadata["chunk_id"] for i, _ in ranked)
            print(f"    {label:<28}{ids:<26}{ranked[0][1]:.3f}")
        print(wrap("\nIf the ordering barely moves, your chunks are similar in "
                   "length and the query terms are not repeated much -- which "
                   "is common for short chunks. On long, uneven documents "
                   "these knobs reorder the top of the list."))


# =============================================================================
def section_disagreement(chunks, dense, sparse, top_k, backend, is_dense):
    rule("4.  WHERE THE TWO METHODS DISAGREE")
    dense_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
    sparse_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}

    rows = []
    for i in range(len(chunks)):
        rows.append((abs(dense_rank[i] - sparse_rank[i]), i))
    rows.sort(reverse=True)

    dense_top = {i for i, _ in dense[:top_k]}
    sparse_top = {i for i, _ in sparse[:top_k]}
    agreed = dense_top & sparse_top

    print(f"""
  top {top_k} by cosine     {' '.join(chunks[i].metadata['chunk_id'] for i, _ in dense[:top_k])}
  top {top_k} by BM25       {' '.join(chunks[i].metadata['chunk_id'] for i, _ in sparse[:top_k])}
  both agree on       {' '.join(sorted(chunks[i].metadata['chunk_id'] for i in agreed)) or 'nothing'}
""")
    moved = [(gap, i) for gap, i in rows if gap > 0]
    if moved:
        print(f"  {'chunk':<9}{'cosine rank':>13}{'bm25 rank':>11}{'gap':>7}"
              f"   opening words")
        print("  " + "-" * 74)
        for gap, i in moved[:5]:
            print(f"  {chunks[i].metadata['chunk_id']:<9}{dense_rank[i]:>13}"
                  f"{sparse_rank[i]:>11}{gap:>7}   {head(chunks[i], 26)}")
    else:
        print("  No chunk is ranked differently by the two methods on this")
        print("  query. That happens when the query's terms are rare enough")
        print("  that both methods key on the same few chunks -- common with")
        print("  short documents and vocabulary-matched queries.")

    overlap_pct = 100 * len(agreed) / max(top_k, 1)
    print(wrap(f"\nThe two methods agree on {overlap_pct:.0f}% of the top "
               f"{top_k}."))

    if not is_dense:
        print(wrap(f"\nExpect high agreement here, and do not read it as "
                   f"confirmation. The {backend} backend and BM25 are BOTH "
                   "bag-of-words methods counting the same tokens over the "
                   "same vocabulary. They weight those counts differently, so "
                   "the ordering can shift, but they cannot disagree about "
                   "which words are present. They are two views of one "
                   "representation, not two independent opinions."))
        print(wrap("\nWhich matters, because hybrid search is justified by "
                   "combining INDEPENDENT evidence. Fusing two bag-of-words "
                   "rankings mostly buys you a smoother version of the same "
                   "answer. Run this again with --backend svd: the dense side "
                   "sees which words co-occur rather than only which are "
                   "present, and the rankings genuinely diverge."))
    else:
        print(wrap("\nThese two are measuring genuinely different things. "
                   "BM25 rewards a chunk for containing the query's exact "
                   "words. A dense vector rewards it for pointing the same way "
                   "in a space built from co-occurrence, which can happen with "
                   "none of those words present -- and can fail when the "
                   "matching word is a code or an identifier the model has no "
                   "good representation for."))
        print(wrap("\nSo when both rank a chunk highly, that is two "
                   "independent methods agreeing and worth believing. When "
                   "only one does, you have to decide -- and that decision is "
                   "what reciprocal rank fusion automates in the next step."))

    print(wrap("\nWhat fusion cannot fix, either way: when both methods are "
               "wrong in the same direction, averaging them produces a "
               "confident consensus error rather than a correction."))


# =============================================================================
def section_index(cost, dimensions, n_chunks):
    rule("5.  WHAT THIS COSTS, AND AT WHAT SCALE IT BREAKS")
    print(f"""
  Exact search compares the query against every vector: N x D multiply-adds.

  this document       {cost['vectors']:,} vectors x {dimensions:,} dimensions
                      = {cost['flops_per_query']:,.0f} operations per query
                      about {_duration(cost['seconds_per_query'])}, {cost['memory_mb']:.2f} MB in memory
""")
    print(f"  {'corpus size':<16}{'ops / query':>18}{'time':>12}{'memory':>12}")
    print("  " + "-" * 60)
    for row in scale_table(dimensions):
        print(f"  {row['vectors']:<16,}{row['flops_per_query']:>18,.0f}"
              f"{_duration(row['seconds_per_query']):>12}"
              f"{row['memory_mb']:>10,.0f} MB")

    print(wrap("\nAt this document's size, exact search is the correct choice "
               "and an approximate index would only add error for no gain. "
               "The numbers above are a scale indicator, not a benchmark -- "
               "real throughput varies by an order of magnitude with SIMD and "
               "memory layout. What they show reliably is the shape: linear in "
               "corpus size, so the wall arrives suddenly."))

    print(f"\n  {'index':<20}{'structure':<20}{'recall':<10}{'query cost'}")
    print("  " + "-" * 74)
    for name, structure, recall, cost_text, _note in INDEX_TYPES:
        print(f"  {name:<20}{structure:<20}{recall:<10}{cost_text}")
    print()
    for name, _s, _r, _c, note in INDEX_TYPES:
        print(wrap(f"{name}: {note}", indent="      "))

    print(wrap("\nEvery one of these except flat trades correctness for speed. "
               "That trade is invisible in testing: an approximate index "
               "returns plausible results, just not always the right ones, and "
               "nothing raises an error when it misses."))


# =============================================================================
def section_filtering(chunks, dense, top_k):
    rule("6.  THE METADATA FILTERING TRAP")
    if len(chunks) < 2:
        print(wrap("\nThis document produced a single chunk, so there is "
                   "nothing to filter between. The trap below needs a corpus "
                   "large enough for the filter to exclude something that "
                   "ranked highly."))
        return
    half = min(len(chunks) - 1, max(1, len(chunks) // 2))
    cutoff = chunks[half].metadata["start_page"]
    keep = [c.metadata["start_page"] >= cutoff for c in chunks]

    after = filter_after(dense, keep, top_k)
    before = filter_before(dense, keep, top_k)

    print(f"""
  Filter: only chunks from page {cutoff} onward. You asked for {top_k} results.

  post-filter   retrieve top {top_k}, then drop what fails   ->  {len(after)} results
  pre-filter    restrict candidates, then take top {top_k}   ->  {len(before)} results
""")
    if len(after) < len(before):
        print(wrap(f"Post-filtering returned {len(after)} instead of {top_k}. "
                   "Nothing failed and nothing warned: the filter was applied "
                   "to a list that had already been truncated, so chunks "
                   "matching your filter but ranked below the cut were never "
                   "candidates. Users see a thin answer and assume the "
                   "document is thin."))
    else:
        print(wrap("Both returned the same count here, which happens when "
                   "enough matching chunks rank highly. It is not a guarantee "
                   "-- change the filter or the query and post-filtering will "
                   "quietly under-return."))
    print(wrap("\nPre-filtering is correct and costs more, because the index "
               "cannot prune as aggressively; with some ANN structures it "
               "degrades toward a full scan. That cost is why post-filtering "
               "keeps getting shipped despite the bug."))


# =============================================================================
def section_evaluate(chunks, cases, embedder, vectors, bm25, top_k):
    rule("7.  MEASURED RETRIEVAL QUALITY")
    if not cases:
        print(wrap("\nNo golden set could be built: this document has no "
                   "detected outline, so there are no section headings to use "
                   "as queries. Evaluation needs either structure or "
                   "hand-written cases."))
        return None

    print(f"""
  {len(cases)} cases, built from this document's own section headings.
  Query = the heading. Correct answers = the chunks covering that section.
  Nothing was labelled by hand.
""")

    def dense_rank(query):
        return cosine_search(embedder.embed_query(query), vectors)

    def sparse_rank(query):
        return bm25.search(query)

    results = {}
    print(f"  {'method':<14}{'recall@' + str(top_k):>11}"
          f"{'precision@' + str(top_k):>14}{'MRR':>8}"
          f"{'nDCG@' + str(top_k):>9}{'hit rate':>11}")
    print("  " + "-" * 69)
    for label, fn in (("cosine", dense_rank), ("bm25", sparse_rank)):
        scores = goldset.evaluate(cases, fn, top_k)
        results[label] = scores
        print(f"  {label:<14}{scores['recall']:>11.0%}{scores['precision']:>14.0%}"
              f"{scores['mrr']:>8.3f}{scores['ndcg']:>9.3f}"
              f"{scores['hit_rate']:>11.0%}")
    print(wrap("\n  Read nDCG first when comparing two methods. Recall only "
               "asks whether the right chunks are present, so it saturates at "
               "100% and then stops moving however the ranking changes. nDCG "
               "keeps moving, because it discounts every correct chunk by how "
               "far down the list it sits.", 74))

    print("\n  Per case, by cosine:\n")
    print(f"    {'query':<34}{'first hit at':>14}{'recall':>9}")
    print("    " + "-" * 58)
    for row in goldset.per_case(cases, dense_rank, top_k):
        position = row["first_hit_rank"] or "never"
        print(f"    {row['query'][:32]:<34}{str(position):>14}{row['recall']:>9.0%}")

    print(wrap("\nThese numbers are optimistic and should be read as an upper "
               "bound. A section heading uses the document's own words, so "
               "there is no vocabulary mismatch -- and the heading text "
               "usually appears verbatim inside its own section, handing the "
               "keyword method a free exact match it would not get from a real "
               "user's phrasing."))
    print(wrap("\nWhat they are good for is comparison. The same bias applies "
               "to every method, so the ORDERING is meaningful even though the "
               "absolute values are inflated. Change chunk size, change "
               "backend, change K, and re-run: the direction of movement is "
               "trustworthy. Replace this with real logged queries the moment "
               "you have any."))
    return results


# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Retrieve chunks for a query and measure how well it worked.")
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--query", default=None,
                    help="the question (default: a section heading from the file)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--backend", choices=list(BACKENDS), default="tfidf")
    ap.add_argument("--dimensions", type=int, default=96)
    ap.add_argument("--strategy", default="recursive",
                    choices=["fixed", "recursive", "structural"])
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--evaluate", action="store_true",
                    help="build a golden set from the outline and score retrieval")
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
    state = {**state, "chunk_strategy": args.strategy, "chunk_size": args.size,
             "chunk_overlap": args.overlap,
             "section_boundaries": [n.char_start for n in tree.walk()
                                    if n.level > 0]}
    state = {**state, **chunk_node(state)}
    state = {**state, "embed_backend": args.backend,
             "embed_dimensions": args.dimensions}
    try:
        state = {**state, **embed_node(state)}
    except ImportError:
        sys.exit("the svd backend needs numpy:  pip install numpy")

    chunks, vectors = state["chunks"], state["vectors"]
    embedder = state["embedder"]
    bm25 = BM25(k1=args.k1, b=args.b).fit([c.page_content for c in chunks])

    cases = goldset.build(tree, chunks)
    if args.query:
        query, relevant = args.query, None
        for case in cases:
            if case["query"].lower() == args.query.lower():
                relevant = case["relevant"]
    elif cases:
        case = cases[len(cases) // 2]
        query, relevant = case["query"], case["relevant"]
    else:
        query, relevant = "summary", None

    dense = cosine_search(embedder.embed_query(query), vectors)
    sparse = bm25.search(query)
    cost = exact_search_cost(len(vectors), state["embed_settings"]["dimensions"])

    print(BAR)
    print("RETRIEVAL REPORT")
    print(BAR)
    print(f"\n  {state['pages'][0].metadata.get('source')}   {len(chunks)} chunks, "
          f"{args.backend}, {args.strategy} at {args.size}/{args.overlap}, K={args.top_k}")
    if not args.query and cases:
        print(wrap("\nNo --query given, so a section heading from your document "
                   "was used, which means the correct answers are known and "
                   "section 2 can be measured."))

    section_ranking(chunks, dense, sparse, query, args.top_k)
    section_topk(chunks, dense, relevant, args.top_k)
    section_bm25(chunks, bm25, sparse, query, args.k1, args.b)
    section_disagreement(chunks, dense, sparse, args.top_k,
                         args.backend, embedder.is_dense)
    section_index(cost, state["embed_settings"]["dimensions"], len(chunks))
    section_filtering(chunks, dense, args.top_k)

    results = None
    if args.evaluate:
        results = section_evaluate(chunks, cases, embedder, vectors, bm25,
                                   args.top_k)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        payload = {"query": query, "top_k": args.top_k,
                   "dense": [[chunks[i].metadata["chunk_id"], round(s, 6)]
                             for i, s in dense[:20]],
                   "sparse": [[chunks[i].metadata["chunk_id"], round(s, 6)]
                              for i, s in sparse[:20]],
                   "index_cost": cost, "evaluation": results}
        with open(os.path.join(args.out, "search.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  written to {args.out}/search.json")
    print()


if __name__ == "__main__":
    main()
