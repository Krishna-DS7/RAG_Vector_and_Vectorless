#!/usr/bin/env python3
"""
verify.py  --  turn every claim the code makes into an assertion

    python3 verify.py                  # against the built-in sample
    python3 verify.py mydoc.pdf        # structural checks against your own file

Run this after changing anything in shared/. The cleaning stages interact in
ways that are easy to break silently -- a reordering, a regex tweak, a new
replacement -- and none of those raise an exception on their own. They just
quietly change what the system can find.
"""

import sys

from pi_rag.parse import parse_node as tree_parse
from rag.chunk import chunk_node, chunk_structural
from rag.embed import (angle_degrees, cosine, dot, embed_node, euclidean,
                       l2_normalise, magnitude, tokenize)
from rag.parse import pages_for_span, parse_node as vector_parse, parse_per_page
from rag.search import (BM25, cosine_search, exact_search_cost, filter_after,
                        filter_before)
from rag.answer import assemble, check_support, estimate_tokens, order_chunks
from rag.rank import LexicalReranker, reciprocal_rank_fusion, rrf_contribution
from pi_rag.summarise import summarise_tree, summary_quality
from pi_rag.tree_search import context_from_leaves, pruned_branches, search
from shared import goldset
from shared.llm import get_llm
from shared.loader import load_node
from shared.measure import leaf_sections, mid_word_starts, ragged_ends

CHECKS = []


def check(name, sample_only=False):
    def wrap(fn):
        CHECKS.append((name, fn, sample_only))
        return fn
    return wrap


path = sys.argv[1] if len(sys.argv) > 1 else None
IS_SAMPLE = path is None

state = load_node({"path": path})
vstate = {**state, **vector_parse(state)}
tstate = {**state, **tree_parse(state)}

TEXT = vstate["document"].page_content
TREE = tstate["tree"]
REPORT = vstate["parse_report"]

BOUNDARIES = [n.char_start for n in TREE.walk() if n.level > 0]
CHUNK_SIZE, CHUNK_OVERLAP = 400, 60
cstate = {**vstate, "chunk_strategy": "recursive", "chunk_size": CHUNK_SIZE,
          "chunk_overlap": CHUNK_OVERLAP, "section_boundaries": BOUNDARIES}
cstate = {**cstate, **chunk_node(cstate)}
CHUNKS, SPANS = cstate["chunks"], cstate["chunk_spans"]

estate = {**cstate, **embed_node(cstate)}
EMBEDDER, VECTORS = estate["embedder"], estate["vectors"]

BM25_INDEX = BM25().fit([c.page_content for c in CHUNKS])
CASES = goldset.build(TREE, CHUNKS)

LLM = get_llm("extractive")
SUMMARY_STATS = summarise_tree(TREE, TEXT, LLM)
QUERY = CASES[len(CASES) // 2]["query"] if CASES else "summary"

# Section page ranges the sample document should produce.
SAMPLE_PAGES = {
    "A.1.1": (3, 4), "A.1.2": (5, 5), "A.2.1": (6, 7),
    "A.2.2": (8, 10), "A.3.1": (11, 11), "A.3.2": (12, 14),
}


# ---- universal: must hold for ANY document ---------------------------------
@check("both pipelines produce byte-identical cleaned text")
def _():
    assert TEXT == tstate["document"].page_content, "the shared cleaning diverged"
    return f"{len(TEXT):,} characters, identical"


@check("no ligature glyphs survive")
def _():
    for glyph in ("\ufb00", "\ufb01", "\ufb02", "\ufb03", "\ufb04"):
        assert glyph not in TEXT, f"ligature {glyph!r} left in the text"
    return "all five ligature code points normalised"


@check("no non-breaking spaces or zero-width characters survive")
def _():
    assert "\u00a0" not in TEXT and "\u200b" not in TEXT
    return "whitespace is ordinary"


@check("cleaning removed something but not everything")
def _():
    kept = len(TEXT) / max(state["raw_chars"], 1)
    assert 0.15 < kept < 1.01, f"kept {kept:.0%} of the input, which is suspicious"
    return f"kept {kept:.0%} of {state['raw_chars']:,} characters"


@check("page spans tile the document without gaps or overlap")
def _():
    spans = [(s, e) for s, e in sorted(vstate["page_spans"].values()) if e > s]
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end, f"page spans overlap at {end}/{start}"
    assert spans[0][0] == 0, "first page does not start at character 0"
    assert spans[-1][1] == len(TEXT), "last page does not reach the end"
    return f"{len(spans)} page spans, contiguous"


@check("every node's offsets slice out real text")
def _():
    nodes = [n for n in TREE.walk() if n.level > 0]
    if not nodes:
        return "no headings in this document, nothing to check"
    for node in nodes:
        assert 0 <= node.char_start < node.char_end <= len(TEXT), \
            f"node {node.node_id} has impossible offsets"
        extract = node.text(TEXT)
        assert extract.strip(), f"node {node.node_id} slices out nothing"
        assert node.title.split()[0] in extract[:120], \
            f"node {node.node_id} does not start at its own heading"
    return f"all {len(nodes)} nodes round-trip correctly"


@check("a child node always sits inside its parent")
def _():
    def descend(node):
        for child in node.nodes:
            assert node.char_start <= child.char_start, "child starts before parent"
            assert child.char_end <= node.char_end, "child ends after parent"
            descend(child)
    descend(TREE)
    return "containment holds throughout the tree"


@check("node ids are unique and sequential")
def _():
    ids = [n.node_id for n in TREE.walk()]
    assert ids == [f"{i:04d}" for i in range(len(ids))], "ids are not depth-first"
    return f"{len(ids)} ids, 0000 to {ids[-1]}"


@check("page ranges are consistent with character offsets")
def _():
    nodes = [n for n in TREE.walk() if n.level > 0]
    if not nodes:
        return "no headings, nothing to check"
    for node in nodes:
        first, last = pages_for_span(node.char_start, node.char_end,
                                     tstate["page_spans"])
        assert (node.start_index, node.end_index) == (first, last), \
            f"node {node.node_id} page range disagrees with its offsets"
    return f"all {len(nodes)} page ranges derived correctly"


@check("contents pages never become sections")
def _():
    numbers = [n.number for n in TREE.walk() if n.number]
    assert len(numbers) == len(set(numbers)), \
        f"duplicate section numbers, probably from a contents page: {numbers}"
    return f"{len(numbers)} sections, no duplicates"


@check("the vector pipeline records no hierarchy")
def _():
    assert "tree" not in vstate and "headings" not in vstate
    return "no tree, no headings -- the loss is real, not described"


# ---- chunking: must hold for ANY document ----------------------------------
@check("chunks cover every character of the document")
def _():
    covered = [False] * len(TEXT)
    for start, end in SPANS:
        for i in range(start, min(end, len(TEXT))):
            covered[i] = True
    missing = covered.count(False)
    assert missing == 0, f"{missing} characters fall in no chunk"
    return f"{len(SPANS)} chunks, nothing dropped"


@check("every chunk's text matches its own offsets")
def _():
    for chunk in CHUNKS:
        meta = chunk.metadata
        assert chunk.page_content == TEXT[meta["char_start"]:meta["char_end"]], \
            f"{meta['chunk_id']} text disagrees with its offsets"
    return f"all {len(CHUNKS)} chunks round-trip"


@check("chunks advance, never repeat or go backwards")
def _():
    for (a_start, _), (b_start, _) in zip(SPANS, SPANS[1:]):
        assert b_start > a_start, "a chunk starts at or before the previous one"
    return "strictly increasing start offsets"


@check("consecutive chunks actually overlap")
def _():
    if len(SPANS) < 2:
        return "only one chunk, nothing to overlap"
    gaps = [b_start - a_end for (_, a_end), (b_start, _) in zip(SPANS, SPANS[1:])]
    assert max(gaps) <= 0, f"a gap of {max(gaps)} characters between chunks"
    return f"overlap between all {len(SPANS) - 1} chunk pairs"


@check("chunk page ranges agree with the parser's span map")
def _():
    for chunk in CHUNKS:
        meta = chunk.metadata
        first, last = pages_for_span(meta["char_start"], meta["char_end"],
                                     vstate["page_spans"])
        assert (meta["start_page"], meta["end_page"]) == (first, last), \
            f"{meta['chunk_id']} page range is wrong"
    return "page attribution survived chunking"


@check("structural chunking never crosses a section boundary")
def _():
    if not BOUNDARIES:
        return "no outline in this document, nothing to check"
    spans = chunk_structural(TEXT, CHUNK_SIZE, CHUNK_OVERLAP, BOUNDARIES)
    inner = [b for b in BOUNDARIES if 0 < b < len(TEXT)]
    for start, end in spans:
        crossed = [b for b in inner if start < b < end]
        assert not crossed, f"chunk {start}-{end} crosses boundaries {crossed}"
    return f"{len(spans)} chunks, none straddling a section"


@check("recursive chunking beats fixed on cut quality")
def _():
    from rag.chunk import chunk_fixed, chunk_recursive
    fixed = chunk_fixed(TEXT, CHUNK_SIZE, CHUNK_OVERLAP)
    rec = chunk_recursive(TEXT, CHUNK_SIZE, CHUNK_OVERLAP)
    f_bad = len(ragged_ends(TEXT, fixed)) + len(mid_word_starts(TEXT, fixed))
    r_bad = len(ragged_ends(TEXT, rec)) + len(mid_word_starts(TEXT, rec))
    assert r_bad <= f_bad, f"recursive scored worse: {r_bad} vs fixed {f_bad}"
    return f"ragged+mid-word: fixed {f_bad}, recursive {r_bad}"


# ---- embedding: must hold for ANY document ---------------------------------
@check("every chunk produced a vector")
def _():
    assert len(VECTORS) == len(CHUNKS), "vector count does not match chunk count"
    empty = [c.metadata["chunk_id"] for c, v in zip(CHUNKS, VECTORS) if not v]
    assert not empty, f"chunks that embedded to nothing: {empty}"
    return f"{len(VECTORS)} vectors over {EMBEDDER.dimensions} dimensions"


@check("vectors are unit length")
def _():
    for chunk, vec in zip(CHUNKS, VECTORS):
        length = magnitude(vec)
        assert abs(length - 1.0) < 1e-9, \
            f"{chunk.metadata['chunk_id']} has length {length}"
    return "all normalised to 1.0"


@check("cosine equals the dot product for unit vectors")
def _():
    for i in range(min(len(VECTORS), 6)):
        for j in range(i + 1, min(len(VECTORS), 6)):
            a, b = VECTORS[i], VECTORS[j]
            assert abs(cosine(a, b) - dot(a, b)) < 1e-9, "they disagree"
    return "identical, which is why indexes store normalised vectors"


@check("cosine stays inside its valid range")
def _():
    for i, a in enumerate(VECTORS):
        assert abs(cosine(a, a) - 1.0) < 1e-9, "a vector is not identical to itself"
        for b in VECTORS[i + 1:]:
            score = cosine(a, b)
            assert -1.0 - 1e-9 <= score <= 1.0 + 1e-9, f"cosine out of range: {score}"
    return "self-similarity 1.0, all pairs within [-1, 1]"


@check("euclidean distance agrees with cosine on unit vectors")
def _():
    import math
    if len(VECTORS) < 2:
        return "only one vector"
    for i in range(min(len(VECTORS), 5)):
        for j in range(i + 1, min(len(VECTORS), 5)):
            a, b = VECTORS[i], VECTORS[j]
            expected = math.sqrt(max(0.0, 2 - 2 * cosine(a, b)))
            assert abs(euclidean(a, b) - expected) < 1e-9, \
                "euclidean and cosine disagree"
    return "distance = sqrt(2 - 2cos), so both rank identically"


@check("normalising twice changes nothing")
def _():
    once = VECTORS[0]
    twice = l2_normalise(once)
    assert all(abs(once[k] - twice[k]) < 1e-12 for k in once), "not idempotent"
    return "l2_normalise is idempotent"


@check("idf falls as a term appears in more chunks")
def _():
    vocab = EMBEDDER.vocab
    if vocab is None:
        return "dense backend, no vocabulary"
    rows = sorted((vocab.df[t], vocab.idf[t]) for t in vocab.terms)
    for (df_a, idf_a), (df_b, idf_b) in zip(rows, rows[1:]):
        if df_b > df_a:
            assert idf_b < idf_a, f"idf rose from df {df_a} to {df_b}"
    return "common terms are weighted down, as intended"


@check("stemming is idempotent")
def _():
    from rag.embed import _stem
    words = ["access", "bonus", "analysis", "leaves", "claiming", "policies",
             "records", "kept", "expenses", "authorisation", "always"]
    if EMBEDDER.vocab is not None:
        words += EMBEDDER.vocab.terms[:80]
    for word in words:
        once = _stem(word)
        assert _stem(once) == once, f"{word!r} -> {once!r} -> {_stem(once)!r}"
    return f"{len(words)} words stem to a fixed point"


@check("a query embeds into the same space")
def _():
    if EMBEDDER.vocab is None:
        return "dense backend"
    term = EMBEDDER.vocab.most_informative(1)[0][0]
    qvec = EMBEDDER.embed_query(term)
    assert qvec, f"query {term!r} embedded to nothing"
    assert abs(magnitude(qvec) - 1.0) < 1e-9, "query vector is not normalised"
    best = max(cosine(qvec, v) for v in VECTORS)
    assert best > 0, "a term from the vocabulary matched no chunk"
    return f"query {term!r} best match {best:.3f}"


@check("terms outside the vocabulary are reported, not silently kept")
def _():
    if EMBEDDER.vocab is None:
        return "dense backend"
    nonsense = "zzqqxx wibblefrotz"
    unknown = EMBEDDER.unknown_terms(nonsense)
    assert len(unknown) == 2, f"expected both terms unknown, got {unknown}"
    assert EMBEDDER.embed_query(nonsense) == {}, "unknown terms produced weight"
    return "unknown terms drop out and can be listed"


# ---- retrieval: must hold for ANY document ---------------------------------
@check("cosine search returns every chunk, sorted")
def _():
    query = EMBEDDER.vocab.most_informative(1)[0][0] if EMBEDDER.vocab else "the"
    ranking = cosine_search(EMBEDDER.embed_query(query), VECTORS)
    assert len(ranking) == len(VECTORS), "not every chunk was scored"
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True), "results are not sorted"
    assert len({i for i, _ in ranking}) == len(VECTORS), "duplicate chunk in results"
    return f"{len(ranking)} chunks ranked"


@check("a chunk retrieves itself first")
def _():
    misses = []
    for i, chunk in enumerate(CHUNKS):
        ranking = cosine_search(VECTORS[i], VECTORS, top_k=1)
        if ranking[0][0] != i:
            misses.append(chunk.metadata["chunk_id"])
    assert not misses, f"chunks that did not retrieve themselves: {misses}"
    return "every chunk is its own nearest neighbour"


@check("bm25 scores are non-negative and rank consistently")
def _():
    query = EMBEDDER.vocab.most_informative(1)[0][0] if EMBEDDER.vocab else "the"
    ranking = BM25_INDEX.search(query)
    assert len(ranking) == len(CHUNKS), "not every chunk was scored"
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True), "results are not sorted"
    assert min(scores) >= 0, f"negative bm25 score: {min(scores)}"
    return f"top score {scores[0]:.3f} for {query!r}"


@check("bm25 explain adds up to the score")
def _():
    query = EMBEDDER.vocab.most_informative(1)[0][0] if EMBEDDER.vocab else "the"
    top, score = BM25_INDEX.search(query, 1)[0]
    total = sum(c for _, _, _, c in BM25_INDEX.explain(query, top))
    assert abs(total - score) < 1e-9, f"parts sum to {total}, score is {score}"
    return "per-term contributions reconstruct the score exactly"


@check("bm25 saturation actually saturates")
def _():
    low = BM25(k1=0.1).fit([c.page_content for c in CHUNKS])
    high = BM25(k1=5.0).fit([c.page_content for c in CHUNKS])
    term = EMBEDDER.vocab.most_informative(1)[0][0] if EMBEDDER.vocab else "the"
    top = BM25_INDEX.search(term, 1)[0][0]
    assert low.score([term], top) <= high.score([term], top), \
        "raising k1 did not increase the score of a repeated term"
    return "higher k1 lets repeated terms count for more, as designed"


@check("a query with no known terms retrieves nothing")
def _():
    if EMBEDDER.vocab is None:
        return "dense backend"
    ranking = cosine_search(EMBEDDER.embed_query("zzqqxx wibblefrotz"), VECTORS)
    assert all(score == 0 for _, score in ranking), "nonsense matched something"
    assert all(s == 0 for _, s in BM25_INDEX.search("zzqqxx wibblefrotz"))
    return "both methods score zero rather than guessing"


@check("pre-filtering returns more results than post-filtering, or the same")
def _():
    query = EMBEDDER.vocab.most_informative(1)[0][0] if EMBEDDER.vocab else "the"
    ranking = cosine_search(EMBEDDER.embed_query(query), VECTORS)
    keep = [i % 2 == 0 for i in range(len(CHUNKS))]
    k = min(3, len(CHUNKS))
    after = filter_after(ranking, keep, k)
    before = filter_before(ranking, keep, k)
    assert len(before) >= len(after), "pre-filtering returned fewer results"
    assert len(before) == min(k, sum(keep)), "pre-filtering under-returned"
    return f"post {len(after)}, pre {len(before)} for k={k}"


@check("index cost scales linearly with corpus size")
def _():
    a = exact_search_cost(1_000, 128)
    b = exact_search_cost(10_000, 128)
    assert b["flops_per_query"] == 10 * a["flops_per_query"], "not linear"
    return "10x the vectors, 10x the work -- which is why indexes exist"


@check("the golden set is built from the outline, with real answers")
def _():
    if not CASES:
        return "no outline in this document, so no cases"
    for case in CASES:
        assert case["relevant"], f"case {case['query']!r} has no correct answer"
        assert all(0 <= i < len(CHUNKS) for i in case["relevant"]), \
            "a case points at a chunk that does not exist"
    return f"{len(CASES)} cases, every one answerable"


@check("retrieval metrics behave at their boundaries")
def _():
    if not CASES:
        return "no cases"
    case = CASES[0]
    perfect = [(i, 1.0) for i in case["relevant"]]
    assert goldset.recall_at_k(perfect, case["relevant"], len(perfect)) == 1.0
    assert goldset.reciprocal_rank(perfect, case["relevant"]) == 1.0
    empty = [(i, 0.0) for i in range(len(CHUNKS)) if i not in case["relevant"]]
    assert goldset.recall_at_k(empty, case["relevant"], 5) == 0.0
    assert goldset.reciprocal_rank(empty, case["relevant"]) == 0.0
    return "perfect ranking scores 1.0, ranking with no hits scores 0.0"


@check("a heading query retrieves its own section")
def _():
    if not CASES:
        return "no cases"
    def rank(query):
        return cosine_search(EMBEDDER.embed_query(query), VECTORS)
    scores = goldset.evaluate(CASES, rank, k=max(3, len(CHUNKS) // 2))
    assert scores["hit_rate"] > 0.5, \
        f"hit rate only {scores['hit_rate']:.0%} -- retrieval is broken"
    return (f"hit rate {scores['hit_rate']:.0%}, recall "
            f"{scores['recall']:.0%} over {scores['cases']} cases")


# ---- fusion and reranking: must hold for ANY document ----------------------
@check("fusion keeps every chunk both rankings contained")
def _():
    dense = cosine_search(EMBEDDER.embed_query(QUERY), VECTORS)
    sparse = BM25_INDEX.search(QUERY)
    fused = reciprocal_rank_fusion([dense, sparse])
    assert len(fused) == len(CHUNKS), "fusion lost or invented a chunk"
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True), "fused output is not sorted"
    return f"{len(fused)} chunks fused from two rankings"


@check("fusion is scale-free: multiplying scores changes nothing")
def _():
    dense = cosine_search(EMBEDDER.embed_query(QUERY), VECTORS)
    sparse = BM25_INDEX.search(QUERY)
    scaled = [(i, s * 1000) for i, s in sparse]
    a = reciprocal_rank_fusion([dense, sparse])
    b = reciprocal_rank_fusion([dense, scaled])
    assert [i for i, _ in a] == [i for i, _ in b], "scale affected the result"
    return "only positions matter, which is the point of RRF"


@check("fusing a ranking with itself preserves its order")
def _():
    dense = cosine_search(EMBEDDER.embed_query(QUERY), VECTORS)
    fused = reciprocal_rank_fusion([dense, dense])
    assert [i for i, _ in fused] == [i for i, _ in dense], "order changed"
    return "no spurious reordering when both inputs agree"


@check("rrf weights fall as rank falls")
def _():
    values = [rrf_contribution(r) for r in range(1, 12)]
    assert values == sorted(values, reverse=True), "not monotonically decreasing"
    assert values[0] > values[-1], "rank 1 does not beat rank 11"
    return f"rank 1 contributes {values[0]:.5f}, rank 11 {values[-1]:.5f}"


@check("reranking reorders without inventing or dropping chunks")
def _():
    dense = cosine_search(EMBEDDER.embed_query(QUERY), VECTORS)
    width = min(10, len(CHUNKS))
    shortlist = [(i, CHUNKS[i].page_content) for i, _ in dense[:width]]
    reranked = LexicalReranker().rerank(QUERY, shortlist)
    assert len(reranked) == width, "reranking changed the shortlist size"
    assert {i for i, _ in reranked} == {i for i, _ in shortlist}, \
        "reranking introduced a chunk that was not shortlisted"
    return f"{width} candidates reordered, membership unchanged"


# ---- assembly and answering ------------------------------------------------
@check("assembly respects the token budget")
def _():
    for budget in (200, 600, 4000):
        built = assemble(QUERY, CHUNKS, budget_tokens=budget)
        assert built["prompt_tokens"] <= budget + 60, \
            f"budget {budget} exceeded: {built['prompt_tokens']} tokens"
        assert len(built["admitted"]) + len(built["dropped"]) == len(CHUNKS), \
            "a chunk was neither admitted nor reported as dropped"
    return "budgets of 200, 600 and 4000 tokens all respected"


@check("a tight budget drops chunks rather than truncating them")
def _():
    # Size the budget relative to the content. A fixed small number assumes
    # the document is big enough to overflow it, which is false for a
    # single-paragraph file and made this test fail on degenerate input.
    total = sum(estimate_tokens(c.page_content) for c in CHUNKS)
    if total < 80:
        return "document fits any budget; nothing to drop"
    built = assemble(QUERY, CHUNKS, budget_tokens=max(60, total // 2))
    assert built["dropped"], f"nothing dropped at half of {total} tokens"
    for chunk in built["admitted"]:
        # assemble collapses whitespace when laying out the context, so
        # compare normalised text. Comparing raw text reports a truncation
        # that did not happen.
        body = " ".join(chunk.page_content.split())
        assert body in built["context"], "a chunk was truncated"
    return f"{len(built['dropped'])} dropped whole, none cut in half"


@check("every ordering keeps the same chunks")
def _():
    for strategy in ("best_first", "best_last", "edges", "document"):
        ordered = order_chunks(CHUNKS, strategy)
        assert len(ordered) == len(CHUNKS), f"{strategy} changed the count"
        assert {id(c) for c in ordered} == {id(c) for c in CHUNKS}, \
            f"{strategy} changed which chunks are present"
    starts = [c.metadata["char_start"] for c in order_chunks(CHUNKS, "document")]
    assert starts == sorted(starts), "document order is not in source order"
    return "best_first, best_last, edges and document reorder only"


@check("every admitted chunk is citable from the prompt")
def _():
    built = assemble(QUERY, CHUNKS, budget_tokens=4000)
    for chunk in built["admitted"]:
        marker = chunk.metadata["chunk_id"]
        assert marker in built["prompt"], f"{marker} has no source marker"
    return "every chunk carries its id and page into the prompt"


@check("the rules are stated after the context, not only before it")
def _():
    from rag.answer import ROLE_PROMPT, SYSTEM_PROMPT
    built = assemble(QUERY, CHUNKS, budget_tokens=4000)
    prompt = built["prompt"]
    assert prompt.startswith(ROLE_PROMPT), "the role line is not first"
    assert prompt.index(SYSTEM_PROMPT) > prompt.index(built["context"][:40]), \
        "the rules sit above the context they govern"
    return "role, then context, then rules, then the question"


@check("an invented citation is caught")
def _():
    from rag.answer import validate_citations
    sent = [c.metadata["chunk_id"] for c in CHUNKS[:2]]
    good = validate_citations(f"The limit applies [{sent[0]}, p3].", sent)
    assert good["valid"], f"a real citation was rejected: {good['unknown']}"
    bad = validate_citations("The limit applies [c9999, p3].", sent)
    assert not bad["valid"], "a citation naming a chunk never sent passed"
    assert "c9999" in bad["unknown"], "the invented id was not reported"
    return "markers resolved against what was actually sent"


@check("nDCG separates ranking quality from mere membership")
def _():
    from shared.goldset import ndcg_at_k, recall_at_k
    top = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]
    low = [(2, 0.9), (3, 0.8), (0, 0.7), (1, 0.6)]
    relevant = [0, 1]
    assert recall_at_k(top, relevant, 4) == recall_at_k(low, relevant, 4), \
        "this pair was chosen because recall cannot tell them apart"
    assert ndcg_at_k(top, relevant, 4) > ndcg_at_k(low, relevant, 4), \
        "nDCG did not reward the better ranking"
    assert abs(ndcg_at_k(top, relevant, 4) - 1.0) < 1e-9, \
        "a perfect ranking did not score 1.0"
    return (f"same recall, nDCG {ndcg_at_k(top, relevant, 4):.2f} "
            f"vs {ndcg_at_k(low, relevant, 4):.2f}")


@check("citation markers are not counted as unsupported claims")
def _():
    fake = "[c0000, p1] The policy applies to all staff."
    support = check_support(fake, [c.page_content for c in CHUNKS[:1]])
    assert "c0000" not in support["unsupported"], \
        "a citation marker was treated as fabricated vocabulary"
    return "markers stripped before the grounding check"


# ---- summarisation and tree search -----------------------------------------
@check("every node has a summary after summarisation")
def _():
    missing = [n.node_id for n in TREE.walk() if not (n.summary or "").strip()]
    assert not missing, f"nodes without a summary: {missing}"
    quality = summary_quality(TREE)
    assert quality["empty"] == 0
    return (f"{SUMMARY_STATS['calls']} calls "
            f"({SUMMARY_STATS['leaves']} leaves, {SUMMARY_STATS['parents']} parents)")


@check("summarisation is bottom-up: children before parents")
def _():
    order = []
    summarise_tree(TREE, TEXT, LLM, on_node=lambda n: order.append(n.node_id))
    position = {node_id: i for i, node_id in enumerate(order)}
    for node in TREE.walk():
        for child in node.nodes:
            assert position[child.node_id] < position[node.node_id], \
                f"parent {node.node_id} summarised before child {child.node_id}"
    return "a parent is only summarised once its children are"


@check("tree search reaches leaves and records its path")
def _():
    if not [n for n in TREE.walk() if n.is_leaf and n.level > 0]:
        return "no outline in this document"
    result = search(TREE, QUERY, LLM, beam=2)
    assert result["leaves"], "search returned no sections"
    assert all(n.is_leaf for n in result["leaves"]), "a non-leaf was returned"
    assert result["trace"], "no traversal was recorded"
    # Per hop the beam may legitimately choose nothing, because the level's
    # beam can fill from a different node. Per level it must choose something,
    # or the descent stopped without reaching anything.
    by_level = {}
    for hop in result["trace"]:
        by_level.setdefault(hop["hop"], []).extend(hop["chose"])
    for level, chosen in by_level.items():
        assert chosen, f"level {level} chose nothing"
    return (f"{result['hops']} hops, {result['llm_calls']} decisions, "
            f"{len(result['leaves'])} sections")


@check("a wider beam never returns fewer sections")
def _():
    if not [n for n in TREE.walk() if n.is_leaf and n.level > 0]:
        return "no outline"
    narrow = search(TREE, QUERY, LLM, beam=1)
    wide = search(TREE, QUERY, LLM, beam=3)
    assert len(wide["leaves"]) >= len(narrow["leaves"]), \
        "beam 3 returned fewer sections than beam 1"
    return (f"beam 1 -> {len(narrow['leaves'])} sections, "
            f"beam 3 -> {len(wide['leaves'])}")


@check("the beam bounds the whole level, not each node")
def _():
    # The sample outline is too shallow to tell a level-wide beam from a
    # per-node one, so this builds a tree deep enough for the difference to
    # show: per-node, width 3 over 5 levels reaches 3**5 = 243 nodes.
    from pi_rag.node import TreeNode, assign_node_ids

    def build(depth, path="r"):
        node = TreeNode(title=f"Section {path} retention of records",
                        level=len(path) - 1)
        node.summary = "how long records are retained and kept"
        if depth:
            node.nodes = [build(depth - 1, path + str(i)) for i in range(3)]
        return node

    tree = assign_node_ids(build(5))
    beam, max_depth = 3, 6
    result = search(tree, "how long are records kept", LLM,
                    beam=beam, max_depth=max_depth)

    ceiling = beam * max_depth
    assert result["llm_calls"] <= ceiling, \
        f"{result['llm_calls']} model calls for one question (cap {ceiling})"
    assert len(result["leaves"]) <= ceiling, \
        f"{len(result['leaves'])} sections returned (cap {ceiling})"
    # A branch dropped by the beam is still reported, and distinguishable.
    pruned = pruned_branches(result["trace"])
    assert any(branch.get("cut_by_beam") for branch in pruned), \
        "a beam-limited search reported no beam cuts"
    return (f"depth 5, beam 3: {result['llm_calls']} calls, "
            f"{len(result['leaves'])} sections (per-node would be 243)")


@check("pruned branches are reported, not hidden")
def _():
    if not [n for n in TREE.walk() if n.is_leaf and n.level > 0]:
        return "no outline"
    result = search(TREE, QUERY, LLM, beam=1)
    pruned = pruned_branches(result["trace"])
    considered = sum(len(hop["considered"]) for hop in result["trace"])
    chosen = sum(len(hop["chose"]) for hop in result["trace"])
    assert len(pruned) == considered - chosen, "the pruned list does not add up"
    return f"{len(pruned)} branches declined, all recorded"


@check("a full budget is spent on the most relevant sections")
def _():
    if not [n for n in TREE.walk() if n.is_leaf and n.level > 0]:
        return "no outline"
    leaves = [n for n in TREE.walk() if n.is_leaf and n.level > 0]
    if len(leaves) < 2:
        return "only one section; nothing to prioritise"
    # A budget big enough for one section only. Traversal order would admit
    # whichever section the walk happened to reach first.
    from rag.answer import estimate_tokens
    one = max(estimate_tokens(" ".join(n.text(TEXT).split())) + 12
              for n in leaves)
    built = context_from_leaves(list(reversed(leaves)), TEXT,
                                budget_tokens=one, question=QUERY)
    assert built["admitted"], "nothing was admitted at a one-section budget"
    from rag.rank import LexicalReranker
    scorer = LexicalReranker()
    best = max(leaves, key=lambda n: scorer.score(QUERY, n.text(TEXT)))
    assert built["admitted"][0].node_id == best.node_id, \
        "the budget went to a section that was not the best match"
    return f"{len(built['dropped'])} dropped, the best match kept"


@check("an out-of-range choice cannot crash the descent")
def _():
    class Rogue:
        """Returns ids a real model invents: out of range, duplicated, wrong type."""
        is_real_model = False
        def summarise(self, text, title="", max_words=45): return title
        def answer(self, question, context, max_sentences=3): return "x"
        def choose(self, question, options, max_pick=2):
            return [99, -1, "two", 0, 0]

    if not [n for n in TREE.walk() if n.is_leaf and n.level > 0]:
        return "no outline"
    result = search(TREE, QUERY, Rogue(), beam=2)
    assert result["leaves"], "the descent returned nothing"
    for hop in result["trace"]:
        assert len(hop["chose"]) == len(set(hop["chose"])), \
            "a branch was opened twice from one invalid reply"
    return "invalid ids dropped, duplicates collapsed, descent survived"


@check("sections are returned whole, never cut")
def _():
    if not [n for n in TREE.walk() if n.is_leaf and n.level > 0]:
        return "no outline"
    result = search(TREE, QUERY, LLM, beam=2)
    built = context_from_leaves(result["leaves"], TEXT, budget_tokens=4000)
    for node in built["admitted"]:
        body = " ".join(node.text(TEXT).split())
        assert body in built["context"], f"section {node.node_id} was truncated"
    return f"{len(built['admitted'])} sections, each intact in the context"


@check("both pipelines answer end to end")
def _():
    from pi_rag.pipeline import build_query_graph as tree_query
    from rag.pipeline import build_graph as vector_index
    from rag.pipeline import build_query_graph as vector_query

    indexed = vector_index().invoke({})
    vector = vector_query().invoke({**indexed, "question": QUERY,
                                    "top_k": 3, "llm": LLM})
    assert vector["answer"].strip(), "the vector pipeline produced no answer"
    assert vector["trace"][-5:] == ["retrieve", "fuse", "rerank",
                                    "assemble", "generate"]

    if [n for n in TREE.walk() if n.is_leaf and n.level > 0]:
        tree_state = {**estate, "tree": TREE, "summarised_tree": TREE,
                      "question": QUERY, "llm": LLM}
        answer = tree_query().invoke(tree_state)
        assert answer["answer"].strip(), "the tree pipeline produced no answer"
    return "question in, answer out, both architectures"


# ---- sample-only: exact expected values ------------------------------------
@check("ambiguous hyphen kept, ordinary ones joined", sample_only=True)
def _():
    assert "one-time" in TEXT and "onetime" not in TEXT
    assert "calendar year" in TEXT and "yourself" in TEXT
    return "one-time kept; calendar and yourself rejoined"


@check("cross-page sentence rejoined", sample_only=True)
def _():
    assert "NOT claimed through this process" in TEXT
    assert REPORT["page_joins"] == [10], REPORT["page_joins"]
    return "the sentence split across pages 9/10 is whole"


@check("strategy A really does break that sentence", sample_only=True)
def _():
    per_page, _ = parse_per_page(state["pages"])
    p9 = next(d for d in per_page if d.metadata["page"] == 9)
    p10 = next(d for d in per_page if d.metadata["page"] == 10)
    assert p9.page_content.rstrip().endswith("through this")
    assert p10.page_content.lstrip().startswith("process;")
    assert "NOT claimed through this process" not in p9.page_content
    return "per-page parsing splits it; merged parsing does not"


@check("page furniture removed", sample_only=True)
def _():
    assert "Confidential" not in TEXT
    assert TEXT.count("Acme Robotics - Employee Policy Handbook") == 0
    return "header and footer gone from all 14 pages"


@check("chunking splits sections the outline keeps whole", sample_only=True)
def _():
    from shared.measure import section_integrity
    integrity = section_integrity(TREE, SPANS)
    assert integrity["split"], "expected some sections to be split at size 400"
    leaves = leaf_sections(TREE)
    assert len(leaves) == 6, len(leaves)
    return (f"{len(integrity['split'])} of {len(leaves)} sections split at "
            f"size {CHUNK_SIZE} -- the damage this step causes")


@check("expected outline, with expected page ranges", sample_only=True)
def _():
    got = {n.number: (n.start_index, n.end_index) for n in TREE.walk() if n.number}
    for number, expected in SAMPLE_PAGES.items():
        assert got.get(number) == expected, \
            f"{number}: got {got.get(number)}, expected {expected}"
    assert REPORT["scheme"] == "dotted_alpha", REPORT["scheme"]
    return f"{len(SAMPLE_PAGES)} sections at the right pages, scheme dotted_alpha"


if __name__ == "__main__":
    label = path or "built-in sample"
    print("=" * 72)
    print(f"VERIFYING: {label}")
    print("=" * 72 + "\n")

    failed = skipped = 0
    for name, fn, sample_only in CHECKS:
        if sample_only and not IS_SAMPLE:
            skipped += 1
            continue
        try:
            detail = fn()
            print(f"  PASS  {name}")
            if detail:
                print(f"        {detail}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}")
            print(f"        {exc}")

    ran = len(CHECKS) - skipped
    print("\n" + "=" * 72)
    print(f"  {ran - failed}/{ran} passed"
          + (f", {skipped} skipped (sample-only)" if skipped else ""))
    print("=" * 72)
    raise SystemExit(1 if failed else 0)
