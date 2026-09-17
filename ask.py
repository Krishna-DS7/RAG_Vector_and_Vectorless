#!/usr/bin/env python3
"""
ask.py  --  ask a document a question, both ways, and compare

    python3 ask.py report.pdf --query "how long are records kept"
    python3 ask.py                                  # built-in sample
    python3 ask.py doc.pdf --pipeline vector
    python3 ask.py doc.pdf --pipeline tree
    python3 ask.py doc.pdf --show-prompt            # the literal prompt sent
    python3 ask.py doc.pdf --beam 3 --top-k 5
    python3 ask.py doc.pdf --out results/

This is the whole system end to end: question in, answer out, twice over.

    vector    retrieve by similarity and keyword, fuse, rerank, assemble
              a prompt inside a token budget, answer

    tree      summarise every section, navigate the outline hop by hop,
              return the chosen sections whole, answer

Without a real language model, summaries, branch choices and answers come from
a deterministic extractive stand-in. The mechanics are exact; the quality is
not. Every affected line says so.
"""

import argparse
import json
import os
import sys

from pi_rag.parse import parse_node as tree_parse
from pi_rag.summarise import summarise_node
from pi_rag.tree_search import (context_from_leaves, pruned_branches, search)
from rag.answer import (assemble, check_support, find_citations,
                        validate_citations)
from rag.chunk import chunk_node
from rag.embed import BACKENDS, embed_node
from rag.parse import parse_node as vector_parse
from rag.rank import LexicalReranker, RRF_K, reciprocal_rank_fusion
from rag.search import BM25, cosine_search
from shared import goldset
from shared.llm import get_llm, warn_if_stub
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


def block(text, indent="      ", limit=None):
    body = text if not limit or len(text) <= limit else text[:limit] + " ..."
    return "\n".join(indent + "| " + l for l in body.split("\n"))


def cid(chunk):
    return chunk.metadata["chunk_id"]


# =============================================================================
def run_vector(state, question, top_k, rerank_width, budget, ordering, llm):
    chunks, vectors = state["chunks"], state["vectors"]
    embedder, bm25 = state["embedder"], state["bm25"]

    dense = cosine_search(embedder.embed_query(question), vectors)
    sparse = bm25.search(question)
    fused = reciprocal_rank_fusion([dense, sparse])

    shortlist = [(i, chunks[i].page_content) for i, _ in fused[:rerank_width]]
    reranked = LexicalReranker().rerank(question, shortlist)

    picked = [chunks[i] for i, _ in reranked[:top_k]]
    built = assemble(question, picked, budget, ordering)
    answer = llm.answer(question, built["context"])

    return {"dense": dense, "sparse": sparse, "fused": fused,
            "reranked": reranked, "picked": picked, "assembly": built,
            "answer": answer,
            "support": check_support(
                answer, [c.page_content for c in built["admitted"]]),
            "citations": find_citations(answer),
            "citation_check": validate_citations(
                answer,
                [c.metadata.get("chunk_id") for c in built["admitted"]])}


def run_tree(state, question, beam, budget, llm):
    tree = state["summarised_tree"]
    result = search(tree, question, llm, beam=beam)
    built = context_from_leaves(result["leaves"],
                                state["document"].page_content, budget,
                                question=question)
    answer = llm.answer(question, built["context"])
    texts = [node.text(state["document"].page_content)
             for node in built["admitted"]]

    return {"result": result, "assembly": built, "answer": answer,
            "support": check_support(answer, texts),
            "citations": find_citations(answer),
            "citation_check": validate_citations(
                answer, [node.node_id for node in built["admitted"]]),
            "pruned": pruned_branches(result["trace"])}


# =============================================================================
def show_vector(out, chunks, question, top_k, show_prompt):
    rule("VECTOR PIPELINE")
    dense_rank = {i: r for r, (i, _) in enumerate(out["dense"], 1)}
    sparse_rank = {i: r for r, (i, _) in enumerate(out["sparse"], 1)}
    fused_rank = {i: r for r, (i, _) in enumerate(out["fused"], 1)}
    rerank_rank = {i: r for r, (i, _) in enumerate(out["reranked"], 1)}

    print(f"\n  How the ranking moved through the stages:\n")
    print(f"  {'chunk':<9}{'cosine':>8}{'bm25':>7}{'fused':>7}{'reranked':>10}"
          f"{'pages':>9}   opening words")
    print("  " + "-" * 74)
    for i, _ in out["reranked"][:8]:
        chunk = chunks[i]
        meta = chunk.metadata
        pages = (f"p{meta['start_page']}" if meta["start_page"] == meta["end_page"]
                 else f"p{meta['start_page']}-{meta['end_page']}")
        head = " ".join(chunk.page_content.split())[:26]
        print(f"  {cid(chunk):<9}{dense_rank[i]:>8}{sparse_rank[i]:>7}"
              f"{fused_rank[i]:>7}{rerank_rank[i]:>10}{pages:>9}   {head}")

    moved = [cid(chunks[i]) for i, _ in out["reranked"][:top_k]
             if fused_rank[i] > top_k]
    if moved:
        print(wrap(f"\nReranking promoted {', '.join(moved)} into the top "
                   f"{top_k} from below the cut. Retrieval had already found "
                   "them; nothing before this stage read them alongside the "
                   "question."))
    else:
        print(wrap(f"\nReranking did not change which chunks make the top "
                   f"{top_k} here, only their order. That happens when "
                   "retrieval was already right -- reranking is insurance, "
                   "and insurance mostly does nothing."))

    a = out["assembly"]
    print(f"""
  context     {a['context_tokens']} tokens of a {a['budget_tokens']} budget, {a['ordering']} ordering
  admitted    {len(a['admitted'])} chunks
  dropped     {len(a['dropped'])} chunks{' -- they did not fit' if a['dropped'] else ''}""")
    if a["dropped"]:
        print(wrap("Those chunks were retrieved and then discarded for space. "
                   "Nothing warns about this in most implementations: the "
                   "answer is generated from what fitted, and reads exactly as "
                   "confident as if nothing was missing.", indent="      "))

    if show_prompt:
        print("\n  The literal prompt:\n")
        print(block(a["prompt"], limit=1400))


def show_tree(out, question, show_prompt, beam, llm):
    rule("TREE PIPELINE")
    result = out["result"]
    hops, calls = result["hops"], result["llm_calls"]
    print(f"\n  {hops} hop{'s' if hops != 1 else ''}, "
          f"{calls} decision{'s' if calls != 1 else ''}.\n")

    for hop in result["trace"]:
        print(f"  hop {hop['hop']}  at {hop['at'][:52]}")
        for option in hop["considered"]:
            mark = "-->" if option["chosen"] else "   "
            print(f"    {mark} {option['node_id']}  {option['title'][:40]:<42}"
                  f"{option['pages']:>9}")
            if option["summary"]:
                print(f"          {option['summary'][:66]}")
        print()

    if out["pruned"]:
        print("  Branches not opened:\n")
        for option in out["pruned"][:5]:
            print(f"      {option['node_id']}  {option['title'][:44]:<46}"
                  f"{option['pages']}")
        print(wrap("Each was declined on the strength of its summary alone. "
                   "If the answer was behind one of them, nothing later "
                   "recovers it, and the answer will be just as confident."))

    # Did the search use the beam it was given?
    narrow = [hop for hop in result["trace"] if len(hop["chose"]) < beam]
    if narrow and beam > 1:
        print(wrap(f"At {len(narrow)} of {len(result['trace'])} hops the search "
                   f"chose fewer than the beam of {beam} allowed, because "
                   "nothing else scored above zero."))
        if not llm.is_real_model:
            print(wrap("With the extractive stand-in that is a vocabulary "
                       "limit, not a judgement: a branch only scores if it "
                       "shares words with the question. A section on retention "
                       "saying 'retained for seven years' scores nothing "
                       "against a question asking how long records are 'kept'. "
                       "This is exactly where a question spanning two sections "
                       "loses one of them, and exactly what a real model is "
                       "for.", indent="      "))

    a = out["assembly"]
    print(f"""
  sections returned   {len(a['admitted'])}, whole
  context             {a['context_tokens']} tokens
  dropped             {len(a['dropped'])}{' -- a whole section did not fit' if a['dropped'] else ''}""")
    if a["dropped"]:
        print(wrap("A section either fits or it does not; there is no "
                   "equivalent of taking its best half. That is the budget "
                   "cost of returning whole units.", indent="      "))


def show_answers(vector_out, tree_out, question, llm):
    rule("ANSWERS")
    print(f"\n  question: {question!r}\n")
    for label, out in (("VECTOR", vector_out), ("TREE", tree_out)):
        if not out:
            continue
        print(f"  {label}")
        print(block(out["answer"], indent="      "))
        support = out["support"]
        print(f"\n      grounding   {support['coverage']:.0%} of the answer's "
              f"words appear in the context")
        if support["unsupported"][:6]:
            print(f"      not in context: {', '.join(support['unsupported'][:6])}")
        print()

    if not llm.is_real_model:
        print(wrap("Both answers came from the extractive stand-in, which "
                   "selects sentences rather than writing them. It cannot "
                   "synthesise across two sources, cannot refuse, and cannot "
                   "resolve a contradiction -- so judge the retrieval above, "
                   "not the prose here."))
    print(wrap("\nThe grounding figure is a crude check: it asks whether the "
               "answer's vocabulary appears in the context, which catches an "
               "answer about things the context never mentions and nothing "
               "subtler. A fabricated claim assembled from context words "
               "scores high. Real faithfulness measurement asks a judge model "
               "whether each claim is entailed."))


def show_comparison(vector_out, tree_out, state, llm, top_k, beam):
    rule("HEAD TO HEAD")
    v, t = vector_out, tree_out
    tree = state["summarised_tree"]
    leaves = [n for n in tree.walk() if n.is_leaf and n.level > 0]

    def row(label, a, b):
        print(f"  {label:<26}{a:<24}{b}")

    print()
    row("", "vector", "tree")
    print("  " + "-" * 74)
    row("unit returned", "chunks", "whole sections")
    n_sections = len(t["assembly"]["admitted"])
    row("results", f"{len(v['assembly']['admitted'])} chunks",
        f"{n_sections} section" + ("s" if n_sections != 1 else ""))
    row("context tokens", str(v["assembly"]["context_tokens"]),
        str(t["assembly"]["context_tokens"]))
    row("model calls at query", "1 (the answer)",
        f"{t['result']['llm_calls'] + 1} ({t['result']['llm_calls']} + answer)")
    row("model calls at index", "0", f"{state['summary_stats']['calls']}")
    row("citations available", "chunk id and page", "node id and page range")
    row("decision is auditable", "no, a score", "yes, a path")
    row("needs structure", "no", f"yes -- {len(leaves)} sections here")

    print(wrap("\nThe shapes of the two failure modes are different, and that "
               "matters more than which won here. Vector retrieval fails by "
               "returning a fragment that reads as complete. Tree search fails "
               "by pruning a branch it should have opened. The first is "
               "detectable by reading the context; the second leaves no trace "
               "in the output at all."))
    print(wrap("\nCost follows the same split. Vector is expensive once, at "
               "indexing, and nearly free per question. Tree is expensive at "
               "indexing AND at every question, because navigation is model "
               "calls all the way down. At high query volume that difference "
               "compounds into the dominant term, which is why the practical "
               "answer is usually both, with a router deciding which questions "
               "deserve the expensive path."))


def show_cost(state, tree_out, llm, queries_per_month=100_000):
    rule("COST SHAPE")
    tree = state["summarised_tree"]
    n_chunks = len(state["chunks"])
    n_nodes = len(list(tree.walk()))
    hops = tree_out["result"]["llm_calls"] if tree_out else 0

    print(f"""
  measured on this document

    chunks to embed                 {n_chunks}
    nodes to summarise              {n_nodes}
    model calls per tree question   {hops + 1}
    model calls per vector question 1
""")
    print(f"  Extrapolated to {queries_per_month:,} questions a month, the shape "
          f"is:\n")
    print(f"    {'':<14}{'indexing calls':>18}{'calls per month':>20}")
    print("    " + "-" * 52)
    print(f"    {'vector':<14}{'0':>18}{queries_per_month:>20,}")
    print(f"    {'tree':<14}{n_nodes:>18,}"
          f"{(hops + 1) * queries_per_month:>20,}")
    print(wrap("\nNo prices here, because they change monthly and a stale "
               "number is worse than none. The ratio is what matters and it "
               "does not change: tree navigation costs a multiple of vector "
               "retrieval per question, and that multiple is roughly the "
               "number of hops. Whether it is worth paying depends entirely "
               "on whether the questions need whole sections."))


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Ask a document a question, both ways.")
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--query", default=None,
                    help="the question (default: a section heading from the file)")
    ap.add_argument("--pipeline", choices=["both", "vector", "tree"], default="both")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--rerank-width", type=int, default=50)
    ap.add_argument("--budget", type=int, default=1200, metavar="TOKENS")
    ap.add_argument("--ordering",
                    choices=["best_first", "best_last", "edges", "document"],
                    default="best_first")
    ap.add_argument("--backend", choices=list(BACKENDS), default="tfidf")
    ap.add_argument("--strategy", default="recursive",
                    choices=["fixed", "recursive", "structural"])
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--llm", default="extractive")
    ap.add_argument("--show-prompt", action="store_true")
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

    llm = get_llm(args.llm)

    state = {**state, **vector_parse(state)}
    tree_state = tree_parse(state)
    tree = tree_state["tree"]
    state = {**state, "tree": tree,
             "chunk_strategy": args.strategy, "chunk_size": args.size,
             "chunk_overlap": args.overlap,
             "section_boundaries": [n.char_start for n in tree.walk()
                                    if n.level > 0]}
    state = {**state, **chunk_node(state)}
    state = {**state, "embed_backend": args.backend}
    try:
        state = {**state, **embed_node(state)}
    except ImportError:
        sys.exit("the svd backend needs numpy:  pip install numpy")
    state["bm25"] = BM25().fit([c.page_content for c in state["chunks"]])
    state["llm"] = llm

    cases = goldset.build(tree, state["chunks"])
    if args.query:
        question = args.query
    elif cases:
        question = cases[len(cases) // 2]["query"]
    else:
        question = "what is this document about"

    print(BAR)
    print("QUESTION ANSWERING REPORT")
    print(BAR)
    print(f"\n  {state['pages'][0].metadata.get('source')}   "
          f"{len(state['chunks'])} chunks, {len(list(tree.walk()))} nodes")

    note = warn_if_stub(llm)
    if note:
        print()
        print(wrap(note))

    state = {**state, **summarise_node(state)}

    vector_out = tree_out = None
    if args.pipeline in ("both", "vector"):
        vector_out = run_vector(state, question, args.top_k, args.rerank_width,
                                args.budget, args.ordering, llm)
        show_vector(vector_out, state["chunks"], question, args.top_k,
                    args.show_prompt)

    if args.pipeline in ("both", "tree"):
        leaves = [n for n in tree.walk() if n.is_leaf and n.level > 0]
        if not leaves:
            rule("TREE PIPELINE")
            print(wrap("\nThis document has no detected outline, so there is "
                       "nothing to navigate and the tree pipeline cannot run. "
                       "That is the honest boundary of the architecture, not a "
                       "bug -- run parse.py for what to check."))
        else:
            tree_out = run_tree(state, question, args.beam, args.budget, llm)
            show_tree(tree_out, question, args.show_prompt,
                      args.beam, llm)

    show_answers(vector_out, tree_out, question, llm)

    if vector_out and tree_out:
        show_comparison(vector_out, tree_out, state, llm, args.top_k, args.beam)
        show_cost(state, tree_out, llm)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        payload = {"question": question, "llm": llm.name,
                   "real_model": llm.is_real_model}
        if vector_out:
            payload["vector"] = {
                "answer": vector_out["answer"],
                "chunks": [cid(c) for c in vector_out["assembly"]["admitted"]],
                "context_tokens": vector_out["assembly"]["context_tokens"],
                "grounding": vector_out["support"]["coverage"]}
        if tree_out:
            payload["tree"] = {
                "answer": tree_out["answer"],
                "sections": [n.title for n in tree_out["assembly"]["admitted"]],
                "trace": tree_out["result"]["trace"],
                "context_tokens": tree_out["assembly"]["context_tokens"],
                "grounding": tree_out["support"]["coverage"]}
        with open(os.path.join(args.out, "answer.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  written to {args.out}/answer.json")
    print()


if __name__ == "__main__":
    main()
