#!/usr/bin/env python3
"""
chunk.py  --  cut a document into chunks and measure what that cost

    python3 chunk.py report.pdf
    python3 chunk.py                                 # built-in sample
    python3 chunk.py doc.pdf --size 500 --overlap 50
    python3 chunk.py doc.pdf --strategy fixed
    python3 chunk.py doc.pdf --compare               # all three, side by side
    python3 chunk.py doc.pdf --show 3                # print 3 chunks in full
    python3 chunk.py doc.pdf --out results/

Run parse.py first if you have not: chunking operates on the cleaned text that
parsing produces, so anything wrong there is inherited here.

WHAT YOU ARE LOOKING FOR
------------------------
Not the chunks themselves -- those always look fine. Look at how many of your
document's sections survive as retrievable units, and how many chunks carry
text from two sections at once. Those two numbers decide what your retriever
will be able to find, and neither is visible from inside a vector pipeline.
"""

import argparse
import json
import os
import sys

from pi_rag.parse import parse_node as tree_parse
from rag.chunk import STRATEGIES, chunk_node
from rag.parse import parse_node as vector_parse
from shared.loader import load_node
from shared.measure import summarise

W = 78
BAR = "=" * W


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


def show_text(body, indent="      ", limit=None):
    if limit and len(body) > limit:
        body = body[:limit].rstrip() + " ..."
    return "\n".join(indent + "| " + l for l in body.split("\n"))


# =============================================================================
def section_settings(state, settings, text, tree):
    rule("1.  SETTINGS")
    leaves = [n for n in tree.walk() if n.is_leaf and n.level > 0]
    print(f"""
  document        {state['pages'][0].metadata.get('source')}
  cleaned text    {len(text):,} characters over {state['n_pages']} pages
  outline         {len(leaves)} sections {'(from the tree pipeline)' if leaves else '(none found)'}

  strategy        {settings['strategy']}
  chunk size      {settings['size']} characters
  overlap         {settings['overlap']} characters""")

    if settings["strategy"] == "structural" and not leaves:
        print(wrap("\nNOTE  structural chunking needs an outline and this "
                   "document has none, so it fell back to recursive. That is "
                   "the same position a vector pipeline is always in."))


# =============================================================================
def section_chunks(chunks, spans, text, report, show_n):
    rule("2.  THE CHUNKS")
    stats = report["size"]
    cov = report["coverage"]
    print(f"""
  produced        {stats['count']} chunks
  size            {stats['min']:,} to {stats['max']:,} characters (median {stats['median']:,})
  coverage        {cov['uncovered_chars']} characters in no chunk
  redundancy      {cov['redundancy']:.2f}x  (overlap means text is stored more than once)
""")

    print("  Every chunk knows where it came from, because parsing kept the")
    print("  page map:\n")
    print(f"    {'id':<9}{'chars':<16}{'pages':<10}{'size':>7}   opening words")
    print("    " + "-" * 70)
    for chunk in chunks[:8]:
        meta = chunk.metadata
        span = f"{meta['char_start']}-{meta['char_end']}"
        pages = (f"p{meta['start_page']}" if meta["start_page"] == meta["end_page"]
                 else f"p{meta['start_page']}-{meta['end_page']}")
        head = " ".join(chunk.page_content.split())[:30]
        print(f"    {meta['chunk_id']:<9}{span:<16}{pages:<10}"
              f"{len(chunk.page_content):>7}   {head}")
    if len(chunks) > 8:
        print(f"    ... and {len(chunks) - 8} more")

    if show_n:
        print()
        for chunk in chunks[:show_n]:
            print(f"\n  {chunk.metadata['chunk_id']}  "
                  f"({len(chunk.page_content)} chars, "
                  f"pages {chunk.metadata['start_page']}-{chunk.metadata['end_page']})")
            print(show_text(chunk.page_content))


# =============================================================================
def section_damage(report, text, tree, chunks):
    rule("3.  WHAT THE CUTTING COST")
    total = report["sections_total"]

    if not total:
        print(wrap("\nThis document has no detected outline, so there is no "
                   "ground truth to measure against. Everything below needs "
                   "sections to compare with. See parse.py for what to check "
                   "before concluding a document has no structure."))
        return

    print(f"""
  sections in the document          {total}
  survive inside a single chunk     {report['sections_intact']}
  split across chunks               {report['sections_split']}  ({report['split_pct']:.0f}%)
  chunks carrying 2+ sections       {report['mixed_chunks']}
  chunks ending mid-sentence        {report['ragged_ends']}
  chunks starting mid-word          {report['mid_word']}
""")

    if report["split_nodes"]:
        print("  Sections no single chunk can return whole:\n")
        for node in report["split_nodes"][:8]:
            print(f"      {node.title[:52]:<54}{node.size:>6,} chars  {node.pages}")
        if len(report["split_nodes"]) > 8:
            print(f"      ... and {len(report['split_nodes']) - 8} more")
        print(wrap("Answering from one of these requires retrieval to return "
                   "two chunks and generation to stitch them. Nothing "
                   "guarantees the second one ranks high enough to be "
                   "retrieved at all."))

    if report["mixed_detail"]:
        idx, touched = report["mixed_detail"][0]
        start, end = chunks[idx].metadata["char_start"], chunks[idx].metadata["char_end"]
        print(f"\n  Example of a chunk spanning a boundary -- "
              f"{chunks[idx].metadata['chunk_id']}, characters {start}-{end}:\n")
        for node in touched[:3]:
            print(f"      contains part of  {node.title[:56]}")
        print()
        print(show_text(" ".join(chunks[idx].page_content.split()), limit=300))
        print(wrap("One vector now represents two topics at once. It will "
                   "match queries about either and answer neither cleanly."))


# =============================================================================
def section_compare(state, text, tree, size, overlap):
    rule("4.  STRATEGY COMPARISON")
    boundaries = [n.char_start for n in tree.walk() if n.level > 0]
    leaves = [n for n in tree.walk() if n.is_leaf and n.level > 0]

    print(f"\n  Same document, same size ({size}) and overlap ({overlap}).\n")
    print(f"  {'strategy':<14}{'chunks':>8}{'median':>9}{'split':>9}"
          f"{'mixed':>8}{'ragged':>9}{'mid-word':>10}")
    print("  " + "-" * 70)

    results = {}
    for name in ("fixed", "recursive", "structural"):
        base = {**state, "chunk_strategy": name, "chunk_size": size,
                "chunk_overlap": overlap, "section_boundaries": boundaries}
        out = chunk_node(base)
        rep = summarise(text, tree, out["chunk_spans"])
        results[name] = rep
        split = (f"{rep['sections_split']}/{rep['sections_total']}"
                 if rep["sections_total"] else "n/a")
        print(f"  {name:<14}{rep['chunks']:>8}{rep['size']['median']:>9,}"
              f"{split:>9}{rep['mixed_chunks']:>8}{rep['ragged_ends']:>9}"
              f"{rep['mid_word']:>10}")

    if not leaves:
        print(wrap("\nNo outline in this document, so split and mixed counts "
                   "are unavailable and structural chunking degrades to "
                   "recursive."))
        return results

    f, r, s = results["fixed"], results["recursive"], results["structural"]
    print(wrap(f"\nragged and mid-word measure CUT QUALITY. fixed ends "
               f"{f['ragged_ends']} chunks mid-sentence and starts "
               f"{f['mid_word']} mid-word, because it cuts at a character "
               f"count and nothing else. recursive backs off to a sentence or "
               f"paragraph break, giving {r['ragged_ends']} and "
               f"{r['mid_word']}. That is the whole reason recursive "
               "splitting is the usual default."))
    print(wrap(f"\nsplit and mixed measure something cut quality cannot fix. "
               f"fixed and recursive split {f['sections_split']} and "
               f"{r['sections_split']} of {f['sections_total']} sections and "
               f"mix {f['mixed_chunks']} and {r['mixed_chunks']} chunks across "
               "boundaries, because neither knows where a section ends. "
               f"structural splits {s['sections_split']} and mixes "
               f"{s['mixed_chunks']}."))
    print(wrap("\nstructural gets there by cutting at section boundaries "
               "first. It needs an outline to do that -- the thing the vector "
               "pipeline discarded during parsing and the tree pipeline kept. "
               "That is the architectural trade, visible in one table, "
               "measured on this document."))
    return results


# =============================================================================
def section_verdict(report, tree, settings):
    rule("5.  WHAT THIS MEANS")
    total = report["sections_total"]

    if not total:
        print(wrap("\nWithout an outline there is nothing to compare against, "
                   "and chunking is the only option available for this "
                   "document. That is a legitimate outcome: chunk-and-embed "
                   "retrieval works on unstructured text, which is exactly "
                   "what it is for."))
        return

    pct = report["split_pct"]
    print(f"""
  {report['sections_split']} of {total} sections ({pct:.0f}%) cannot be returned whole by any
  single chunk at size {settings['size']}.
""")
    if pct > 50:
        print(wrap("More than half your sections are split. Retrieval will "
                   "routinely return a fragment that reads as complete while "
                   "missing the qualifier, exception or figure that lived in "
                   "the other half. Try a larger chunk size, or structural "
                   "chunking, and watch this number move."))
    elif pct > 15:
        print(wrap("A meaningful minority are split. Worth checking which ones "
                   "in the list above -- if the split sections are the ones "
                   "users ask about, size is the wrong dial and structural "
                   "chunking is the answer."))
    else:
        print(wrap("Most sections survive intact at this size, which is a good "
                   "sign. Check the largest sections anyway: they are the ones "
                   "that split, and they are often the ones that matter."))

    print(wrap("\nThe useful experiment is to run --compare, then change "
               "--size and run it again. Chunk size is usually tuned by feel; "
               "this turns it into a measurement. Note that raising the size "
               "keeps sections together at the cost of diluting each vector, "
               "so there is a real optimum rather than a bigger-is-better "
               "rule."))
    print(wrap("\nNext: embedding, which turns each chunk into a vector. From "
               "there on, nothing can see the text -- only the numbers these "
               "chunks produce."))


# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Chunk a document and measure what the cutting cost.")
    ap.add_argument("path", nargs="?", default=None,
                    help="a .pdf, .txt or .md file (omit for the built-in sample)")
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="recursive")
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--compare", action="store_true",
                    help="run all three strategies and compare the damage")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="print the first N chunks in full")
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
    tstate = tree_parse(state)
    tree = tstate["tree"]
    text = state["document"].page_content

    boundaries = [n.char_start for n in tree.walk() if n.level > 0]
    state = {**state, "chunk_strategy": args.strategy, "chunk_size": args.size,
             "chunk_overlap": args.overlap, "section_boundaries": boundaries}
    state = {**state, **chunk_node(state)}

    chunks, spans = state["chunks"], state["chunk_spans"]
    report = summarise(text, tree, spans)

    print(BAR)
    print("CHUNKING REPORT")
    print(BAR)

    section_settings(state, state["chunk_settings"], text, tree)
    section_chunks(chunks, spans, text, report, args.show)
    section_damage(report, text, tree, chunks)
    if args.compare:
        section_compare(state, text, tree, args.size, args.overlap)
    section_verdict(report, tree, state["chunk_settings"])

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        payload = [{"chunk_id": c.metadata["chunk_id"],
                    "text": c.page_content, **c.metadata} for c in chunks]
        with open(os.path.join(args.out, "chunks.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
        with open(os.path.join(args.out, "chunk_report.json"), "w") as fh:
            json.dump({k: v for k, v in report.items()
                       if k not in ("split_nodes", "mixed_detail")}, fh, indent=2)
        print(f"\n  written to {args.out}/")
        print("      chunks.json")
        print("      chunk_report.json")
    print()


if __name__ == "__main__":
    main()
