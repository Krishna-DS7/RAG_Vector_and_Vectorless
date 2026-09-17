#!/usr/bin/env python3
"""
parse.py  --  run both parsing pipelines over a document and report what happened

    python3 parse.py report.pdf
    python3 parse.py notes.md
    python3 parse.py                            # built-in sample, no deps needed

    python3 parse.py doc.pdf --show vector      # one pipeline only
    python3 parse.py doc.pdf --show tree
    python3 parse.py doc.pdf --extractor pdfplumber
    python3 parse.py doc.pdf --out results/     # write the artefacts to disk
    python3 parse.py doc.pdf --quiet            # summary only

PDFs need `pip install pypdf` (or pdfplumber). Text and markdown need nothing.

Everything printed is measured from the document you passed in. Nothing is
canned, so the output is worth reading closely: it tells you what your file
actually contains and what each architecture was able to keep.
"""

import argparse
import json
import os
import sys

from pi_rag.parse import parse_node as tree_parse
from rag.parse import pages_for_span, parse_node as vector_parse
from shared.headings import DESCRIPTIONS, score_schemes
from shared.loader import load_node
from shared.text_ops import detect_page_furniture, normalise_pages, strip_furniture

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


def excerpt(text, limit=380, indent="    "):
    body = text[:limit].rstrip()
    if len(text) > limit:
        body += " ..."
    return "\n".join(indent + "| " + l for l in body.split("\n"))


# =============================================================================
def section_source(state, args):
    rule("1.  SOURCE")
    pages = state["pages"]
    meta = pages[0].metadata
    print(f"""
  file          {meta.get('source')}
  extractor     {meta.get('extractor')}
  pages         {state['n_pages']}
  characters    {state['raw_chars']:,} as extracted""")

    empty = state.get("empty_pages", [])
    if len(empty) == state["n_pages"]:
        print(f"""
  Every page extracted zero characters.

  This is a scanned document: pictures of text, not text. Nothing downstream
  can help, because there is nothing to read. Run OCR first --

      pip install ocrmypdf
      ocrmypdf in.pdf out.pdf

  then run this again on the output.""")
        return False

    if empty:
        pct = 100 * len(empty) / state["n_pages"]
        print(f"""
  WARNING   {len(empty)} of {state['n_pages']} pages ({pct:.0f}%) extracted no text.
            pages {empty[:12]}{' ...' if len(empty) > 12 else ''}
            Probably scans or images. They are invisible to everything below.""")

    print("\n  What the extractor handed back, page 1:\n")
    print(excerpt(pages[0].page_content, 300))
    print(wrap("\nThis is the real input. Any damage visible here -- broken "
               "words, repeated headers, odd characters -- has to be repaired "
               "before it reaches an embedding model or an LLM, because "
               "neither will tell you it was there."))
    return True


# =============================================================================
def section_cleaning(state, report, text):
    rule("2.  CLEANING")
    raw_chars = state["raw_chars"]
    kept = 100 * len(text) / max(raw_chars, 1)

    print(f"\n  {raw_chars:,} characters in  ->  {len(text):,} out   ({kept:.0f}% kept)\n")

    furniture = report["furniture"]
    print(f"  repeated lines removed from every page   {len(furniture)}")
    for line in furniture[:5]:
        print(f"      {line!r}")
    if len(furniture) > 5:
        print(f"      ... and {len(furniture) - 5} more")
    print(wrap("Running headers and footers appear in every chunk if left in, "
               "so every chunk ends up sharing the same words. That adds a "
               "constant to every similarity score and flattens the contrast "
               "ranking depends on.", indent="      "))

    print(f"\n  contents pages dropped                   "
          f"{report['dropped_pages'] or 'none'}")
    if report["dropped_pages"]:
        print(wrap("A contents entry looks exactly like a heading. Left in, "
                   "the outline fills with sections that contain nothing but "
                   "the title of a real section elsewhere.", indent="      "))

    joins = report["page_joins"]
    print(f"\n  sentences rejoined across a page break   {len(joins)}")
    if joins:
        print(f"      at pages {joins[:10]}{' ...' if len(joins) > 10 else ''}")
        print(wrap("Each of these was a sentence split in half by a page "
                   "boundary. Treating page breaks as paragraph breaks would "
                   "leave both halves reading as complete statements while "
                   "meaning something different.", indent="      "))

    hyphens = report["hyphens"]
    joined = [d for a, d in hyphens if a == "joined"]
    kepth = [d for a, d in hyphens if a == "kept hyphen"]
    print(f"\n  hyphenated line breaks handled           {len(hyphens)}")
    for detail in joined[:4]:
        print(f"      joined       {detail}")
    for detail in kepth[:4]:
        print(f"      kept hyphen  {detail}   <- a real hyphen, not typesetting")
    if hyphens:
        print(wrap("No rule reliably separates these two cases. A word split "
                   "for line width must be rejoined; a genuinely hyphenated "
                   "word must not. Getting it wrong produces tokens no query "
                   "will ever contain.", indent="      "))

    if kept < 40:
        print(wrap("\nNOTE  Under 40% of the text survived. Either this file is "
                   "mostly furniture, or furniture detection is eating real "
                   "content. Check the output before trusting it."))


# =============================================================================
def section_vector(vstate, text):
    rule("3.  VECTOR-SEARCH PIPELINE   (rag/)")
    per_page = vstate["page_documents"]
    spans = vstate["page_spans"]
    joins = vstate["parse_report"]["page_joins"]

    print(f"""
  Output: ONE Document of {len(text):,} characters, ready to be chunked.

  There was a choice at every page break:

    A  keep pages separate    {len(per_page):>4} Documents, page numbers free,
                                   but sentences and sections cut at every break
    B  merge, then split      {1:>4} Document, boundaries chosen on content,
                                   page numbers tracked separately     <- used
""")

    if joins:
        page = joins[0]
        before = next((d for d in per_page if d.metadata["page"] == page - 1), None)
        after = next((d for d in per_page if d.metadata["page"] == page), None)
        if before and after:
            print(f"  What strategy A would cost, at the page {page - 1}/{page} break "
                  f"in this file:\n")
            print(f"      page {page - 1} ends    ...{before.page_content[-42:]!r}")
            print(f"      page {page} starts   {after.page_content[:42]!r}...")
            print(wrap("Two Documents, one broken sentence. No chunker can "
                       "repair that, because by then the halves are in "
                       "different objects.", indent="      "))
    else:
        print("  No sentence in this file crosses a page break, so both "
              "strategies\n  would give the same text here.")

    print(f"\n  Page attribution is kept by tracking spans "
          f"({len(spans)} recorded):\n")
    probe = [(0, min(120, len(text))),
             (max(0, len(text) - 500), len(text))]
    for lo, hi in probe:
        first, last = pages_for_span(lo, hi, spans)
        print(f"      characters {lo:>6}-{hi:<6} -> pages {first}-{last}")
    print(wrap("So \"vector search cannot cite pages\" is false. Page numbers "
               "are recoverable, and most loaders keep them anyway."))

    print(wrap("\nWhat it genuinely cannot keep is hierarchy: which section "
               "contains which, where a section stops, and how to return one "
               "whole. A flat list of chunks has nowhere to put that, so the "
               "parser does not record it. Chunking will now cut this text "
               "wherever the character budget runs out."))


# =============================================================================
def section_tree(tstate, text, probe_lines):
    rule("4.  TREE-NAVIGATION PIPELINE   (pi_rag/)")
    tree = tstate["tree"]
    headings = tstate["headings"]
    nodes = list(tree.walk())
    leaves = [n for n in nodes if n.is_leaf and n.level > 0]
    scheme = tstate["parse_report"]["scheme"]

    print("\n  Which numbering convention does this document use?\n")
    print(f"  {'scheme':<15}{'matches':>9}{'levels':>8}   what it looks for")
    print("  " + "-" * 72)
    for name, hits, levels in score_schemes(probe_lines):
        mark = "  <- chosen" if name == scheme else ""
        print(f"  {name:<15}{hits:>9}{levels:>8}   {DESCRIPTIONS[name]}{mark}")
    print(wrap("Depth matters as much as count. Thirty matches all at one "
               "level usually means the pattern is catching ordinary prose; a "
               "dozen across three levels is probably a real outline."))

    if not leaves:
        print(f"""
  RESULT   No outline found. The tree is a single node covering everything.

  There is nothing here to navigate, so this architecture has no advantage on
  this document -- every search would return the whole file.

  Before concluding the document has no structure, check whether it simply
  uses a convention this code does not know. Open the file and look at how
  sections are marked. Adding a matcher to shared/headings.py is about fifteen
  lines. If the document genuinely has no sections -- prose, a form, a scan --
  then chunk-and-embed retrieval is the right choice, and that is a finding
  rather than a failure.""")
        return

    depth = max(n.level for n in nodes)
    sizes = sorted(n.size for n in leaves)
    median = sizes[len(sizes) // 2]

    print(f"""
  headings found  {len(headings)}
  nodes           {len(nodes)}
  leaf sections   {len(leaves)}
  depth           {depth} level{'s' if depth != 1 else ''}
  section size    {sizes[0]:,} to {sizes[-1]:,} characters (median {median:,})

  The outline:
""")
    print(f"    {'id':<6}{'section':<48}{'pages':>9}{'size':>10}")
    print("    " + "-" * 71)
    for node in nodes[:18]:
        label = f"{'   ' * node.level}{node.title}"[:46]
        tag = "leaf" if node.is_leaf else ""
        print(f"    {node.node_id:<6}{label:<48}{node.pages:>9}"
              f"{node.size:>8,}  {tag}")
    if len(nodes) > 18:
        print(f"    ... and {len(nodes) - 18} more")

    # ---- prove the offsets actually work --------------------------------
    target = max(leaves, key=lambda n: n.size)
    extract = target.text(text)
    print(f"""
  Do the recorded boundaries work? Slice the cleaned text using the offsets
  stored on the largest section, node {target.node_id}:
""")
    print(f"      title        {target.title}")
    print(f"      pages        {target.start_index} to {target.end_index}")
    print(f"      characters   {target.char_start} to {target.char_end}")
    print()
    print(excerpt(extract.strip(), 420, indent="      "))
    print(wrap(f"\nThat came back whole -- {len(extract):,} characters, one "
               "coherent section, no cutting. A chunk-based pipeline can only "
               "return pieces of it, sized by a character budget that knows "
               "nothing about where the section ends."))

    if sizes[-1] > 20000:
        print(wrap(f"\nNOTE  The largest section is {sizes[-1]:,} characters. "
                   "Sections that big may not fit a context window whole, "
                   "which removes the main advantage. Consider treating a "
                   "deeper level as the retrievable unit."))


# =============================================================================
def section_verdict(vstate, tstate, text):
    rule("5.  WHAT THIS MEANS FOR THIS DOCUMENT")
    tree = tstate["tree"]
    nodes = list(tree.walk())
    leaves = [n for n in nodes if n.is_leaf and n.level > 0]
    scheme = tstate["parse_report"]["scheme"]

    depth = max((n.level for n in nodes), default=0)

    def row(label, a, b):
        print(f"  {label:<28}{a:<24}{b}")

    print()
    row("", "vector search", "tree navigation")
    print("  " + "-" * 74)
    row("loading", "shared", "shared")
    row("cleaning", "shared", "shared")
    row("cleaned text", f"{len(text):,} chars", f"{len(text):,} chars  (identical)")
    row("output", "1 Document", f"{len(nodes)} nodes")
    row("page attribution", "via span map", "via node page range")
    row("heading hierarchy", "discarded",
        f"{depth} level{'s' if depth != 1 else ''}" if leaves else "none found")
    row("section boundaries", "none",
        f"{len(leaves)} sections, exact" if leaves else "none")
    row("return a section whole", "no", "yes" if leaves else "no")
    row("needs next", "chunker + embeddings", "an LLM for summaries")
    print()

    print("  vector search")
    print(wrap("Works here regardless of layout. It needs no structure and "
               "will chunk this text and search it by similarity. Next step: "
               "chunking, in rag/.", indent="      "))

    print("\n  tree navigation")
    if not leaves:
        print(wrap("Poor fit. No outline was found, so there is nothing to "
                   "navigate. See section 4 for what to check before "
                   "concluding the document has no structure.", indent="      "))
    else:
        sizes = sorted(n.size for n in leaves)
        median = sizes[len(sizes) // 2]
        if 200 <= median <= 20000:
            print(wrap(f"Good fit. {len(leaves)} sections across {depth} "
                       f"level{'s' if depth != 1 else ''} using {scheme} "
                       f"numbering, median section {median:,} characters -- "
                       "small enough to return whole, large enough to be a "
                       "real answer. Next step: summarisation, in pi_rag/.",
                       indent="      "))
        else:
            print(wrap(f"Workable, with care. Structure exists but section "
                       f"sizes are awkward (median {median:,} characters) -- "
                       "some too small to answer anything, or too large to "
                       "return whole. Look at the outline above and decide "
                       "which level is the retrievable unit.", indent="      "))

    print()
    print(wrap("These are not competitors so much as different trades. "
               "Chunking is cheap per query and indifferent to layout; tree "
               "navigation preserves whole sections and costs several model "
               "calls per question. Most real systems end up running both "
               "with a router in front."))


# =============================================================================
def _build_report(text, vstate, tstate):
    return {
        "source": vstate["pages"][0].metadata.get("source"),
        "extractor": vstate["pages"][0].metadata.get("extractor"),
        "pages": vstate["n_pages"],
        "raw_chars": vstate["raw_chars"],
        "clean_chars": len(text),
        "heading_scheme": tstate["parse_report"]["scheme"],
        "furniture_removed": vstate["parse_report"]["furniture"],
        "contents_pages_dropped": vstate["parse_report"]["dropped_pages"],
        "cross_page_sentences": vstate["parse_report"]["page_joins"],
        "hyphen_decisions": vstate["parse_report"]["hyphens"],
        "headings": len(tstate["headings"]),
        "nodes": len(list(tstate["tree"].walk())),
    }


def write_outputs(out_dir, text, vstate, tstate):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cleaned_text.txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(out_dir, "tree.json"), "w") as fh:
        json.dump(tstate["tree"].to_dict(), fh, indent=2)
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(_build_report(text, vstate, tstate), fh, indent=2)
    print(f"\n  written to {out_dir}/")
    for name in ("cleaned_text.txt", "tree.json", "report.json"):
        print(f"      {name}")


def write_outputs_split(text, vstate, tstate, base=None):
    """Each pipeline's output into its own folder: vector -> rag/, tree -> pi_rag/.

    Paths resolve relative to parse.py, so it works from any working directory.
    cleaned_text.txt lives in rag/; note that pi_rag/tree.json's char offsets
    index into that same text.
    """
    base = base or os.path.dirname(os.path.abspath(__file__))
    rag_dir, pi_dir = os.path.join(base, "rag"), os.path.join(base, "pi_rag")
    os.makedirs(rag_dir, exist_ok=True)
    os.makedirs(pi_dir, exist_ok=True)

    with open(os.path.join(rag_dir, "cleaned_text.txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(rag_dir, "report.json"), "w") as fh:
        json.dump(_build_report(text, vstate, tstate), fh, indent=2)
    with open(os.path.join(pi_dir, "tree.json"), "w") as fh:
        json.dump(tstate["tree"].to_dict(), fh, indent=2)

    print("\n  written to project folders:")
    for name in ("rag/cleaned_text.txt", "rag/report.json", "pi_rag/tree.json"):
        print(f"      {name}")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Parse a document two ways and report what each kept.")
    ap.add_argument("path", nargs="?", default=None,
                    help="a .pdf, .txt or .md file (omit for the built-in sample)")
    ap.add_argument("--show", choices=["both", "vector", "tree"], default="both")
    ap.add_argument("--extractor", choices=["pypdf", "pdfplumber"], default=None)
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="write cleaned text, tree.json and report.json here")
    ap.add_argument("--split", action="store_true",
                    help="write vector artefacts to rag/ and the tree to pi_rag/")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    try:
        state = load_node({"path": args.path, "extractor": args.extractor})
    except FileNotFoundError:
        sys.exit(f"file not found: {args.path}")
    except ImportError as exc:
        sys.exit(str(exc))

    if not state["pages"]:
        sys.exit("no pages could be read from that file")

    print(BAR)
    print("DOCUMENT PARSING REPORT")
    print(BAR)

    if not args.quiet:
        if not section_source(state, args):
            return

    vstate = {**state, **vector_parse(state)}
    tstate = {**state, **tree_parse(state)}
    text = vstate["document"].page_content

    assert text == tstate["document"].page_content, \
        "the two pipelines disagree on the cleaned text"

    raw = [(d.metadata["page"], d.page_content) for d in state["pages"]]
    normalised = normalise_pages(raw)
    furniture = detect_page_furniture(normalised)
    probe_lines = []
    for _, page_text in normalised:
        probe_lines.extend(strip_furniture(page_text, furniture).split("\n"))

    if not args.quiet:
        section_cleaning(state, vstate["parse_report"], text)
        if args.show in ("both", "vector"):
            section_vector(vstate, text)
        if args.show in ("both", "tree"):
            section_tree(tstate, text, probe_lines)

    section_verdict(vstate, tstate, text)

    if args.out:
        write_outputs(args.out, text, vstate, tstate)
    if args.split:
        write_outputs_split(text, vstate, tstate)
    print()


if __name__ == "__main__":
    main()
