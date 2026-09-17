"""
pi_rag/parse.py

Parsing for tree navigation.

GOAL
----
Turn raw page Documents into a TREE, where every node knows its title, its
exact extent in the text, and the pages it covers.

THE CLEANING IS THE SAME
------------------------
This imports the identical functions from shared/text_ops.py, in the identical
order, as rag/parse.py does. Both produce byte-identical text; verify.py
checks that rather than asking you to believe it.

What differs is what happens next:

    rag/parse.py       merge pages -> hand one string to a chunker
    pi_rag/parse.py    merge pages -> find the document's own outline and
                       record where every section begins and ends

So this is not a different way of reading a file. It is the same reading,
followed by a refusal to discard the outline.

WHY KEEPING THE OUTLINE MATTERS
-------------------------------
It makes a section retrievable as a unit. Chunk-based retrieval has to cut the
document into pieces sized for an embedding model, and those cuts land where
the character budget runs out, not where a thought ends. Tree navigation never
cuts: it walks to a node and returns that node whole.

The cost lands at query time instead. Every search is several LLM calls rather
than one vector comparison, so it is slower and far more expensive per
question. It is a trade, not an upgrade.
"""

from typing import Any, Dict, List, Optional, Tuple

from shared.document import Document
from shared.headings import detect_scheme, is_toc_page, make_matcher
from shared.text_ops import (
    clean_page,
    collapse_spaces,
    continues_sentence,
    detect_page_furniture,
    normalise_pages,
    strip_furniture,
)

from pi_rag.node import TreeNode, assign_node_ids


# -----------------------------------------------------------------------------
#  1. CLEAN AND MERGE
# -----------------------------------------------------------------------------
def clean_and_merge(pages: List[Document], drop_toc: bool = True):
    raw = [(d.metadata["page"], d.page_content) for d in pages]

    # Normalise before detecting furniture; see the note in rag/parse.py.
    normalised = normalise_pages(raw)
    furniture = detect_page_furniture(normalised)

    # Profile the document: the heading scheme decides how pages are unwrapped.
    probe: List[str] = []
    for _, text in normalised:
        probe.extend(strip_furniture(text, furniture).split("\n"))
    scheme = detect_scheme(probe)
    is_heading = make_matcher(scheme)

    report: Dict[str, Any] = {"furniture": sorted(furniture), "hyphens": [],
                              "scheme": scheme, "dropped_pages": [],
                              "page_joins": []}

    text = ""
    page_spans: Dict[int, Tuple[int, int]] = {}

    for page_number, page_text in normalised:
        if drop_toc and is_toc_page(page_text):
            report["dropped_pages"].append(page_number)
            page_spans[page_number] = (len(text), len(text))
            continue

        cleaned = clean_page(page_text, furniture, report["hyphens"], is_heading)
        if not cleaned:
            page_spans[page_number] = (len(text), len(text))
            continue

        if not text:
            joiner = ""
        elif continues_sentence(text, cleaned):
            joiner = " "
            report["page_joins"].append(page_number)
        else:
            joiner = "\n\n"

        start = len(text) + len(joiner)
        text += joiner + cleaned
        page_spans[page_number] = (start, len(text))

    return collapse_spaces(text), page_spans, report, is_heading


# -----------------------------------------------------------------------------
#  2. FIND EVERY HEADING AND ITS EXACT OFFSET
# -----------------------------------------------------------------------------
#  Walk line by line counting characters, so each heading gets a real offset
#  into the merged string. Splitting on "\n" and adding len(line)+1 is exact
#  because that is precisely how the string was assembled.
# -----------------------------------------------------------------------------
def find_headings(text: str, is_heading) -> List[Dict[str, Any]]:
    headings, offset = [], 0
    for line in text.split("\n"):
        found = is_heading(line)
        if found:
            headings.append({"number": found.number, "title": found.title,
                             "level": found.level, "char_start": offset})
        offset += len(line) + 1
    return headings


# -----------------------------------------------------------------------------
#  3. GIVE EACH HEADING ITS SPAN
# -----------------------------------------------------------------------------
#  A section runs from its own heading to the next heading at the same level or
#  higher (a smaller level number). Everything between, including sub-sections,
#  belongs to it:
#
#      3        starts ................................. ends at 4 or EOF
#        3.1    starts ....... ends at 3.2
#        3.2    starts ................................. ends at EOF
# -----------------------------------------------------------------------------
def assign_spans(headings: List[Dict[str, Any]], text_length: int):
    for i, heading in enumerate(headings):
        end = text_length
        for later in headings[i + 1:]:
            if later["level"] <= heading["level"]:
                end = later["char_start"]
                break
        heading["char_end"] = end
    return headings


# -----------------------------------------------------------------------------
#  4. CHARACTER RANGE -> PAGE RANGE
# -----------------------------------------------------------------------------
def pages_for_span(char_start: int, char_end: int,
                   page_spans: Dict[int, Tuple[int, int]]):
    hits = [p for p, (s, e) in page_spans.items()
            if s < char_end and e > char_start and e > s]
    return (min(hits), max(hits)) if hits else (None, None)


# -----------------------------------------------------------------------------
#  5. BUILD THE TREE
# -----------------------------------------------------------------------------
#  A stack walk: a heading is a child of the most recent heading with a smaller
#  level. Pop until the top of the stack is shallower than the current heading.
# -----------------------------------------------------------------------------
def build_tree(headings, text, page_spans, doc_title) -> TreeNode:
    real_pages = [p for p, (s, e) in page_spans.items() if e > s] or [1]
    root = TreeNode(title=doc_title, start_index=min(real_pages),
                    end_index=max(real_pages), char_start=0,
                    char_end=len(text), level=0)

    stack: List[Tuple[int, TreeNode]] = [(0, root)]

    for heading in headings:
        first, last = pages_for_span(heading["char_start"], heading["char_end"],
                                     page_spans)
        node = TreeNode(
            title=(f"{heading['number']} {heading['title']}"
                   if heading["number"] else heading["title"]),
            start_index=first, end_index=last,
            char_start=heading["char_start"], char_end=heading["char_end"],
            number=heading["number"], level=heading["level"],
        )
        while stack and stack[-1][0] >= heading["level"]:
            stack.pop()
        stack[-1][1].nodes.append(node)
        stack.append((heading["level"], node))

    return assign_node_ids(root)


# -----------------------------------------------------------------------------
#  THE PARSE NODE  (state -> updates)
# -----------------------------------------------------------------------------
def parse_node(state: dict) -> dict:
    text, page_spans, report, is_heading = clean_and_merge(state["pages"])
    headings = assign_spans(find_headings(text, is_heading), len(text))
    title = state["pages"][0].metadata.get("source", "document")
    tree = build_tree(headings, text, page_spans, title)

    document = Document(
        page_content=text,
        metadata={"source": title, "stage": "parsed_merged",
                  "n_pages": len(state["pages"]),
                  "heading_scheme": report["scheme"], "page_spans": page_spans},
    )
    return {"document": document, "page_spans": page_spans,
            "headings": headings, "tree": tree, "parse_report": report}
