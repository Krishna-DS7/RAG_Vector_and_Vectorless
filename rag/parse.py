"""
rag/parse.py

Parsing for vector search.

GOAL
----
Turn a list of raw page Documents into clean, continuous text that a chunker
can cut up.

THE PAGE-BOUNDARY DECISION
--------------------------
The loader gave us pages. Chunking wants continuous text. Something has to
decide what happens at each page break, and both answers cost something.

    STRATEGY A -- keep pages separate            (many loaders' default)
        clean each page, split each page independently
        + every chunk knows its page number for free
        - a sentence spanning two pages is cut in half
        - a section spanning four pages is fragmented before chunking begins
        - chunk boundaries are forced at every page break, sensible or not

    STRATEGY B -- merge, then split              (used here)
        clean each page, rejoin into one string, split that
        + sentences and sections survive page breaks
        + chunk boundaries chosen on content, not on where the page ended
        - page attribution is no longer free; it has to be tracked

We take B and do the tracking: record where each page starts and ends in the
merged string, so a chunk can still report which page it came from.

Both are implemented, because seeing what A costs on your own document is more
convincing than being told.

WHAT VECTOR SEARCH ACTUALLY LOSES
---------------------------------
Not page numbers -- as this file shows, those are recoverable, and most real
loaders keep them anyway. What it loses is HIERARCHY: that section 3.2 sits
under section 3, where 3.2 stops, and how to retrieve it whole. A flat list of
chunks has nowhere to store that, so parsing does not record it.
"""

from typing import Dict, List, Optional, Tuple

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


def profile(normalised: List[Tuple[int, str]], furniture) -> Optional[str]:
    """
    Work out which heading convention this document uses, before cleaning it.

    The scheme decides how pages get unwrapped, so it has to be known first.
    Two passes over the text is cheap, and profiling before parsing is what
    real document processors do.
    """
    lines = []
    for _, text in normalised:
        lines.extend(strip_furniture(text, furniture).split("\n"))
    return detect_scheme(lines)


# -----------------------------------------------------------------------------
#  STRATEGY A: one cleaned Document per page
# -----------------------------------------------------------------------------
def parse_per_page(pages: List[Document], drop_toc: bool = True):
    raw = [(d.metadata["page"], d.page_content) for d in pages]
    normalised = normalise_pages(raw)
    furniture = detect_page_furniture(normalised)
    is_heading = make_matcher(profile(normalised, furniture))

    out, report = [], {"hyphens": [], "dropped_pages": []}
    for page_number, text in normalised:
        if drop_toc and is_toc_page(text):
            report["dropped_pages"].append(page_number)
            continue
        cleaned = clean_page(text, furniture, report["hyphens"], is_heading)
        if cleaned:
            out.append(Document(
                page_content=cleaned,
                metadata={"source": pages[0].metadata["source"],
                          "page": page_number, "stage": "parsed_per_page"}))
    return out, report


# -----------------------------------------------------------------------------
#  STRATEGY B: one merged Document plus a page -> character-range map
# -----------------------------------------------------------------------------
def parse_merged(pages: List[Document], drop_toc: bool = True):
    raw = [(d.metadata["page"], d.page_content) for d in pages]

    # ORDER MATTERS. Normalise first, then detect furniture. Detecting on raw
    # text memorises a header containing an em-dash, while stripping runs after
    # normalisation where it contains a hyphen. They never match, and the
    # header survives into every chunk with no error raised anywhere.
    normalised = normalise_pages(raw)
    furniture = detect_page_furniture(normalised)
    scheme = profile(normalised, furniture)
    is_heading = make_matcher(scheme)

    report: Dict = {"furniture": sorted(furniture), "hyphens": [], "scheme": scheme,
                    "dropped_pages": [], "page_joins": []}

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
            joiner = " "                    # a sentence crosses this page break
            report["page_joins"].append(page_number)
        else:
            joiner = "\n\n"

        start = len(text) + len(joiner)
        text += joiner + cleaned
        page_spans[page_number] = (start, len(text))

    document = Document(
        page_content=collapse_spaces(text),
        metadata={"source": pages[0].metadata["source"], "stage": "parsed_merged",
                  "n_pages": len(raw), "heading_scheme": scheme,
                  "page_spans": page_spans},
    )
    return document, page_spans, report


def pages_for_span(char_start: int, char_end: int,
                   page_spans: Dict[int, Tuple[int, int]]):
    """
    Which pages does a character range touch? The chunker will call this to
    stamp each chunk with a page number, which is how page attribution
    survives strategy B.
    """
    hits = [p for p, (s, e) in page_spans.items()
            if s < char_end and e > char_start and e > s]
    return (min(hits), max(hits)) if hits else (None, None)


# -----------------------------------------------------------------------------
#  THE PARSE NODE  (state -> updates)
# -----------------------------------------------------------------------------
def parse_node(state: dict) -> dict:
    document, page_spans, report = parse_merged(state["pages"])
    page_documents, _ = parse_per_page(state["pages"])
    return {"document": document, "page_documents": page_documents,
            "page_spans": page_spans, "parse_report": report}
