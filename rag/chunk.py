"""
rag/chunk.py

Step two of the vector pipeline: cut the cleaned document into pieces.

WHY THIS STEP DECIDES MORE THAN IT LOOKS LIKE
---------------------------------------------
An embedding model turns a piece of text into one vector. One vector is one
point of meaning, so a piece has to be small enough to be about a single thing
and large enough to answer something. Chunking picks those pieces, and every
later stage inherits the choice: retrieval can only return chunks that exist,
and generation can only use what retrieval returned.

If a fact gets split across two chunks, no amount of clever ranking recovers
it. The damage is done here, before any model is involved.

THREE STRATEGIES, IMPLEMENTED SO YOU CAN COMPARE THEM
-----------------------------------------------------
    fixed        cut every N characters, no exceptions
                 the honest baseline; fast, and cuts through words

    recursive    prefer to cut at a paragraph break, then a line break, then a
                 sentence end, then a space -- whichever is available near the
                 size limit. This is what most libraries do by default.

    structural   cut at section boundaries first, and only split further when
                 a section exceeds the size limit. Needs an outline, which is
                 exactly what the other pipeline produces.

All three return CHARACTER OFFSETS rather than strings. That matters: offsets
let a chunk report which pages it covers and which sections it overlaps, and
they make the whole thing checkable.

WHAT THIS MODULE DELIBERATELY DOES NOT KNOW
-------------------------------------------
Sections. A real vector pipeline has no outline -- it discarded that during
parsing. chunk.py (the CLI) borrows the tree from the other pipeline to
MEASURE the damage, but the chunker itself works blind, like the real thing.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from shared.document import Document

Span = Tuple[int, int]

#  Break preference, best first. A paragraph break is a better place to cut
#  than a space, and a space is better than the middle of a word.
SEPARATORS: Sequence[Tuple[str, str]] = (
    ("\n\n", "paragraph"),
    ("\n", "line"),
    (". ", "sentence"),
    ("; ", "clause"),
    (" ", "word"),
)


# -----------------------------------------------------------------------------
def _validate_window(size: int, overlap: int) -> None:
    """
    Reject a size/overlap pair that cannot make progress.

    An overlap at or above the chunk size leaves no forward step: both
    strategies fall back to advancing one character, so a 20 KB document
    becomes ~20,000 chunks of 800 characters each and the embedder is handed
    several hundred times the original text. Nothing raises on its own, so the
    only symptom is a run that appears to hang.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than size ({size}); "
            "otherwise each chunk advances one character and the document is "
            "duplicated once per character")


def chunk_fixed(text: str, size: int = 800, overlap: int = 100) -> List[Span]:
    """Cut every `size` characters. Cuts through words without hesitating."""
    _validate_window(size, overlap)
    step = max(1, size - overlap)
    spans, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start += step
    return spans


# -----------------------------------------------------------------------------
def _best_break(text: str, window_start: int, hard_end: int) -> int:
    """
    Find the nicest place to cut inside [window_start, hard_end].

    Walks the separators best-first and takes the LAST occurrence of the best
    one available, so chunks stay close to the target size instead of ending
    early at the first paragraph break they meet.
    """
    for sep, _name in SEPARATORS:
        found = text.rfind(sep, window_start, hard_end)
        if found != -1:
            return found + len(sep)
    return hard_end                       # nothing to work with; hard cut


def _snap_to_word_start(text: str, pos: int) -> int:
    """
    Move a start offset back to the beginning of the word it landed in.

    Without this, the overlap makes the next chunk begin partway through a
    word -- "eriod must be approved" instead of "period must be approved".
    The embedding model would see a token that is not a word, and a human
    reading the retrieved context would see nonsense. Snapping backwards
    gives slightly more overlap than requested, which is the right trade.
    """
    if pos <= 0 or pos >= len(text):
        return max(0, pos)
    if not (text[pos - 1].isalnum() and text[pos].isalnum()):
        return pos                        # already on a boundary
    space = text.rfind(" ", 0, pos)
    newline = text.rfind("\n", 0, pos)
    boundary = max(space, newline)
    return boundary + 1 if boundary != -1 else pos


def chunk_recursive(text: str, size: int = 800, overlap: int = 100) -> List[Span]:
    """
    Fill up to `size`, then back off to the best break point in the last 40%
    of the chunk. This is the behaviour of the recursive character splitters
    most libraries ship with, expressed over offsets.
    """
    _validate_window(size, overlap)
    spans, start = [], 0
    while start < len(text):
        hard_end = min(start + size, len(text))
        if hard_end == len(text):
            spans.append((start, hard_end))
            break
        window_start = start + int(size * 0.6)
        end = _best_break(text, window_start, hard_end)
        spans.append((start, end))
        nxt = _snap_to_word_start(text, end - overlap)
        start = max(nxt, start + 1)       # never go backwards
    return spans


# -----------------------------------------------------------------------------
def chunk_structural(text: str, size: int = 800, overlap: int = 100,
                     boundaries: Optional[Sequence[int]] = None) -> List[Span]:
    """
    Cut at section boundaries, splitting a section further only when it is
    larger than `size`.

    `boundaries` is a sorted list of character offsets where sections begin.
    Without them this falls back to recursive splitting -- which is precisely
    the position a vector pipeline is in by default, having thrown the outline
    away during parsing.
    """
    _validate_window(size, overlap)
    if not boundaries:
        return chunk_recursive(text, size, overlap)

    edges = sorted({0, *[b for b in boundaries if 0 < b < len(text)], len(text)})
    spans: List[Span] = []
    for start, end in zip(edges, edges[1:]):
        if end - start <= size:
            spans.append((start, end))
        else:
            # Section too big for one chunk: split it, but never across the
            # section boundary. Sub-chunks stay inside their own section.
            for sub_start, sub_end in chunk_recursive(text[start:end], size, overlap):
                spans.append((start + sub_start, start + sub_end))
    return spans


STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "structural": chunk_structural,
}


# -----------------------------------------------------------------------------
def spans_to_documents(text: str, spans: Sequence[Span],
                       page_spans: Dict[int, Span], source: str,
                       strategy: str) -> List[Document]:
    """
    Turn offsets into Documents.

    Note the page numbers: they come from the span map the parser kept. This
    is where that bookkeeping pays off -- without it these chunks would have
    no idea where they came from.
    """
    from rag.parse import pages_for_span

    docs = []
    for i, (start, end) in enumerate(spans):
        first, last = pages_for_span(start, end, page_spans)
        docs.append(Document(
            page_content=text[start:end],
            metadata={"source": source, "chunk_id": f"c{i:04d}",
                      "char_start": start, "char_end": end,
                      "start_page": first, "end_page": last,
                      "strategy": strategy, "stage": "chunk"},
        ))
    return docs


# -----------------------------------------------------------------------------
#  THE CHUNK NODE  (state -> updates)
# -----------------------------------------------------------------------------
def chunk_node(state: dict) -> dict:
    text = state["document"].page_content
    strategy = state.get("chunk_strategy", "recursive")
    size = state.get("chunk_size", 800)
    overlap = state.get("chunk_overlap", 100)

    if strategy == "structural":
        spans = chunk_structural(text, size, overlap,
                                 state.get("section_boundaries"))
    else:
        spans = STRATEGIES[strategy](text, size, overlap)

    chunks = spans_to_documents(text, spans, state["page_spans"],
                                state["document"].metadata.get("source", "document"),
                                strategy)
    return {"chunks": chunks, "chunk_spans": spans,
            "chunk_settings": {"strategy": strategy, "size": size,
                               "overlap": overlap}}
