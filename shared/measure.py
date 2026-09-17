"""
shared/measure.py

Measuring what chunking cost.

THE TRICK, AND WHY IT IS HONEST
-------------------------------
To know whether chunking broke a section, you need to know where the sections
were. A vector pipeline does not know that -- it discarded the outline during
parsing, which is the whole point.

So these measurements borrow the tree from the other pipeline and use it as
ground truth. That is a legitimate thing to do in a diagnostic tool and an
illegitimate thing to do in production: a real chunk-and-embed system is blind
to every number computed here.

Which is the finding, really. The damage is not hard to measure. It is hard to
measure FROM INSIDE the architecture that causes it.

DEFINITIONS
-----------
intact section    some single chunk contains the whole section
split section     no single chunk does, so no retrieval can return it whole
mixed chunk       covers substantial text from two or more sections at once
mid-sentence cut  the chunk begins partway through a sentence or a word
"""

from typing import Dict, List, Sequence, Tuple

Span = Tuple[int, int]

MIN_OVERLAP_CHARS = 25      # below this, treat an overlap as incidental


def leaf_sections(tree) -> List:
    return [n for n in tree.walk() if n.is_leaf and n.level > 0]


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def section_integrity(tree, spans: Sequence[Span]) -> Dict:
    """
    For each leaf section: is there a single chunk that contains all of it?

    If not, answering from that section requires stitching two or more chunks
    together, which only happens if retrieval returns both -- and nothing
    guarantees it will.
    """
    sections = leaf_sections(tree)
    intact, split = [], []
    for node in sections:
        section_span = (node.char_start, node.char_end)
        contained = any(s <= node.char_start and e >= node.char_end
                        for s, e in spans)
        (intact if contained else split).append(node)
    return {"total": len(sections), "intact": intact, "split": split}


def mixed_chunks(tree, spans: Sequence[Span]) -> List[Tuple[int, List]]:
    """Chunks carrying substantial text from more than one section."""
    sections = leaf_sections(tree)
    out = []
    for i, span in enumerate(spans):
        touched = [n for n in sections
                   if _overlap(span, (n.char_start, n.char_end)) >= MIN_OVERLAP_CHARS]
        if len(touched) > 1:
            out.append((i, touched))
    return out


def mid_word_starts(text: str, spans: Sequence[Span]) -> List[int]:
    """
    Chunks that begin inside a word: "eriod must be approved".

    Purely an artefact of where the overlap lands. The fix is to snap the
    start back to a word boundary, which rag/chunk.py does for the strategies
    that care. Fixed-size chunking does not, on purpose, so you can see the
    difference.
    """
    out = []
    for i, (start, _end) in enumerate(spans):
        if start == 0:
            continue
        if text[start - 1].isalnum() and text[start:start + 1].isalnum():
            out.append(i)
    return out


def ragged_ends(text: str, spans: Sequence[Span]) -> List[int]:
    """
    Chunks that STOP partway through a sentence.

    This is the metric that actually measures cut quality, and the one
    recursive splitting exists to improve. A chunk ending mid-sentence gets
    embedded as a fragment whose final clause is missing, and the missing
    clause is often the qualifier that changes the meaning.
    """
    out = []
    for i, (_start, end) in enumerate(spans):
        if end >= len(text):
            continue
        tail = text[:end]
        # A cut landing on a line or paragraph break is clean by definition,
        # even when the text before it has no full stop -- a heading is the
        # obvious case. Check this BEFORE stripping, or every chunk that ends
        # neatly at a section boundary gets reported as ragged.
        if tail.endswith("\n"):
            continue
        stripped = tail.rstrip()
        if stripped and stripped[-1] not in ".!?:;":
            out.append(i)
    return out


def size_stats(spans: Sequence[Span]) -> Dict:
    sizes = sorted(e - s for s, e in spans)
    if not sizes:
        return {"count": 0, "min": 0, "max": 0, "median": 0, "mean": 0}
    return {"count": len(sizes), "min": sizes[0], "max": sizes[-1],
            "median": sizes[len(sizes) // 2],
            "mean": sum(sizes) // len(sizes)}


def coverage(text: str, spans: Sequence[Span]) -> Dict:
    """Is every character of the document inside at least one chunk?"""
    covered = [False] * len(text)
    for start, end in spans:
        for i in range(start, min(end, len(text))):
            covered[i] = True
    missing = covered.count(False)
    duplicated = sum(e - s for s, e in spans) - (len(text) - missing)
    return {"uncovered_chars": missing,
            "duplicated_chars": max(0, duplicated),
            "redundancy": (sum(e - s for s, e in spans) / len(text)) if text else 0}


def summarise(text: str, tree, spans: Sequence[Span]) -> Dict:
    integrity = section_integrity(tree, spans)
    mixed = mixed_chunks(tree, spans)
    midword = mid_word_starts(text, spans)
    ragged = ragged_ends(text, spans)
    stats = size_stats(spans)
    cov = coverage(text, spans)
    total = integrity["total"]
    return {
        "chunks": stats["count"],
        "size": stats,
        "sections_total": total,
        "sections_intact": len(integrity["intact"]),
        "sections_split": len(integrity["split"]),
        "split_pct": (100 * len(integrity["split"]) / total) if total else 0,
        "split_nodes": integrity["split"],
        "mixed_chunks": len(mixed),
        "mixed_detail": mixed,
        "mid_word": len(midword),
        "ragged_ends": len(ragged),
        "coverage": cov,
    }
