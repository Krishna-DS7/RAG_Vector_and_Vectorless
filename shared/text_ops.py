"""
shared/text_ops.py

Cleaning primitives. Both pipelines import these and use them identically,
because "repair the ligatures" is not an opinion about retrieval strategy,
it is just correct.

What differs between the pipelines is what they do with the cleaned pages
afterwards. That lives in rag/parse.py and pi_rag/parse.py.

    python3 -m shared.text_ops      # exercise each function on its own
"""

import math
import re
from typing import Dict, List, Optional, Set, Tuple

from shared.headings import match_dotted_alpha

# =============================================================================
#  1. CHARACTER NORMALISATION
# =============================================================================
#  A PDF stores whatever glyphs its fonts had. A font with a decorative "fi"
#  ligature stores "office" as o + <single glyph> + ce, and that middle code
#  point is not the letters f and i. A tokenizer sees an unknown character and
#  someone typing "office" never matches the document.
#
#  One dictionary lookup decides whether a whole class of queries works.

REPLACEMENTS = {
    "\ufb00": "ff",
    "\ufb01": "fi",     # the one hiding inside "office" and "certificate"
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",      # en-dash
    "\u2014": "-",      # em-dash
    "\u00a0": " ",      # non-breaking space
    "\u200b": "",       # zero-width space
}


def normalise_characters(text: str) -> str:
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def normalise_pages(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Normalise every page BEFORE anything else looks at the text."""
    return [(page_number, normalise_characters(raw)) for page_number, raw in pages]


# =============================================================================
#  2. PAGE FURNITURE  (running headers and footers)
# =============================================================================
#  These repeat on every page. Left in, they land in every chunk, so every
#  chunk shares the same words. That adds a constant to every similarity
#  score, flattening the contrast that ranking depends on.
#
#  Footers differ only by their page number, so blank the digits before
#  counting and "Page 3 of 24" and "Page 4 of 24" become one pattern.
#
#  That blinding is necessary and it is also dangerous, because it makes any two
#  lines differing only in digits identical. "Revenue for the period was 12,000"
#  and the same sentence on the next page collapse to one pattern, so ordinary
#  body text in a numeric document looks exactly like a repeating footer. Two
#  further conditions keep real content out:
#
#    POSITION   a running header is the FIRST line of a page and a footer the
#               LAST. Anything further in is content until proved otherwise.
#    PROPORTION a running header appears on most pages, not three of them. An
#               absolute count of 3 means a cross-reference appearing on 3 pages
#               of 400 is stripped from all 400.
#
#  edge_lines is 1 because the two mistakes are not equally bad. Leaving a
#  header in adds the same words to every chunk, which flattens similarity
#  scores -- bad, measurable, recoverable. Removing a line of body text loses
#  content silently and nothing downstream can tell. So the edge stays at the
#  outermost line, and a document with a genuine two-line header keeps its
#  second line: raise edge_lines for that document rather than paying the risk
#  on every other one.

FURNITURE_EDGE_LINES = 1
FURNITURE_MIN_RATIO = 0.5


def _digit_blind(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def detect_page_furniture(pages: List[Tuple[int, str]], min_pages: int = 3,
                          min_ratio: float = FURNITURE_MIN_RATIO,
                          edge_lines: int = FURNITURE_EDGE_LINES) -> Set[str]:
    """
    Find the running headers and footers repeated across ``pages``.

    A line is furniture when it sits within ``edge_lines`` of the top or bottom
    of its page AND its digit-blinded form recurs on at least ``min_ratio`` of
    the pages (never fewer than ``min_pages``).

    Raise ``edge_lines`` only for a document known to carry a multi-line header;
    every line it admits is a line that can be deleted as furniture.
    """
    threshold = max(min_pages, math.ceil(min_ratio * len(pages)))
    seen: Dict[str, int] = {}
    for _, text in pages:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        edges = set(lines[:edge_lines]) | set(lines[-edge_lines:])
        for line in edges:
            key = _digit_blind(line)
            seen[key] = seen.get(key, 0) + 1
    return {key for key, count in seen.items() if count >= threshold}


def strip_furniture(text: str, furniture: Set[str]) -> str:
    return "\n".join(l for l in text.split("\n") if _digit_blind(l) not in furniture)


# =============================================================================
#  3. DE-HYPHENATION
# =============================================================================
#  "calen-\ndar" -> "calendar"   the hyphen was inserted by typesetting
#  "one-\ntime"  -> "one-time"   the hyphen is part of the word
#
#  No reliable rule separates these. Every library gets some wrong. This uses a
#  small allow-list; production code checks a dictionary and is still wrong
#  sometimes. A permanent, boring source of retrieval bugs.

KEEP_HYPHEN = {
    "one-time", "full-time", "part-time", "multi-factor", "long-term",
    "short-term", "e-mail", "on-site", "self-service", "day-to-day",
    "end-to-end", "read-only", "real-time", "third-party", "co-operate",
}


def join_hyphenated_linebreaks(text: str, report: Optional[List] = None) -> str:
    def decide(match: "re.Match") -> str:
        left, right = match.group(1), match.group(2)
        candidate = f"{left}-{right}"
        if candidate.lower() in KEEP_HYPHEN:
            if report is not None:
                report.append(("kept hyphen", candidate))
            return candidate
        if report is not None:
            report.append(("joined", f"{left}-{right} -> {left}{right}"))
        return left + right

    return re.sub(r"(\w+)-\n(\w+)", decide, text)


# =============================================================================
#  4. PARAGRAPH UNWRAPPING
# =============================================================================
#  A PDF gives a hard newline wherever the line ran out of width, not where a
#  paragraph ended. Leave them in and any later step that splits on newlines
#  shreds sentences.
#
#  Blank lines separate blocks; inside a block, join with a space. Headings
#  stay on their own line so structure survives for the tree builder.

def unwrap_paragraphs(text: str, is_heading=None) -> str:
    """
    is_heading: callable returning truthy for a heading line. Defaults to one
    scheme; parse.py passes the scheme actually detected in the document being
    processed.
    """
    if is_heading is None:
        is_heading = match_dotted_alpha
    blocks = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        if is_heading(lines[0]):
            blocks.append(lines[0])
            if lines[1:]:
                blocks.append(" ".join(lines[1:]))
        else:
            blocks.append(" ".join(lines))
    return "\n\n".join(blocks)


def collapse_spaces(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", text)


# =============================================================================
#  5. CROSS-PAGE SENTENCES
# =============================================================================
#  Does page N+1 begin a new paragraph, or finish the sentence page N started?
#  If page N does not end in sentence punctuation and page N+1 starts
#  lowercase, it is a continuation and must be joined with a space.
#
#  Get this wrong and any statement that happens to straddle a page break is
#  destroyed, silently, in two halves that each read as complete.

def continues_sentence(previous_text: str, next_text: str) -> bool:
    if not previous_text or not next_text:
        return False
    ends_open = not previous_text.rstrip().endswith((".", "!", "?", ":", ";"))
    starts_lower = next_text.lstrip()[:1].islower()
    return ends_open and starts_lower


# =============================================================================
#  6. THE PER-PAGE PIPELINE
# =============================================================================
#  ORDER MATTERS. Characters are normalised before furniture is detected. Do it
#  the other way round and the header you memorise contains an em-dash while
#  the header you later strip contains a hyphen. They never match, and the
#  running header survives into every chunk without any error being raised.

def clean_page(raw: str, furniture: Set[str], hyphen_report: Optional[List] = None,
               is_heading=None) -> str:
    text = strip_furniture(raw, furniture)
    text = join_hyphenated_linebreaks(text, hyphen_report)
    text = unwrap_paragraphs(text, is_heading)
    return collapse_spaces(text).strip()


if __name__ == "__main__":
    print("=" * 74)
    print("EACH CLEANING FUNCTION, ON ITS OWN")
    print("=" * 74)

    print("\n1. normalise_characters")
    sample = "home o\ufb03ce costs 60,000\u00a0units \u2014 that\u2019s the \u201climit\u201d"
    print(f"   in  : {sample!r}")
    print(f"   out : {normalise_characters(sample)!r}")

    print("\n2. de-hyphenation (note the ambiguity)")
    for raw in ["calen-\ndar year", "a one-\ntime payment", "pay for it your-\nself"]:
        log = []
        print(f"   {raw!r:28} -> {join_hyphenated_linebreaks(raw, log)!r:24} {log}")

    print("\n3. unwrap_paragraphs")
    wrapped = "Employees may take up\nto 12 days of paid\nleave per year."
    print(f"   in  : {wrapped!r}")
    print(f"   out : {unwrap_paragraphs(wrapped)!r}")

    print("\n4. continues_sentence (cross-page detection)")
    for a, b in [("...is not claimed through this", "process; see section 4"),
                 ("It refreshes every 4 years.", "Order equipment through the")]:
        print(f"   {a[-30:]:32} + {b[:28]:30} -> {continues_sentence(a, b)}")

    print("\n5. furniture detection (digit-blind)")
    for line in ["Confidential - Page 3 of 24", "Confidential - Page 17 of 24"]:
        print(f"   {line:32} -> {_digit_blind(line)!r}")
    print("   identical patterns, so both are recognised as the same footer")
