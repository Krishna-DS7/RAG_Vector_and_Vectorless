"""
shared/headings.py

Finding a document's outline.

A heading in a PDF is a visual idea: bigger, bolder, more space above it.
Plain-text extraction discards all of that, so structure has to be inferred
from numbering and capitalisation instead.

This module knows several conventions and picks one by counting how many lines
each explains. Profiling a document before parsing it is what real document
processors do, and it is why a tool written for one numbering style silently
finds nothing in a document that uses another.

    python3 -m shared.headings      # see it accept headings and reject sentences

WHERE THIS FAILS
----------------
It works well on specifications, regulations, contracts, manuals and reports.
It fails on novels, essays, marketing material and anything in columns.

That failure is informative rather than embarrassing. A tree-navigating
retriever needs an outline to navigate; a document without one gives it
nothing. Chunk-and-embed retrieval does not care. Which of those you are
holding is the actual architectural decision, so the parser reports what it
found rather than pretending.
"""

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# A contents entry looks exactly like a heading:
#     "3.2  Termination for Cause .......... 21"    solid leader
#     "Documentation Conventions . . . . . . . 12"  spaced leader (some PDFs)
# The dot leader is the reliable signal; the page number that follows may be
# "21" or "p 40". A solid run and a spaced run are the same idea, so one pattern
# covers both -- and because a run of >=3 dots also matches a solid run, older
# solid-leader documents behave exactly as before.
DOT_LEADER_RE = re.compile(r"(?:\.\s*){3,}")
# Tail form: an entry that ENDS on a leader + page number. Used to stop a single
# contents line from being taken for a heading.
TOC_TAIL_RE = re.compile(r"(?:\.\s*){3,}(?:p\.?\s*)?\d+\s*$", re.IGNORECASE)

MAX_HEADING_LEN = 95
TRAILING_PUNCT = (",", ";", ":")

# A heading is a title, not a sentence. These let match_decimal reject numbered
# procedure steps and instructions ("7 Wait until the LED turns off.") that are
# otherwise shaped exactly like a numbered heading.
CLAUSE_END = (".", "!", "?") + TRAILING_PUNCT
SENTENCE_INNER = re.compile(r"[.!?]\s")
# Trailing blanks plus font-junk glyphs -- zero-width, replacement char, and the
# private-use bullets/arrows PDFs leave behind -- stripped before the end check
# so "... the pools: <glyph>" is seen to end on its colon.
TRAILING_GLYPH_JUNK = re.compile(r"[\s\u200b\ufffd\ue000-\uf8ff]+$")


@dataclass
class Heading:
    number: Optional[str]
    title: str
    level: int


def _plausible(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_LEN:
        return False
    return not TOC_TAIL_RE.search(stripped)


# ---- 1. lettered + decimal:  "A.3.2  Equipment Allowance" -------------------
_DOTTED_ALPHA = re.compile(r"^([A-Z]\.\d+(?:\.\d+)*)\s+(\S.*?)\s*$")


def match_dotted_alpha(line: str) -> Optional[Heading]:
    if not _plausible(line):
        return None
    m = _DOTTED_ALPHA.match(line.strip())
    if not m:
        return None
    return Heading(m.group(1), m.group(2), m.group(1).count("."))


# ---- 2. pure decimal:  "1  Introduction"  /  "2.3.1  Scope" -----------------
_DECIMAL = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+(\S.*?)\s*$")


def match_decimal(line: str) -> Optional[Heading]:
    if not _plausible(line):
        return None
    stripped = line.strip()
    m = _DECIMAL.match(stripped)
    if not m:
        return None
    number, title = m.group(1), m.group(2)

    parts = number.split(".")
    if len(parts) == 1 and int(parts[0]) > 40:
        return None                       # "2024 was a strong year" is not s.2024
    if title[0].islower():
        return None                       # "5,000 units require approval"

    # A heading is a title, not a sentence. Numbered procedure steps read as
    # clauses -- they end on sentence/clause punctuation, or contain a sentence
    # boundary -- so reject those. Strip trailing font-junk glyphs first, so a
    # step ending "... the pools: <glyph>" is still seen to end on its colon.
    core = TRAILING_GLYPH_JUNK.sub("", title)
    if core.endswith(CLAUSE_END):
        return None                       # "... turns off."   "... the pools:"
    if SENTENCE_INNER.search(core):
        return None                       # "... syncing. Enter: SHOW ..."
    return Heading(number, title, len(parts))


# ---- 3. chapter / part / section / article ---------------------------------
_CHAPTER = re.compile(
    r"^((?:CHAPTER|Chapter|PART|Part|SECTION|Section|ARTICLE|Article|ANNEX|Annex)"
    r"\s+(?:\d+|[IVXLC]+|[A-Z]))\s*[:.\u2013-]?\s*(.*?)\s*$"
)


def match_chapter(line: str) -> Optional[Heading]:
    if not _plausible(line):
        return None
    m = _CHAPTER.match(line.strip())
    if not m:
        return None
    return Heading(m.group(1), m.group(2).strip() or m.group(1), 1)


# ---- 4. markdown, for .md and .txt input -----------------------------------
_MARKDOWN = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")


def match_markdown(line: str) -> Optional[Heading]:
    m = _MARKDOWN.match(line.strip())
    return Heading(None, m.group(2), len(m.group(1))) if m else None


# ---- 5. short lines in capitals --------------------------------------------
def match_allcaps(line: str) -> Optional[Heading]:
    stripped = line.strip()
    if not _plausible(stripped) or not 3 <= len(stripped) <= 60:
        return None
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 3 or not all(c.isupper() for c in letters):
        return None
    if stripped.endswith((".",) + TRAILING_PUNCT):
        return None
    if "_" in stripped:
        return None                       # LOG_AMPD_7700 is an identifier, not a heading
    return Heading(None, stripped, 1)


SCHEMES: List[Tuple[str, Callable[[str], Optional[Heading]], str]] = [
    ("dotted_alpha", match_dotted_alpha, "A.1, A.3.2   lettered part + decimals"),
    ("decimal",      match_decimal,      "1, 2.3, 4.1.2   pure decimal numbering"),
    ("chapter",      match_chapter,      "Chapter 3, Part II, Article 7"),
    ("markdown",     match_markdown,     "# Title, ## Subtitle   (md/txt input)"),
    ("allcaps",      match_allcaps,      "SHORT LINES IN CAPITALS"),
]

SCHEMES_BY_NAME = {name: fn for name, fn, _ in SCHEMES}
DESCRIPTIONS = {name: desc for name, _, desc in SCHEMES}


def score_schemes(lines: List[str]) -> List[Tuple[str, int, int]]:
    """
    Per scheme: how many lines it matches, and how many distinct levels it
    produces. Depth matters. Thirty hits all at level 1 usually means the
    pattern is matching ordinary prose; twelve hits across three levels has
    probably found a real outline.
    """
    out = []
    for name, fn, _ in SCHEMES:
        hits = [h for h in (fn(line) for line in lines) if h]
        out.append((name, len(hits), len({h.level for h in hits})))
    return sorted(out, key=lambda row: row[1] * max(row[2], 1), reverse=True)


def detect_scheme(lines: List[str], min_matches: int = 3) -> Optional[str]:
    ranked = score_schemes(lines)
    if not ranked or ranked[0][1] < min_matches:
        return None
    return ranked[0][0]


def make_matcher(scheme: Optional[str]) -> Callable[[str], Optional[Heading]]:
    if scheme is None:
        return lambda line: None
    return SCHEMES_BY_NAME[scheme]


def is_toc_page(text: str) -> bool:
    # A contents or index page is mostly dot-leader lines. Count a leader
    # anywhere on the line, not only at the end: a dense index puts several
    # entries on one line and the page number can wrap away. The >50% threshold
    # is what keeps ordinary pages -- which have the odd "..." at most -- safe.
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return False
    return sum(1 for l in lines if DOT_LEADER_RE.search(l)) / len(lines) > 0.5


if __name__ == "__main__":
    samples = [
        "A.3.2  Equipment Allowance",
        "A.3.2  Equipment Allowance ..... 21",
        "2.3  Scope of the Agreement",
        "4.1.2  Termination for Cause",
        "Chapter 7: Risk Factors",
        "Part II - Financial Statements",
        "## Installing the SDK",
        "SUMMARY OF ACCOUNTING POLICIES",
        "2024 was a strong year for the company.",
        "5,000 units require manager approval before purchase.",
        "The allowance covers a desk, a chair and a monitor.",
    ]
    print(f"  {'line':<56}{'scheme':<14}result")
    print("  " + "-" * 90)
    for line in samples:
        hits = [(name, fn(line)) for name, fn, _ in SCHEMES if fn(line)]
        if hits:
            name, h = hits[0]
            print(f"  {line[:54]:<56}{name:<14}L{h.level}  {h.number} | {h.title[:24]}")
        else:
            print(f"  {line[:54]:<56}{'-':<14}not a heading")
    print("\n  The last three are sentences and a contents entry. They must not")
    print("  match, or the tree fills with sections that look entirely normal")
    print("  and contain nothing.")
