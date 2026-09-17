"""
shared/loader.py

Reading a file into one Document per page.

Mirrors LangChain's BaseLoader contract, and matches what PyPDFLoader returns
(one Document per page, metadata carrying source and page), so the real thing
is a drop-in replacement:

    from langchain_community.document_loaders import PyPDFLoader
    pages = PyPDFLoader("report.pdf").load()

DEPENDENCIES
------------
PDFs need one of these:

    pip install pypdf          faster, pure python
    pip install pdfplumber     slower, better with columns and tables

.txt and .md need nothing, and neither does the built-in sample.

THE EXTRACTOR MATTERS MORE THAN YOU EXPECT
------------------------------------------
pypdf and pdfplumber return different text for the same file: different line
breaks, different handling of columns, sometimes a different reading order.
Everything downstream inherits those differences, so a chunk boundary or a
detected heading can move because the library changed. If retrieval quality
shifts after swapping extractors, this is usually why. Pass --extractor to
compare them on your own file.

ONE DOCUMENT PER PAGE IS A DECISION
-----------------------------------
The loader hands back pages, not a document. Whether to keep them as pages or
merge them into continuous text is the first real choice in the pipeline, and
it costs something either way. rag/parse.py implements both so the trade-off
is visible rather than assumed.
"""

import os
from typing import Iterator, List, Optional

from shared.document import Document

INSTALL_HINT = ("No PDF library found. Install one of:\n"
                "    pip install pypdf\n"
                "    pip install pdfplumber")


def _extract_pypdf(path: str):
    from pypdf import PdfReader
    for i, page in enumerate(PdfReader(path).pages, start=1):
        yield i, page.extract_text() or ""


def _extract_pdfplumber(path: str):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            yield i, page.extract_text() or ""


def _extract_text_file(path: str, lines_per_page: int = 45):
    """
    Plain text has no pages, so we invent them. Honest about what it is: a
    fixed-size split, not a real page boundary.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    for i in range(0, len(lines), lines_per_page):
        yield i // lines_per_page + 1, "\n".join(lines[i:i + lines_per_page])


EXTRACTORS = {"pypdf": _extract_pypdf,
              "pdfplumber": _extract_pdfplumber,
              "text": _extract_text_file}


class FileLoader:
    """.load() -> List[Document], one per page."""

    def __init__(self, path: str, extractor: Optional[str] = None):
        self.path = os.path.expanduser(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        self.extractor = extractor or self._pick()

    def _pick(self) -> str:
        if self.path.lower().endswith((".txt", ".md", ".markdown")):
            return "text"
        for name in ("pypdf", "pdfplumber"):
            try:
                __import__(name)
                return name
            except ImportError:
                continue
        raise ImportError(INSTALL_HINT)

    def lazy_load(self) -> Iterator[Document]:
        source = os.path.basename(self.path)
        for page_number, text in EXTRACTORS[self.extractor](self.path):
            yield Document(page_content=text,
                           metadata={"source": source, "page": page_number,
                                     "stage": "raw", "extractor": self.extractor})

    def load(self) -> List[Document]:
        return list(self.lazy_load())


class SampleLoader:
    """The built-in sample document. No file, no dependencies."""

    def load(self) -> List[Document]:
        from shared.sample import PAGES, SOURCE_NAME
        return [Document(page_content=raw,
                         metadata={"source": SOURCE_NAME, "page": page_number,
                                   "stage": "raw", "extractor": "built-in"})
                for page_number, raw in PAGES]


# -----------------------------------------------------------------------------
#  THE LOAD NODE  (state -> updates)
# -----------------------------------------------------------------------------
def load_node(state: dict) -> dict:
    path = state.get("path")
    loader = (FileLoader(path, extractor=state.get("extractor"))
              if path else SampleLoader())
    pages = loader.load()
    return {
        "pages": pages,
        "n_pages": len(pages),
        "raw_chars": sum(len(p.page_content) for p in pages),
        "empty_pages": [p.metadata["page"] for p in pages
                        if not p.page_content.strip()],
    }
