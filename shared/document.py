"""
shared/document.py

A piece of text plus what we know about where it came from.

Deliberately identical in shape to LangChain's Document, so this project can
be swapped onto the real library by changing one import:

    from langchain_core.documents import Document

Every LangChain component speaks Document:

    loader.load()              -> List[Document]
    splitter.split_documents() -> List[Document]
    vectorstore.add_documents(List[Document])
    retriever.invoke(query)    -> List[Document]

Emitting Documents from the parser means the later stages (chunk, embed,
index, retrieve) can be swapped between hand-written and library versions
without touching anything around them.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        head = self.page_content[:52].replace("\n", " ")
        keys = ", ".join(f"{k}={v!r}" for k, v in list(self.metadata.items())[:3])
        return f"Document({head!r}..., {keys})"
