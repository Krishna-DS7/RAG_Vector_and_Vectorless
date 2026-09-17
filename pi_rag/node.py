"""
pi_rag/node.py

A node in the document tree.

THE SCHEMA
----------
This follows the PageIndex project's node format
(github.com/VectifyAI/PageIndex):

    {
      "title":       "Equipment Allowance",
      "node_id":     "0011",
      "start_index":  12,          <- PAGE index, not a character offset
      "end_index":    14,
      "summary":     "written by an LLM",
      "nodes":       [ ...children... ]
    }

Those field names are matched exactly. Two are added here:

    char_start / char_end    exact offsets into the cleaned text
    number / level           the section number and heading depth

start_index and end_index being page numbers is what lets an answer cite
"pages 12-14". The character offsets are ours, so a section can be pulled back
out exactly without re-reading the source file.

ABOUT summary
-------------
It stays None after parsing. Summaries are written by an LLM in a later step,
and they are not decoration: a tree search navigates by reading them, so a node
whose summary omits a fact is a node the search walks straight past. Retrieval
quality in this architecture is largely summary quality.
"""

from dataclasses import asdict, dataclass, field
from typing import Iterator, List, Optional

from shared.document import Document


@dataclass
class TreeNode:
    title: str
    node_id: Optional[str] = None
    start_index: Optional[int] = None      # first page
    end_index: Optional[int] = None        # last page
    summary: Optional[str] = None          # filled in by an LLM later
    nodes: List["TreeNode"] = field(default_factory=list)

    # additions beyond the published schema
    char_start: int = 0
    char_end: int = 0
    number: Optional[str] = None
    level: int = 0                         # 0 = document root

    @property
    def is_leaf(self) -> bool:
        return not self.nodes

    @property
    def size(self) -> int:
        return self.char_end - self.char_start

    @property
    def pages(self) -> str:
        if self.start_index == self.end_index:
            return f"p{self.start_index}"
        return f"p{self.start_index}-{self.end_index}"

    def text(self, document_text: str) -> str:
        """Pull this node's text back out of the cleaned document."""
        return document_text[self.char_start:self.char_end]

    def walk(self) -> Iterator["TreeNode"]:
        yield self
        for child in self.nodes:
            yield from child.walk()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_document(self, document_text: str) -> Document:
        """
        Express a node as a Document, so this pipeline can feed anything that
        expects the LangChain shape.

        Note what the metadata carries that a text chunk cannot: the node id,
        its depth, its section number and its exact extent -- enough for a
        caller to fetch the parent, the siblings, or the section whole.
        """
        return Document(
            page_content=self.text(document_text),
            metadata={"node_id": self.node_id, "title": self.title,
                      "number": self.number, "level": self.level,
                      "start_page": self.start_index, "end_page": self.end_index,
                      "char_start": self.char_start, "char_end": self.char_end,
                      "is_leaf": self.is_leaf, "stage": "tree_node"},
        )


def assign_node_ids(root: TreeNode) -> TreeNode:
    """Depth-first, zero-padded to four digits."""
    for i, node in enumerate(root.walk()):
        node.node_id = f"{i:04d}"
    return root
