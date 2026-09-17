"""
pi_rag/state.py

What flows through the tree-navigation pipeline.

Read this beside rag/state.py. The two share their first half exactly --
pages, document, page_spans -- and then diverge:

    vector search  ->  chunks, vectors, index      a flat searchable list
    tree search    ->  tree, summaries             a navigable structure

Note what is absent here: no vectors, no index, no query_vector. This
architecture has no embedding step and no vector store at all. That is not an
omission, it is the idea.
"""

from typing import Any, Dict, List, Tuple, TypedDict

from shared.document import Document

from pi_rag.node import TreeNode


class PIState(TypedDict, total=False):
    # ---- input -------------------------------------------------------------
    path: str
    extractor: str

    # ---- after load  (identical to the vector pipeline) --------------------
    pages: List[Document]
    n_pages: int
    raw_chars: int
    empty_pages: List[int]

    # ---- after parse -------------------------------------------------------
    document: Document
    page_spans: Dict[int, Tuple[int, int]]
    headings: List[Dict[str, Any]]
    tree: TreeNode
    parse_report: Dict[str, Any]

    # ---- summarisation (not written yet) -----------------------------------
    #  An LLM writes node.summary bottom-up. Tree search navigates by READING
    #  those summaries, so their quality is the retrieval quality.
    summarised_tree: TreeNode

    # ---- query time (not written yet) --------------------------------------
    question: str
    traversal: List[Dict[str, Any]]
    selected_nodes: List[TreeNode]
    context: str
    answer: str

    trace: List[str]
