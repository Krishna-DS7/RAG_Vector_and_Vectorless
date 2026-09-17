"""
rag/state.py

What flows through the vector-search pipeline.

Declaring the state as a TypedDict is documentation, not enforcement: it is
the one place to look to see what the pipeline knows at any point. Fields are
grouped by the step that fills them, and the steps not written yet are listed
so the shape of the finished system is visible from the start.

    from langgraph.graph import StateGraph
    StateGraph(RAGState)          # the real library takes this same class
"""

from typing import Any, Dict, List, Tuple, TypedDict

from shared.document import Document


class RAGState(TypedDict, total=False):
    # ---- input -------------------------------------------------------------
    path: str
    extractor: str

    # ---- after load --------------------------------------------------------
    pages: List[Document]
    n_pages: int
    raw_chars: int
    empty_pages: List[int]

    # ---- after parse -------------------------------------------------------
    document: Document                        # the whole cleaned text
    page_documents: List[Document]            # per-page alternative
    page_spans: Dict[int, Tuple[int, int]]    # page -> char range in document
    parse_report: Dict[str, Any]

    # ---- chunk settings (inputs) -------------------------------------------
    chunk_strategy: str                       # fixed | recursive | structural
    chunk_size: int
    chunk_overlap: int
    section_boundaries: List[int]             # only used by structural

    # ---- after chunk -------------------------------------------------------
    chunks: List[Document]
    chunk_spans: List[Tuple[int, int]]
    chunk_settings: Dict[str, Any]

    # ---- embed settings (inputs) -------------------------------------------
    embed_backend: str                        # tfidf | svd
    embed_dimensions: int                     # components, for svd

    # ---- after embed -------------------------------------------------------
    embedder: Any                             # keeps the vocabulary and idf
    vectors: List[Dict[int, float]]           # sparse: term index -> weight
    vocabulary: Any
    embed_settings: Dict[str, Any]

    # ---- after index -------------------------------------------------------
    bm25: Any                                 # the keyword index
    index_cost: Dict[str, float]
    bm25_k1: float
    bm25_b: float

    # ---- query time --------------------------------------------------------
    top_k: int

    question: str
    query_vector: Dict[int, float]
    dense_ranking: List[Tuple[int, float]]
    sparse_ranking: List[Tuple[int, float]]
    retrieved: List[Document]

    # ---- not written yet ---------------------------------------------------
    reranked: List[Document]
    context: str
    answer: str

    trace: List[str]
