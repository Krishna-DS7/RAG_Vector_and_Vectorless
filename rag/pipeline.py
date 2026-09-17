"""
rag/pipeline.py

Wiring the vector-search pipeline.

Today:

    START -> load -> parse -> chunk -> embed -> index -> END

and at query time:

    retrieve -> fuse -> rerank -> assemble -> generate

Indexing runs once per document; the query graph runs once per question. They
share the vector store and the keyword index and nothing else.

The pipeline is complete. What is left is not more stages but better ones:
query rewriting in front of retrieve, a real cross-encoder in place of the
lexical stand-in, and a router deciding which questions deserve the expensive
path.

To use the real library:

    from langgraph.graph import StateGraph, START, END

Nothing else in this file changes.
"""

from shared.graph import END, START, StateGraph
from shared.loader import load_node

from rag.chunk import chunk_node
from rag.embed import embed_node
from rag.parse import parse_node
from rag.search import index_node, retrieve_node
from rag.state import RAGState

from rag.answer import assemble_node, generate_node
from rag.rank import fuse_node, rerank_node

INDEXING = ["load", "parse", "chunk", "embed", "index"]
QUERY = ["retrieve", "fuse", "rerank", "assemble", "generate"]


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("load", load_node)
    graph.add_node("parse", parse_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed", embed_node)
    graph.add_node("index", index_node)

    graph.add_edge(START, "load")
    graph.add_edge("load", "parse")
    graph.add_edge("parse", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "index")
    graph.add_edge("index", END)

    return graph.compile()


def build_query_graph():
    """
    Query time is a separate graph. It shares the vector store and the keyword
    index with the indexing graph and nothing else -- indexing runs once per
    document, this runs once per question.
    """
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("fuse", fuse_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("assemble", assemble_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "fuse")
    graph.add_edge("fuse", "rerank")
    graph.add_edge("rerank", "assemble")
    graph.add_edge("assemble", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
