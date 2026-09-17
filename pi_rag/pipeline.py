"""
pi_rag/pipeline.py

Wiring the tree-navigation pipeline.

Today:

    START -> load -> parse -> END

Remaining steps, in the order they get written:

    summarise     an LLM writes node.summary, bottom-up
    ----------------------------------------------------- indexing ends here
    tree_search   an LLM walks the outline, hop by hop
    select        gather the chosen nodes whole
    generate      answer, citing page ranges

That list is much shorter than the vector pipeline's. There is no chunk step,
no embed step, no vector store and no nearest-neighbour index; all four are
replaced by one summarisation pass and a model that can read an outline.

The cost moves to query time: every search is several LLM calls, so it is
slower and far more expensive per question than a vector lookup.
"""

from shared.graph import END, START, StateGraph
from shared.loader import load_node

from pi_rag.parse import parse_node
from pi_rag.state import PIState
from pi_rag.summarise import summarise_node
from pi_rag.tree_search import (tree_context_node, tree_generate_node,
                                tree_search_node)

INDEXING = ["load", "parse", "summarise"]
QUERY = ["tree_search", "select", "generate"]


def build_graph():
    graph = StateGraph(PIState)
    graph.add_node("load", load_node)       # the same node the vector side uses
    graph.add_node("parse", parse_node)

    graph.add_node("summarise", summarise_node)

    graph.add_edge(START, "load")
    graph.add_edge("load", "parse")
    graph.add_edge("parse", "summarise")
    graph.add_edge("summarise", END)

    return graph.compile()


def build_query_graph():
    graph = StateGraph(PIState)
    graph.add_node("tree_search", tree_search_node)
    graph.add_node("select", tree_context_node)
    graph.add_node("generate", tree_generate_node)

    graph.add_edge(START, "tree_search")
    graph.add_edge("tree_search", "select")
    graph.add_edge("select", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
