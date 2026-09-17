"""
shared/graph.py

A small stand-in for LangGraph. Same API, same vocabulary, no install:

    from langgraph.graph import StateGraph, START, END

THE MODEL
---------
1. One state dict flows through the whole pipeline.
2. A node is a plain function:  state -> dict of updates.
3. A node returns only what it changed; the runner merges it in.
4. Edges say what runs next.

Adopting this shape early matters because every later stage is just another
node. Adding one means writing a function and an add_edge line. Writing the
parser as a script with a main() instead means rewriting it at step two.

WHAT THIS VERSION DOES NOT DO
-----------------------------
Real LangGraph has conditional edges, loops, parallel branches, checkpointing
and interrupts. This does linear chains only, which is all an indexing
pipeline needs. Query time wants conditional edges (route the question, retry
on empty results) and that is the point to install the real thing.
"""

from typing import Any, Callable, Dict, List

START = "__start__"
END = "__end__"

NodeFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class StateGraph:
    def __init__(self, state_schema: type = dict):
        self.state_schema = state_schema
        self.nodes: Dict[str, NodeFn] = {}
        self.edges: Dict[str, List[str]] = {}

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name in self.nodes:
            raise ValueError(f"node {name!r} already exists")
        self.nodes[name] = fn
        return self

    def add_edge(self, start_key: str, end_key: str) -> "StateGraph":
        self.edges.setdefault(start_key, []).append(end_key)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        return self.add_edge(START, name)

    def compile(self) -> "CompiledGraph":
        if START not in self.edges:
            raise ValueError("no entry point: call set_entry_point(...)")
        return CompiledGraph(self)


class CompiledGraph:
    def __init__(self, graph: StateGraph):
        self.graph = graph
        self.order = self._linearise()

    def _linearise(self) -> List[str]:
        order, seen = [], set()
        current = self.graph.edges[START][0]
        while current != END:
            if current in seen:
                raise ValueError(f"loop at {current!r}; use real LangGraph")
            seen.add(current)
            order.append(current)
            nxt = self.graph.edges.get(current)
            if not nxt:
                break
            current = nxt[0]
        return order

    def invoke(self, state: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        state = dict(state)
        state.setdefault("trace", [])
        for name in self.order:
            if verbose:
                print(f"    [node] {name}")
            updates = self.graph.nodes[name](state)
            if updates:
                state.update(updates)
            state["trace"].append(name)
        return state
