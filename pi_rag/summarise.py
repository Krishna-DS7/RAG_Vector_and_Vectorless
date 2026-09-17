"""
pi_rag/summarise.py

Giving every node a summary.

WHY THIS IS THE WHOLE ARCHITECTURE
----------------------------------
Tree search navigates by READING summaries. It never sees the sections
themselves until it has already decided to open one. So a summary is not
documentation -- it is the only evidence the search has when choosing where to
go, and a node whose summary omits a fact is a node the search will walk past.

The failure is specific and worth stating: if a section on expense claims
contains an exclusion, and its summary says "covers expense reimbursement
procedures" rather than "excludes home office equipment, which is claimed
under the equipment allowance", then a question about home office equipment
will be routed INTO that section rather than past it. The retrieval failure
happened during summarisation, hours earlier, and nothing at query time can
detect it.

So a good summary states what the section CONCLUDES, not what it is ABOUT.
Prefer "records are kept seven years then destroyed" over "discusses retention
policy". Include limits, exceptions and figures, because those are what
questions are about.

BOTTOM-UP
---------
Leaves are summarised from their own text. A parent is summarised from its
children's summaries, not from its full text -- cheaper, and it produces a
summary that describes the branch's coverage rather than restating one child at
length.

COST
----
One model call per node. A 300-section document is 300 calls at indexing time,
which is real money but paid once. Compare with embedding, which is one cheap
call per chunk. This is the first genuinely expensive step in either pipeline.
"""

from typing import Dict, List, Optional


def summarise_tree(tree, document_text: str, llm, max_words: int = 45,
                   on_node=None) -> Dict[str, int]:
    """
    Walk the tree depth-first, summarising children before parents.

    Returns counts, and mutates node.summary in place.
    """
    stats = {"leaves": 0, "parents": 0, "calls": 0, "chars_sent": 0}

    def visit(node):
        for child in node.nodes:
            visit(child)

        if node.is_leaf:
            text = node.text(document_text)
            stats["leaves"] += 1
            stats["chars_sent"] += len(text)
            node.summary = llm.summarise(text, node.title, max_words)
        else:
            # Summarise from the children's summaries, not the full subtree.
            joined = " ".join(
                f"{child.title}: {child.summary or ''}" for child in node.nodes)
            stats["parents"] += 1
            stats["chars_sent"] += len(joined)
            node.summary = llm.summarise(joined, node.title, max_words)
        stats["calls"] += 1
        if on_node:
            on_node(node)

    visit(tree)
    return stats


def summary_quality(tree) -> Dict:
    """
    Cheap structural checks on the summaries.

    None of these measure whether a summary is CORRECT -- that needs a human or
    a judge model. They catch the crude failures: a summary that is empty, that
    merely repeats the title, or that is so long it defeats the purpose.
    """
    rows = []
    for node in tree.walk():
        summary = node.summary or ""
        words = len(summary.split())
        title_words = set(node.title.lower().split())
        summary_words = set(summary.lower().split())
        echo = (len(title_words & summary_words) / len(title_words)
                if title_words else 0.0)
        rows.append({"node_id": node.node_id, "title": node.title,
                     "words": words, "title_echo": echo,
                     "empty": not summary.strip()})
    total = len(rows) or 1
    return {"nodes": len(rows),
            "empty": sum(1 for r in rows if r["empty"]),
            "mean_words": sum(r["words"] for r in rows) / total,
            "mostly_title": sum(1 for r in rows if r["title_echo"] > 0.8),
            "rows": rows}


# -----------------------------------------------------------------------------
#  THE NODE  (state -> updates)
# -----------------------------------------------------------------------------
def summarise_node(state: dict) -> dict:
    tree = state["tree"]
    stats = summarise_tree(tree, state["document"].page_content, state["llm"])
    return {"summarised_tree": tree, "summary_stats": stats,
            "summary_quality": summary_quality(tree)}
