"""
pi_rag/tree_search.py

Retrieval by navigation instead of by similarity.

THE PROCEDURE
-------------
Start at the root. Look at the children's titles and summaries. Pick the ones
that could contain the answer. Descend. Repeat until you reach leaves, then
return those sections whole.

It is how a person uses a table of contents, and the mechanics are worth
noticing: at no point is anything compared by distance. There are no vectors,
no index, and no similarity threshold. There is a model reading an outline and
making a decision, several times.

WHAT THIS BUYS
--------------
    whole sections      a leaf is returned entire, so an answer split across
                        two paragraphs is never split across two results
    real citations      every node carries its page range by construction
    auditable path      the traversal is a list of decisions you can read,
                        which is the opposite of a cosine score you cannot

WHAT IT COSTS
-------------
    latency             one model call per hop, sequential and unavoidable
    money               several calls per question, against one cheap vector
                        comparison
    structure           it needs an outline. On a document without one, the
                        root has no children and the whole thing collapses to
                        returning everything
    scale               the outline of a million-document corpus does not fit
                        in a prompt, so this works per-document or per-folder,
                        not over a warehouse

BEAM WIDTH
----------
Picking one child per hop is fast and brittle: one wrong turn at the root and
the answer is unreachable, with no way back. Picking several keeps
alternatives alive. Width 2 is a reasonable default; width 1 is greedy descent
and will fail on any question spanning two branches.

The beam is the width of the WHOLE LEVEL, not a quota per node. That
distinction is the difference between a bounded search and an unbounded one:
if every node on the frontier may keep `beam` children, the frontier is
beam ** depth, so width 3 on a tree six deep reaches 729 nodes and asks the
model 364 questions to answer one. Capping the level instead keeps the cost
linear -- at most `beam` nodes per hop, so at most beam x max_depth model
calls, whatever the shape of the tree.

The cost of capping is that a node's chosen child can be dropped because
better candidates were found elsewhere on the same level. That is a real
pruning decision and it is recorded as one: such branches carry
`cut_by_beam`, so they are distinguishable from branches the model itself
declined.

MULTI-HOP
---------
A question needing two different sections is where this pulls ahead of
chunk-based retrieval. Similarity search ranks every chunk against one query
vector, so the second-topic chunks compete with the first-topic ones and often
lose. A beam keeps both branches alive deliberately.
"""

from typing import Dict, List, Optional, Sequence


def _valid_choices(picked: Sequence, child_count: int, beam: int) -> List[int]:
    """
    Keep only the choices that name a real child, in order, without repeats.

    A model asked for "at most 2 numbers" will sometimes return three, a
    duplicate, a string, or an index that does not exist. Every one of those is
    a crash or a silently doubled branch if it is fed straight to a list
    subscript, so the contract is enforced here rather than trusted. Falling
    back to the first child keeps the descent alive when a reply is entirely
    unusable, which is the difference between a worse answer and no answer.
    """
    seen, out = set(), []
    for choice in picked:
        if not isinstance(choice, int) or isinstance(choice, bool):
            continue
        if not 0 <= choice < child_count or choice in seen:
            continue
        seen.add(choice)
        out.append(choice)
        if len(out) == beam:
            break
    return out or ([0] if child_count else [])


def search(tree, question: str, llm, beam: int = 2, max_depth: int = 6,
           ) -> Dict:
    """
    Walk down the tree, recording every decision.

    Returns the chosen leaves and a hop-by-hop trace, so the path can be read
    rather than inferred.
    """
    trace: List[Dict] = []
    frontier = [tree]
    leaves: List = []
    # Identity, not equality: TreeNode is a mutable dataclass, so `in` would
    # compare whole subtrees field by field and call two distinct sections with
    # the same wording the same node.
    collected: set = set()
    calls = 0
    depth = 0

    def collect(node) -> None:
        if id(node) not in collected:
            collected.add(id(node))
            leaves.append(node)

    while frontier and depth < max_depth:
        depth += 1
        expanded = []

        for node in frontier:
            if node.is_leaf:
                collect(node)
                continue

            options = [{"title": child.title,
                        "summary": child.summary,
                        "pages": child.pages} for child in node.nodes]
            picked = llm.choose(question, options, max_pick=beam)
            calls += 1
            # Never index the tree with a number the model handed back
            # unchecked. A model that invents child 7 of a node with three
            # children crashes the retriever, and it does it in production
            # rather than here, because the offline stand-in never misbehaves.
            picked = _valid_choices(picked, len(node.nodes), beam)
            expanded.append((node, picked))

        # One beam for the level. `picked` is best-first within a node, so rank
        # orders candidates across nodes: every node's first choice is weighed
        # before any node's second.
        candidates = []
        for order, (node, picked) in enumerate(expanded):
            for rank, child_index in enumerate(picked):
                candidates.append((rank, order, node, child_index))
        candidates.sort(key=lambda row: (row[0], row[1]))
        survivors = candidates[:beam]
        kept = {id(node.nodes[child_index])
                for _rank, _order, node, child_index in survivors}

        for node, picked in expanded:
            trace.append({
                "hop": depth,
                "at": node.title,
                "at_id": node.node_id,
                "considered": [
                    {"title": child.title, "node_id": child.node_id,
                     "pages": child.pages,
                     "summary": (child.summary or "")[:110],
                     "chosen": id(child) in kept,
                     "cut_by_beam": i in picked and id(child) not in kept}
                    for i, child in enumerate(node.nodes)],
                "chose": [child.title for child in node.nodes
                          if id(child) in kept],
            })

        frontier = [node.nodes[child_index]
                    for _rank, _order, node, child_index in survivors]

    for node in frontier:
        collect(node)

    # Report hops as decisions taken, not loop iterations. The final pass
    # only collects leaves and makes no choice, so counting it would overstate
    # the work by one on every query.
    return {"leaves": leaves, "trace": trace, "llm_calls": calls,
            "hops": len(trace)}


def pruned_branches(trace: Sequence[Dict]) -> List[Dict]:
    """
    Every branch the search declined to open, for either reason.

    Worth looking at: a pruned branch is a decision made on the strength of a
    summary alone, and it is where this architecture fails silently. If the
    answer was behind one of these, nothing downstream can recover it, and the
    final answer will be confident regardless.

    Two reasons are mixed together here and ``cut_by_beam`` separates them. A
    branch without it lost on its merits: the model read the summary and passed.
    A branch with it was chosen and then dropped because the level's beam was
    full, which is a budget decision rather than a judgement -- and the one to
    look at first, because widening the beam is all it would have taken.
    """
    out = []
    for hop in trace:
        for option in hop["considered"]:
            if not option["chosen"]:
                out.append({"hop": hop["hop"], "at": hop["at"], **option})
    return out


def context_from_leaves(leaves: Sequence, document_text: str,
                        budget_tokens: int = 2000,
                        question: str = "") -> Dict:
    """
    Sections go into the prompt WHOLE. That is the point of the architecture,
    and it is also its budget problem: a section either fits or it does not,
    and there is no equivalent of taking the best half.

    Which means the admission order decides what survives, so it is worth
    choosing. Traversal order is not a relevance order -- it is an artefact of
    where sections sit in the tree -- and spending a fixed budget in it drops
    sections for no better reason than that their branch was walked late.
    Sections are scored against the question and admitted best first, the same
    rule the vector pipeline follows.

    They are then laid out in DOCUMENT order, because whole consecutive
    sections of one document read as a narrative, and a model handed 3.2 before
    2.1 has to reconstruct the sequence before it can use either.

    Without a question there is nothing to score against, so traversal order
    stands.
    """
    from rag.answer import estimate_tokens
    from rag.rank import LexicalReranker

    ranked = list(leaves)
    if question:
        scorer = LexicalReranker()
        ranked.sort(key=lambda node: -scorer.score(
            question, node.text(document_text)))

    admitted, dropped = [], []
    used = 0
    for node in ranked:
        cost = estimate_tokens(" ".join(node.text(document_text).split())) + 12
        if used + cost <= budget_tokens:
            admitted.append(node)
            used += cost
        else:
            dropped.append(node)

    admitted.sort(key=lambda node: node.char_start)
    blocks = [f"[{node.node_id}, {node.pages}] {node.title}\n"
              f"{' '.join(node.text(document_text).split())}"
              for node in admitted]
    return {"context": "\n\n".join(blocks), "admitted": admitted,
            "dropped": dropped, "context_tokens": used}


# -----------------------------------------------------------------------------
#  THE NODES  (state -> updates)
# -----------------------------------------------------------------------------
def tree_search_node(state: dict) -> dict:
    result = search(state["summarised_tree"], state["question"], state["llm"],
                    beam=state.get("beam", 2))
    return {"traversal": result["trace"], "selected_nodes": result["leaves"],
            "tree_llm_calls": result["llm_calls"],
            "pruned": pruned_branches(result["trace"])}


def tree_context_node(state: dict) -> dict:
    built = context_from_leaves(state["selected_nodes"],
                                state["document"].page_content,
                                state.get("budget_tokens", 2000),
                                question=state.get("question", ""))
    return {"context": built["context"], "tree_assembly": built}


def tree_generate_node(state: dict) -> dict:
    from rag.answer import check_support, find_citations, validate_citations

    answer = state["llm"].answer(state["question"], state["context"])
    admitted = state["tree_assembly"]["admitted"]
    texts = [node.text(state["document"].page_content) for node in admitted]

    return {"answer": answer, "support": check_support(answer, texts),
            "citations": find_citations(answer),
            "citation_check": validate_citations(
                answer, [node.node_id for node in admitted])}
