"""
rag/rank.py

Steps five and six: reconcile two rankings, then reorder the survivors.

THE PROBLEM FUSION SOLVES
-------------------------
Cosine returns scores bounded at 1. BM25 returns unbounded scores that grow
with query length. They cannot be added, averaged, or compared -- 0.8 and 4.2
are not on the same scale and no amount of min-max normalising fixes it,
because the range shifts with every query.

Reciprocal rank fusion sidesteps the problem by throwing the scores away and
keeping only the POSITIONS:

    score(chunk) = SUM over rankings of  1 / (k + rank_in_that_ranking)

Rank 1 contributes 1/61, rank 2 contributes 1/62, and so on with k=60. Scale
disappears because ranks are already comparable.

WHY k = 60
----------
It is the value from the original paper and it has stuck. What it controls is
how sharply the top of a list is favoured. With k small, rank 1 dwarfs rank 2
and one ranking can dominate. With k large, the curve flattens and positions
20 and 30 become nearly equivalent. At 60 the first few positions matter
clearly while the tail still contributes something. Nobody tunes it, and it is
worth knowing you could.

WHAT FUSION CANNOT DO
---------------------
It combines evidence; it does not add judgement. If both rankings put the same
wrong chunk first, fusion puts it first too -- with more confidence, because
two methods agreed. Fusion is a smoothing operation that damps one system's
isolated mistake. It is not a correction mechanism.

RERANKING IS THE DIFFERENT KIND OF THING
----------------------------------------
Retrieval scores a query and a chunk SEPARATELY and compares the results. That
is what makes it fast: chunk vectors are computed once, at index time. It is
also what makes it shallow -- nothing ever reads the query and the chunk
together.

A cross-encoder does exactly that: it takes the pair, jointly, and scores
relevance. Far more accurate and far too slow to run over a whole corpus,
which is why it runs over the top 20 or 50 that retrieval already shortlisted.
Retrieve wide, rerank narrow.

The reranker here is a lexical stand-in, not a trained cross-encoder. It reads
the pair together -- which is the structural point -- using term coverage and
proximity rather than a learned model. It is labelled as a stand-in wherever
its output appears.
"""

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from shared.llm import content_words

Ranking = Sequence[Tuple[int, float]]

RRF_K = 60


# =============================================================================
def reciprocal_rank_fusion(rankings: Sequence[Ranking], k: int = RRF_K,
                           weights: Optional[Sequence[float]] = None
                           ) -> List[Tuple[int, float]]:
    """
    Merge any number of rankings into one. Scores are discarded; only
    positions are used, which is what makes different scales combinable.
    """
    weights = weights or [1.0] * len(rankings)
    fused: Dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for position, (index, _score) in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + weight / (k + position)
    out = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    return out


def rrf_contribution(position: int, k: int = RRF_K) -> float:
    return 1.0 / (k + position)


# =============================================================================
class LexicalReranker:
    """
    A stand-in for a cross-encoder.

    It scores the query and the chunk TOGETHER, which is the structural
    difference from retrieval, using three signals a trained model would learn
    rather than be told:

        coverage    what fraction of the query's content words appear
        density     how concentrated those matches are in the chunk
        proximity   whether the matched words appear near each other, which
                    is weak evidence they are used in the same claim

    A real cross-encoder handles negation, paraphrase and word order. This
    handles none of those. It exists to show what reranking DOES to a ranking,
    not to show how well reranking works.
    """

    name = "lexical stand-in"
    is_real_model = False

    def score(self, query: str, text: str) -> float:
        query_words = set(content_words(query))
        if not query_words:
            return 0.0
        tokens = content_words(text)
        if not tokens:
            return 0.0

        positions: Dict[str, List[int]] = {}
        for i, token in enumerate(tokens):
            if token in query_words:
                positions.setdefault(token, []).append(i)

        matched = set(positions)
        coverage = len(matched) / len(query_words)
        if not matched:
            return 0.0

        hits = sum(len(v) for v in positions.values())
        density = hits / math.sqrt(len(tokens))

        # How tightly the matched terms cluster: the narrowest window
        # containing one occurrence of each matched term.
        if len(matched) > 1:
            firsts = [min(v) for v in positions.values()]
            lasts = [max(v) for v in positions.values()]
            span = max(lasts) - min(firsts) + 1
            proximity = len(matched) / span
        else:
            proximity = 0.5

        return coverage * 2.0 + density * 0.5 + proximity

    def rerank(self, query: str, candidates: Sequence[Tuple[int, str]]
               ) -> List[Tuple[int, float]]:
        scored = [(index, self.score(query, text)) for index, text in candidates]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored


# -----------------------------------------------------------------------------
#  THE NODES  (state -> updates)
# -----------------------------------------------------------------------------
def fuse_node(state: dict) -> dict:
    fused = reciprocal_rank_fusion(
        [state["dense_ranking"], state["sparse_ranking"]],
        k=state.get("rrf_k", RRF_K))
    top_k = state.get("top_k", 5)
    return {"fused_ranking": fused,
            "retrieved": [state["chunks"][i] for i, _ in fused[:top_k]]}


def rerank_node(state: dict) -> dict:
    """
    Rerank a WIDE shortlist and cut it narrow. The width matters: reranking
    can only reorder what retrieval already found, so a chunk missing from the
    shortlist is unreachable no matter how good the reranker is.
    """
    chunks = state["chunks"]
    ranking = state.get("fused_ranking") or state["dense_ranking"]
    # Retrieve 30-100, rerank, keep 3-5. Below about 20 candidates there is
    # nothing for the reranker to fix -- it can only reorder what it was given
    # -- and above 100 the latency stops paying for itself.
    width = state.get("rerank_width", 50)
    top_k = state.get("top_k", 5)

    shortlist = [(i, chunks[i].page_content) for i, _ in ranking[:width]]
    reranked = LexicalReranker().rerank(state["question"], shortlist)
    return {"reranked_ranking": reranked,
            "reranked": [chunks[i] for i, _ in reranked[:top_k]]}
