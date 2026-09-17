"""
shared/goldset.py

Measuring retrieval quality without hand-labelling anything.

THE PROBLEM
-----------
You cannot tell whether retrieval works without knowing the right answer, and
knowing the right answer normally means a human writing queries and marking
which chunks answer them. That is a day of work per corpus, which is why most
projects never do it and tune chunk size by feel instead.

THE SHORTCUT
------------
The document already contains a set of questions with known answers: its own
section headings. "Expense Reimbursement" is a reasonable stand-in for a real
query, and the chunks overlapping that section are, by construction, the
correct results.

That gives a golden set on any structured document, for free, with no labelling.

WHAT THIS IS NOT
----------------
It is not a substitute for real evaluation, and the numbers it produces are
optimistic. Three reasons, all worth holding in mind:

  1. A heading uses the document's own words. Real users use their own, so
     real queries suffer vocabulary mismatch that these do not.
  2. Headings are short noun phrases. Real questions are longer, contain
     filler words, and often ask about something spanning two sections.
  3. The heading text usually appears verbatim inside its own section, so a
     keyword method gets a free exact match that it would not get in practice.

So treat the absolute numbers as an upper bound. What they are genuinely good
for is COMPARISON: the bias applies equally to every strategy, so if hybrid
beats dense on this set, that ordering is meaningful even though both scores
are inflated. Tuning is comparison, which is exactly what this supports.

Replace it with real logged queries as soon as you have any.
"""

import math
from typing import Dict, List, Sequence, Tuple

Ranking = Sequence[Tuple[int, float]]

#  How much of a chunk must lie inside a section before we call that chunk a
#  correct answer for it. Too low and a chunk that merely brushes the section
#  counts; too high and a chunk spanning a boundary is excluded even though it
#  contains the answer.
MIN_CHUNK_FRACTION = 0.30
MIN_OVERLAP_CHARS = 40


def build(tree, chunks) -> List[Dict]:
    """
    One case per leaf section: the heading as query, the overlapping chunks as
    correct answers. Sections that no chunk covers substantially are dropped,
    since an unanswerable case measures nothing.
    """
    cases = []
    for node in tree.walk():
        if not node.is_leaf or node.level == 0:
            continue
        title = node.title
        if node.number and title.startswith(node.number):
            title = title[len(node.number):].strip()
        if len(title) < 3:
            continue

        relevant = []
        for i, chunk in enumerate(chunks):
            start = max(node.char_start, chunk.metadata["char_start"])
            end = min(node.char_end, chunk.metadata["char_end"])
            overlap = max(0, end - start)
            chunk_len = (chunk.metadata["char_end"]
                         - chunk.metadata["char_start"]) or 1
            if overlap >= MIN_OVERLAP_CHARS and \
                    overlap / chunk_len >= MIN_CHUNK_FRACTION:
                relevant.append(i)

        if relevant:
            cases.append({"query": title, "relevant": relevant,
                          "node_id": node.node_id, "pages": node.pages})
    return cases


# =============================================================================
#  METRICS
# =============================================================================
def recall_at_k(ranking: Ranking, relevant: Sequence[int], k: int) -> float:
    """Of the correct chunks, what fraction made the top k?

    The metric that matters most for retrieval. If the answer is not in the
    context, nothing downstream can recover it."""
    if not relevant:
        return 0.0
    got = {i for i, _ in ranking[:k]}
    return len(got & set(relevant)) / len(relevant)


def precision_at_k(ranking: Ranking, relevant: Sequence[int], k: int) -> float:
    """Of what we returned, what fraction was correct? Matters when context
    is expensive: low precision means the model wades through noise."""
    if not k:
        return 0.0
    relevant_set = set(relevant)
    got = [i for i, _ in ranking[:k]]
    return sum(1 for i in got if i in relevant_set) / len(got) if got else 0.0


def reciprocal_rank(ranking: Ranking, relevant: Sequence[int]) -> float:
    """1/rank of the first correct result. Right when there is one answer and
    the reader only looks at the top of the list."""
    relevant_set = set(relevant)
    for position, (i, _) in enumerate(ranking, start=1):
        if i in relevant_set:
            return 1.0 / position
    return 0.0


def hit_at_k(ranking: Ranking, relevant: Sequence[int], k: int) -> float:
    """Did anything correct appear at all? A floor, too forgiving to tune on."""
    return 1.0 if recall_at_k(ranking, relevant, k) > 0 else 0.0


def ndcg_at_k(ranking: Ranking, relevant: Sequence[int], k: int) -> float:
    """
    Are the correct chunks near the TOP, or merely present?

    Recall counts membership and stops there: two correct chunks at ranks 1 and
    2 score the same as the same two at ranks 4 and 5. nDCG discounts each hit
    by log2(position + 1), so a hit further down is worth less, and divides by
    the best arrangement possible for this case. That makes it the most
    complete single number here -- it moves when ranking quality moves, which
    is exactly what reranking changes and recall cannot see.

    1.0 means every correct chunk is as high as it could be.
    """
    if not relevant or not k:
        return 0.0
    relevant_set = set(relevant)
    gain = sum(1.0 / math.log2(position + 1)
               for position, (i, _) in enumerate(ranking[:k], start=1)
               if i in relevant_set)
    ideal = sum(1.0 / math.log2(position + 1)
                for position in range(1, min(len(relevant_set), k) + 1))
    return gain / ideal if ideal else 0.0


def evaluate(cases: Sequence[Dict], rank_fn, k: int) -> Dict[str, float]:
    """
    rank_fn(query) -> ranking. Averaged over every case.
    """
    if not cases:
        return {"cases": 0, "recall": 0.0, "precision": 0.0,
                "mrr": 0.0, "ndcg": 0.0, "hit_rate": 0.0}
    totals = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0,
              "hit_rate": 0.0}
    for case in cases:
        ranking = rank_fn(case["query"])
        relevant = case["relevant"]
        totals["recall"] += recall_at_k(ranking, relevant, k)
        totals["precision"] += precision_at_k(ranking, relevant, k)
        totals["mrr"] += reciprocal_rank(ranking, relevant)
        totals["ndcg"] += ndcg_at_k(ranking, relevant, k)
        totals["hit_rate"] += hit_at_k(ranking, relevant, k)
    n = len(cases)
    out = {key: value / n for key, value in totals.items()}
    out["cases"] = n
    return out


def per_case(cases: Sequence[Dict], rank_fn, k: int) -> List[Dict]:
    rows = []
    for case in cases:
        ranking = rank_fn(case["query"])
        relevant_set = set(case["relevant"])
        first = next((position for position, (i, _) in enumerate(ranking, 1)
                      if i in relevant_set), None)
        rows.append({"query": case["query"],
                     "relevant": case["relevant"],
                     "first_hit_rank": first,
                     "recall": recall_at_k(ranking, case["relevant"], k),
                     "pages": case["pages"]})
    return rows
