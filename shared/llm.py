"""
shared/llm.py

The three places this project needs a language model, and a way to run without
one.

WHAT ACTUALLY NEEDS A MODEL
---------------------------
    summarise(text, title)        writing a node summary for the tree index
    choose(question, options)     picking which branch of the tree to descend
    answer(question, context)     writing the final answer from retrieved text

Everything else -- parsing, chunking, embedding, retrieval, fusion, ranking --
is arithmetic and runs offline.

THE OFFLINE FALLBACK, AND WHAT IT IS NOT
----------------------------------------
ExtractiveLLM implements those three methods without any model, using sentence
selection and term overlap. It exists so the pipeline runs end to end on a
laptop with no key, so the mechanics are visible, and so the tests are
deterministic.

It is not a language model and the output says so. Specifically:

    summarise   returns the opening sentence plus the section's distinctive
                terms. A real summary states what the section CONCLUDES;
                this states what it is ABOUT. The difference matters most
                exactly where tree search is supposed to shine -- a section
                whose summary should say "excludes home office equipment"
                will instead say "covers expense reimbursement", and the
                search will walk into it rather than past it.

    choose      scores each option's title and summary against the question
                lexically. A real model reasons about whether a branch could
                CONTAIN the answer, which is a different question from
                whether it shares words with it.

    answer      returns the sentences from the context that best match the
                question, stitched together. It cannot synthesise across two
                chunks, cannot say "the document does not answer this", and
                cannot resolve a contradiction between sources.

Read every offline result as "this is the shape of the output", never as a
demonstration that the approach works. Plug in a real model to judge quality.

PLUGGING IN A REAL MODEL
------------------------
Implement the same three methods. An OpenAI version is written out below and
left unregistered because this environment has no key and no network to test
it -- shipping untested network code would be worse than shipping none.
"""

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "its", "many", "may", "much",
    "must", "my", "not", "of", "on", "or", "shall", "should", "that", "the",
    "their", "there", "this", "to", "was", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your",
}


def sentences(text: str) -> List[str]:
    parts = [s.strip() for s in SENTENCE_RE.split(" ".join(text.split()))]
    return [s for s in parts if len(s) > 2]


def content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in STOPWORDS and len(w) > 2]


def overlap_score(query: str, text: str) -> float:
    """Fraction of the query's content words present in the text."""
    q = set(content_words(query))
    if not q:
        return 0.0
    t = set(content_words(text))
    return len(q & t) / len(q)


# =============================================================================
class ExtractiveLLM:
    """No model. Deterministic. Clearly labelled everywhere it is used."""

    name = "extractive"
    is_real_model = False
    cost_per_1k_input = 0.0
    cost_per_1k_output = 0.0

    def summarise(self, text: str, title: str = "", max_words: int = 45) -> str:
        body = sentences(text)
        if not body:
            return title or "(empty section)"
        opening = body[0]
        words = opening.split()
        if len(words) > max_words:
            opening = " ".join(words[:max_words]) + "..."

        # Add distinctive terms from the rest, so the summary at least hints at
        # what the later sentences cover.
        seen = set(content_words(opening))
        extra = []
        for sentence in body[1:]:
            for word in content_words(sentence):
                if word not in seen and word not in extra:
                    extra.append(word)
        tail = ", ".join(extra[:8])
        return f"{opening} Also covers: {tail}." if tail else opening

    def choose(self, question: str, options: Sequence[Dict],
               max_pick: int = 2) -> List[int]:
        """
        options: [{"title": str, "summary": str}, ...]
        Returns indices, best first. Lexical, not reasoned.
        """
        if not options:
            return []
        scored = []
        for i, option in enumerate(options):
            text = f"{option.get('title', '')} {option.get('summary') or ''}"
            scored.append((overlap_score(question, text), -i, i))
        scored.sort(reverse=True)
        picked = [i for score, _, i in scored if score > 0][:max_pick]
        return picked or [scored[0][2]]

    def answer(self, question: str, context: str, max_sentences: int = 3) -> str:
        candidates = sentences(context)
        if not candidates:
            return "No context was retrieved, so there is nothing to answer from."
        scored = sorted(((overlap_score(question, s), -i, s)
                         for i, s in enumerate(candidates)), reverse=True)
        picked = {-position for score, position, _ in scored[:max_sentences]
                  if score > 0}
        if not picked:
            return ("Nothing in the retrieved context shares wording with the "
                    "question. A real model would say it cannot answer; this "
                    "stand-in can only report the absence.")
        # Select by POSITION, not by text. Matching on the string repeats a
        # sentence that appears twice in the context -- and a repeated sentence
        # is normal, because retrieved chunks overlap by design.
        return " ".join(s for i, s in enumerate(candidates) if i in picked)


# =============================================================================
#  A REAL BACKEND
# =============================================================================
#  Written out rather than shipped: this environment has no key and no network
#  to reach the service, so it has never been executed. Uncomment, install the
#  client, set the key, and register it in BACKENDS.
#
#      class OpenAILLM:
#          name, is_real_model = "openai", True
#          cost_per_1k_input, cost_per_1k_output = 0.00015, 0.0006
#
#          def __init__(self, model="gpt-4o-mini"):
#              from openai import OpenAI
#              self.client, self.model = OpenAI(), model
#
#          def _complete(self, system, user, max_tokens=300):
#              reply = self.client.chat.completions.create(
#                  model=self.model, max_tokens=max_tokens, temperature=0,
#                  messages=[{"role": "system", "content": system},
#                            {"role": "user", "content": user}])
#              return reply.choices[0].message.content.strip()
#
#          def summarise(self, text, title="", max_words=45):
#              return self._complete(
#                  "Summarise this document section in one or two sentences. "
#                  "State what it CONCLUDES or REQUIRES, not what it is about. "
#                  "Include any exclusion, limit or exception, because a reader "
#                  "will use this summary to decide whether to open the section.",
#                  f"Section: {title}\n\n{text}")
#
#          def choose(self, question, options, max_pick=2):
#              listing = "\n".join(
#                  f"{i}. {o['title']}: {o.get('summary') or '(no summary)'}"
#                  for i, o in enumerate(options))
#              reply = self._complete(
#                  "Pick the sections most likely to CONTAIN the answer. "
#                  f"Reply with at most {max_pick} numbers, comma separated, "
#                  "nothing else.",
#                  f"Question: {question}\n\nSections:\n{listing}")
#              picked = [int(n) for n in re.findall(r"\d+", reply)
#                        if int(n) < len(options)]
#              return picked[:max_pick] or [0]
#
#          def answer(self, question, context, max_sentences=3):
#              return self._complete(
#                  "Answer using ONLY the context. Cite the chunk or page for "
#                  "each claim. If the context does not contain the answer, say "
#                  "so plainly rather than guessing.",
#                  f"Question: {question}\n\nContext:\n{context}")
# =============================================================================

BACKENDS = {"extractive": ExtractiveLLM}


def get_llm(name: str = "extractive"):
    if name not in BACKENDS:
        available = ", ".join(BACKENDS)
        raise ValueError(f"unknown llm backend {name!r}; available: {available}")
    return BACKENDS[name]()


def warn_if_stub(llm) -> Optional[str]:
    if llm.is_real_model:
        return None
    return ("Running without a language model. Summaries, branch choices and "
            "answers below come from a deterministic extractive stand-in, not "
            "from a model. The shapes are real; the quality is not. See "
            "shared/llm.py to plug in a real backend.")
