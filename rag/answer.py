"""
rag/answer.py

Steps seven and eight: build the prompt, then answer from it.

CONTEXT ASSEMBLY IS NOT CONCATENATION
-------------------------------------
Three decisions get made here and all three change the answer:

    BUDGET    a context window is finite and shared with the question, the
              instructions and the answer. Chunks that do not fit are simply
              absent, and the model will answer anyway without mentioning the
              gap.

    ORDER     models attend unevenly across a long context. Material in the
              middle is used less reliably than material at either end -- the
              "lost in the middle" effect. Putting the best chunk last, right
              before the question, is often better than putting it first.

    LABELS    a chunk with no source marker cannot be cited. If the prompt
              does not carry chunk ids and page numbers, the answer cannot
              either, and every claim becomes unverifiable.

TOKEN COUNTING
--------------
Real tokenisers split on subwords and are model specific. Four characters per
token is the usual rough figure for English prose and is what this uses. It is
an estimate: code, tables and non-English text tokenise far worse, sometimes
at one token per character. Treat the budget as approximate and leave headroom.

GENERATION IS WHERE FAILURES BECOME INVISIBLE
---------------------------------------------
Every earlier stage fails loudly enough to measure. This one fails fluently.
The model will produce a confident, well-formed answer whether the context
contained the answer, half the answer, or a contradiction. The three failure
modes worth naming:

    unsupported     a claim not present in any retrieved chunk
    partial         a correct-sounding answer built from half the evidence,
                    because the other half was never retrieved
    unrefused       an answer to a question the context cannot support, given
                    instead of "the document does not say"

None of the three raises an error. All three require either a human or a
second model to detect.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from shared.document import Document

CHARS_PER_TOKEN = 4.0

#  The role goes first so the model knows what it is reading before it reads
#  it; the rules go last, after the context. A model follows the most recent
#  instruction more reliably, and these particular rules -- cite everything,
#  refuse rather than guess -- are the ones that decay when they sit thousands
#  of tokens above the material they govern.
ROLE_PROMPT = ("You are a document assistant. Below is context retrieved from "
               "one document, followed by a question about it.")

SYSTEM_PROMPT = """Answer the question using ONLY the context above.

Rules:
- Cite the source marker for every claim, like [c0003, p8-10].
- If the context does not contain the answer, say so plainly. Do not guess.
- If two sources disagree, say that rather than picking one.
- Do not add information from outside the context, however obvious it seems."""


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def order_chunks(chunks: Sequence[Document], strategy: str = "best_first"
                 ) -> List[Document]:
    """
    chunks arrive best-first from retrieval.

        best_first   descending relevance: best chunk first, worst last. The
                     default, because it is simple and predictable and the
                     model sees the strongest evidence before anything else.
        edges        sandwich, or edge-loading: best first, second-best last,
                     weakest buried in the middle. The two positions attention
                     favours, so this is the one to reach for once the context
                     is long enough that the middle is genuinely being lost.
        document     source order, by character offset. The right choice when
                     the chunks are sequential and narrative -- a procedure, a
                     legal argument, a timeline -- because reading steps 1, 2
                     and 3 out of order confuses a model exactly as it would
                     confuse a reader.
        best_last    the reverse of best_first, putting the strongest chunk
                     immediately before the question. A blunter version of
                     `edges`, kept because it isolates the end position.
    """
    ordered = list(chunks)
    if strategy == "best_first":
        return ordered
    if strategy == "best_last":
        return ordered[::-1]
    if strategy == "document":
        return sorted(ordered, key=lambda c: c.metadata.get("char_start", 0))
    if strategy == "edges":
        out: List[Document] = []
        tail: List[Document] = []
        for i, chunk in enumerate(ordered):
            (out if i % 2 == 0 else tail).append(chunk)
        return out + tail[::-1]
    raise ValueError(f"unknown ordering {strategy!r}")


def assemble(question: str, chunks: Sequence[Document],
             budget_tokens: int = 2000,
             ordering: str = "best_first") -> Dict:
    """
    Fit as many chunks as the budget allows, then lay them out.

    Chunks are admitted best-first so the budget is spent on the strongest
    material, and only then reordered for position. Dropping happens silently
    in most implementations; here it is reported, because "the answer was in
    chunk five and chunk five did not fit" is a failure worth seeing.

    The rules are placed AFTER the context, not before it: see ROLE_PROMPT.
    """
    overhead = (estimate_tokens(ROLE_PROMPT) + estimate_tokens(SYSTEM_PROMPT)
                + estimate_tokens(question) + 40)
    remaining = budget_tokens - overhead
    if remaining <= 0:
        raise ValueError(
            f"budget_tokens ({budget_tokens}) leaves no room for context: the "
            f"instructions and question alone need about {overhead} tokens. "
            "Raise the budget or shorten the question.")

    admitted: List[Document] = []
    dropped: List[Document] = []
    used = 0
    for chunk in chunks:
        marker = f"[{chunk.metadata.get('chunk_id')}, " \
                 f"p{chunk.metadata.get('start_page')}]"
        cost = estimate_tokens(marker + chunk.page_content) + 4
        if used + cost <= remaining:
            admitted.append(chunk)
            used += cost
        else:
            dropped.append(chunk)

    laid_out = order_chunks(admitted, ordering)
    blocks = []
    for chunk in laid_out:
        meta = chunk.metadata
        pages = (f"p{meta.get('start_page')}"
                 if meta.get("start_page") == meta.get("end_page")
                 else f"p{meta.get('start_page')}-{meta.get('end_page')}")
        blocks.append(f"[{meta.get('chunk_id')}, {pages}]\n"
                      f"{' '.join(chunk.page_content.split())}")
    context = "\n\n".join(blocks)
    prompt = (f"{ROLE_PROMPT}\n\nContext:\n{context}\n\n"
              f"{SYSTEM_PROMPT}\n\nQuestion: {question}")

    return {"context": context, "prompt": prompt, "admitted": laid_out,
            "dropped": dropped, "context_tokens": used,
            "prompt_tokens": estimate_tokens(prompt),
            "budget_tokens": budget_tokens, "ordering": ordering}


# =============================================================================
def check_support(answer: str, passages: Sequence[str]) -> Dict:
    """
    A crude groundedness check: what fraction of the answer's content words
    appear anywhere in the context.

    ``passages`` is the text that was actually sent to the model. Plain strings,
    not Documents, because a tree pipeline sends whole sections and a vector
    pipeline sends chunks, and this check has no interest in the difference.

    This is not faithfulness measurement. A high score can still describe a
    fabricated claim assembled from context vocabulary, and a low score can
    describe a correct paraphrase. Real evaluation asks a judge model whether
    each claim is entailed by the context. This catches only the crude case --
    an answer talking about things the context never mentions.
    """
    import re
    from shared.llm import content_words

    # Strip the citation markers first. "[c0005, p9-12]" is a source label the
    # prompt asked for, not a claim, and counting it as unsupported vocabulary
    # makes every well-cited answer look partly fabricated.
    stripped = re.sub(r"\[[^\]]*\]", " ", answer)
    answer_words = set(content_words(stripped))
    if not answer_words:
        return {"coverage": 0.0, "unsupported": []}
    context_words = set()
    for passage in passages:
        context_words.update(content_words(passage))
    unsupported = sorted(answer_words - context_words)
    coverage = 1 - len(unsupported) / len(answer_words)
    return {"coverage": coverage, "unsupported": unsupported}


def find_citations(answer: str) -> List[str]:
    import re
    return re.findall(r"\[([a-z]?\d+[^\]]*)\]", answer)


def validate_citations(answer: str, sent: Sequence[str]) -> Dict:
    """
    Do the answer's citations point at sources that were actually sent?

    This is the one generation check that needs no judge model and no
    labelling: parse the markers, and confirm each resolves to an id that went
    into the prompt. A marker naming a chunk the model never saw is invented,
    and an invented citation is worse than none -- it reads as evidence and
    survives exactly the skim-reading a citation is supposed to reward.

    ``sent`` is the source ids that were in the prompt (chunk ids, node ids).
    It cannot tell whether a cited source SUPPORTS the claim attached to it;
    that needs a judge. It only catches the citation that resolves to nothing.
    """
    known = set(sent)
    cited, unknown = [], []
    for marker in find_citations(answer):
        source_id = marker.split(",")[0].strip()
        cited.append(source_id)
        if source_id not in known:
            unknown.append(source_id)
    return {"cited": cited, "unknown": unknown,
            "valid": not unknown, "sources_sent": len(known)}


# -----------------------------------------------------------------------------
#  THE NODES  (state -> updates)
# -----------------------------------------------------------------------------
def assemble_node(state: dict) -> dict:
    chunks = state.get("reranked") or state.get("retrieved") or []
    built = assemble(state["question"], chunks,
                     budget_tokens=state.get("budget_tokens", 2000),
                     ordering=state.get("ordering", "best_last"))
    return {"context": built["context"], "prompt": built["prompt"],
            "assembly": built}


def generate_node(state: dict) -> dict:
    llm = state["llm"]
    chunks = state["assembly"]["admitted"]
    answer = llm.answer(state["question"], state["context"])
    return {"answer": answer,
            "support": check_support(answer, [c.page_content for c in chunks]),
            "citations": find_citations(answer),
            "citation_check": validate_citations(
                answer, [c.metadata.get("chunk_id") for c in chunks])}
