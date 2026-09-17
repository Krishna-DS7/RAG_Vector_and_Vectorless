# Document parsing for retrieval: two architectures, side by side

Give it a PDF. It processes the document two ways — once the way chunk-and-embed
retrieval needs, once the way tree navigation needs — and reports what each
approach was able to keep.

Everything printed is measured from your file. Nothing is canned.

```bash
python3 parse.py report.pdf      # step 1: load, clean, structure
python3 chunk.py report.pdf      # step 2: cut it up, measure what that cost
python3 embed.py report.pdf      # step 3: turn chunks into vectors
python3 search.py report.pdf     # step 4: retrieve, and score the retrieval
python3 ask.py report.pdf        # end to end: a question, answered both ways
```

## Start with the workbook

`RAG_Course_Vector_and_Vectorless.xlsx` is the same pipeline worked by hand:
33 sheets of live formulas, one per step, from the term matrix through cosine,
BM25, rank fusion, reranking and tree search. Change the query in the control
panel on sheet 00 and every downstream sheet recomputes.

It is the better place to begin. The code shows you a working system; the
workbook shows you why each step is there, with the arithmetic visible rather
than executed. It is also the specification this code is written against —
where the two ever disagree, the workbook is right.

## Install

Python 3.8+. For PDFs, one extractor:

```bash
pip install pypdf          # or: pip install pdfplumber
```

`.txt` and `.md` need nothing, and neither does the built-in sample.

## Usage

```bash
python3 parse.py report.pdf                    # full report
python3 parse.py                               # built-in sample, zero deps
python3 parse.py doc.pdf --show vector         # one pipeline only
python3 parse.py doc.pdf --show tree
python3 parse.py doc.pdf --extractor pdfplumber
python3 parse.py doc.pdf --out results/        # write artefacts to disk
python3 parse.py doc.pdf --quiet               # verdict only

python3 verify.py                              # 68 self-tests
python3 verify.py doc.pdf                      # structural checks on your file
```

`--out` writes `cleaned_text.txt`, `tree.json` and `report.json`.

## Chunking

```bash
python3 chunk.py doc.pdf                       # recursive, 800/100
python3 chunk.py doc.pdf --size 500 --overlap 50
python3 chunk.py doc.pdf --strategy fixed      # fixed | recursive | structural
python3 chunk.py doc.pdf --compare             # all three, damage side by side
python3 chunk.py doc.pdf --show 3              # print chunks in full
python3 chunk.py doc.pdf --out results/
```

Three strategies:

| | cuts at |
|---|---|
| `fixed` | every N characters, no exceptions — the honest baseline |
| `recursive` | the best paragraph, sentence or word break near the limit |
| `structural` | section boundaries first, splitting further only when needed |

`--compare` is the one to run. It reports, for your document:

- **split** — sections no single chunk can return whole
- **mixed** — chunks carrying substantial text from two or more sections
- **ragged** — chunks that stop partway through a sentence
- **mid-word** — chunks that start partway through a word

The first two measure structural damage; the last two measure cut quality.
Recursive splitting fixes cut quality and does nothing for structural damage,
because it does not know where a section ends. Structural chunking fixes both —
using the outline the vector pipeline discarded during parsing.

That comparison only works because the tool runs both pipelines. **A production
chunk-and-embed system is blind to every number in it.** The damage is not hard
to measure; it is hard to measure from inside the architecture that causes it.

## Embedding

```bash
python3 embed.py doc.pdf
python3 embed.py doc.pdf --query "how long are records kept"
python3 embed.py doc.pdf --backend svd --dimensions 96   # dense, needs numpy
python3 embed.py doc.pdf --terms 16                      # wider term matrix
```

The default backend is TF-IDF over **your document's own vocabulary** — real
retrieval maths, pure Python, and every dimension is a word you can point at.
The report walks the whole conversion: vocabulary and IDF weights, the raw term
matrix, one chunk shown as a vector before and after normalisation, and the
dot / magnitude / cosine / angle / euclidean relationships between two real
chunks from your file.

`--backend svd` rotates those sparse vectors into dense ones with numpy. Same
maths, same reports, different results — useful for seeing what dense buys you
and what it costs in readability.

Two things the report will show you on your own document:

- **Query terms that don't exist in the vocabulary are silently dropped.**
  Ask "how long are records kept" and `how` and `long` vanish. You still get
  results; they just weren't scored on everything you asked. This is the
  sharpest limit of sparse retrieval and it's invisible at query time.
- **Absolute cosine values mean nothing on their own.** They depend on query
  length, chunk length, and which terms happen to be rare in your file. Compare
  within one query's results, never across queries, and never set a fixed
  relevance threshold.

## What the report tells you

**1. Source** — pages, characters, and an excerpt of what the extractor
actually returned. If every page comes back empty it says so: that's a scanned
document and you need `ocrmypdf` before anything here can help.

**2. Cleaning** — what was removed and why: running headers, contents pages,
hyphenated line breaks, sentences rejoined across page boundaries. Each one is
reported with the actual strings from your file.

**3. Vector pipeline** — one Document, ready to chunk. Shows what the
page-boundary decision cost, using a real broken sentence from your document if
it has one.

**4. Tree pipeline** — which numbering convention your document uses and how
that was decided, the outline it produced, and a proof that the recorded
section boundaries actually work by slicing a section back out.

**5. Verdict** — a comparison table and a plain judgement about which
architecture suits this document.

## Retrieval

```bash
python3 search.py doc.pdf --query "how long are records kept"
python3 search.py doc.pdf --top-k 3
python3 search.py doc.pdf --k1 1.2 --b 0.5      # BM25 saturation and length
python3 search.py doc.pdf --evaluate            # score it, no labelling needed
python3 search.py doc.pdf --backend svd
```

Two independent rankings — cosine over the vectors, BM25 over the tokens — plus
what they cost and where they disagree. The report covers:

- **Top-K sensitivity.** Recall and precision at K = 1, 2, 3, 5, 10, against
  the chunks that genuinely answer the query. K is usually set to whatever the
  tutorial said; this shows what it costs on your document.
- **BM25 taken apart.** Per-term contributions that sum exactly to the score,
  and what happens to the ranking as `k1` and `b` move.
- **Index cost.** N × D per query for your corpus, extrapolated to 100k and
  10M, then what IVF, HNSW and quantisation trade away to avoid it.
- **The metadata filtering trap.** Post-filtering silently returns fewer
  results than you asked for, because the filter runs on an already-truncated
  list.

### Measuring quality without labelling anything

`--evaluate` builds a golden set from your document's own section headings:
query = the heading, correct answers = the chunks covering that section. That
gives recall@K, precision@K, MRR, nDCG@K and hit rate on any structured PDF
with no manual labelling. Read nDCG first when comparing two methods: recall
saturates at 100% and stops moving, while nDCG keeps scoring how high the
correct chunks sit.

**These numbers are optimistic and the tool says so.** A heading uses the
document's own words, so there is no vocabulary mismatch, and the heading text
usually appears verbatim inside its own section — a free exact match a keyword
method would never get from a real user's phrasing. Read them as an upper
bound.

What they are genuinely good for is *comparison*. The same bias applies to
every method, so the ordering is meaningful even when the absolute values are
inflated. Change chunk size, backend or K and re-run: the direction of movement
is trustworthy. Replace it with real logged queries as soon as you have any.

### One result worth knowing before you build hybrid search

With the TF-IDF backend, cosine and BM25 agree on ~100% of the top K on most
documents. That is not confirmation that retrieval is working — it is because
**both are bag-of-words methods counting the same tokens over the same
vocabulary.** They weight the counts differently but cannot disagree about
which words are present.

Hybrid search is justified by combining *independent* evidence. Fusing two
bag-of-words rankings mostly buys a smoother version of the same answer. Switch
to `--backend svd` and agreement drops (60% on the test policy PDF), because
the dense side sees which words co-occur rather than only which are present.
On a query phrased as "what happens if data is leaked", TF-IDF ranked the
retention section first; SVD ranked incident response first — the better
answer.

The tool reports which situation you are in rather than assuming the second.

## Asking a question

```bash
python3 ask.py doc.pdf --query "how long are records kept"
python3 ask.py doc.pdf --pipeline vector     # or tree
python3 ask.py doc.pdf --show-prompt         # the literal prompt sent
python3 ask.py doc.pdf --beam 3 --top-k 5
python3 ask.py doc.pdf --budget 800 --ordering edges
```

The whole system, twice over:

- **vector** — retrieve by cosine and BM25, fuse with reciprocal rank fusion,
  rerank the shortlist, assemble a prompt inside a token budget, answer
- **tree** — summarise every section, navigate the outline hop by hop, return
  the chosen sections whole, answer

Four orderings decide where material sits in the prompt, which matters because
models attend unevenly across a long context:

| `--ordering` | arrangement | when |
|---|---|---|
| `best_first` | descending relevance | the default: simple and predictable |
| `edges` | best first, second-best last, weakest buried | long contexts, where the middle is genuinely lost |
| `document` | source order | sequential material — a procedure, an argument, a timeline |
| `best_last` | strongest immediately before the question | a blunter `edges` |

Both pipelines report a **citation check**: every marker in the answer is
resolved against the sources actually sent, so an invented `[c9999]` is caught
without a judge model. It cannot tell you whether a real source supports the
claim attached to it — only that the source existed.

The report shows the ranking moving through each stage, the traversal
hop-by-hop with the branches it declined to open, both answers side by side,
and a head-to-head comparison including where the model calls go.

### About the language model

Three things genuinely need one: writing node summaries, choosing which branch
to descend, and writing the final answer. Everything else is arithmetic.

Without a key, those three run on a **deterministic extractive stand-in**. It
selects sentences rather than writing them; it cannot synthesise across two
sources, cannot refuse, and cannot tell that "retained" answers a question
about "kept". Every affected line in the output says so. Judge the retrieval,
not the prose.

`shared/llm.py` has a working OpenAI implementation written out in comments —
three methods, about twenty lines. It is left unregistered because this was
built without a key or network access to that service, and shipping untested
network code seemed worse than shipping none.

### Two failure modes worth internalising

They are not symmetric, and that matters more than which pipeline wins:

- **Vector retrieval fails by returning a fragment that reads as complete.**
  Detectable by reading the context.
- **Tree search fails by pruning a branch it should have opened.** Leaves no
  trace in the output at all. The answer is exactly as confident either way.

`ask.py` prints every declined branch for this reason.

## Layout

```
parse.py         entry point: parse, chunk, embed, search and ask are the
                 others, one per stage
verify.py        68 self-tests

shared/          both pipelines use these, unchanged
  document.py      Document — same shape as LangChain's
  loader.py        FileLoader — same contract as PyPDFLoader
  text_ops.py      cleaning primitives
  headings.py      structure detection, five conventions, auto-selected
  measure.py       chunking damage, measured against the outline
  goldset.py       evaluation cases built from the document's own headings
  llm.py           pluggable model, with a deterministic offline fallback
  graph.py         stand-in for LangGraph
  sample.py        built-in document for when no file is given

rag/             chunk-and-embed pipeline
  state.py         every field the finished pipeline will hold
  parse.py         two page-boundary strategies and what each costs
  chunk.py         three chunking strategies, over character offsets
  embed.py         TF-IDF and SVD backends, plus the vector maths
  search.py        cosine ranking, BM25, index cost, metadata filtering
  rank.py          reciprocal rank fusion, reranking
  answer.py        context assembly under a budget, generation, grounding
  pipeline.py      indexing: load -> parse -> chunk -> embed -> index
                   query:    retrieve -> fuse -> rerank -> assemble -> generate

pi_rag/          tree-navigation pipeline
  state.py         note what's absent: no vectors, no index
  node.py          TreeNode, following the PageIndex schema
  parse.py         same cleaning, then keeps the outline
  summarise.py     an LLM writes every node's summary, bottom-up
  tree_search.py   navigate the outline, beam search, hop-by-hop trace
  pipeline.py      indexing: load -> parse -> summarise
                   query:    tree_search -> select -> generate
```

## The central point

Both pipelines call the *same* cleaning functions in the *same* order and
produce byte-identical text. `verify.py` asserts this rather than asking you to
believe it. They differ only in what they retain afterwards.

Tree navigation is not a different way of reading a file. It's the same
reading, followed by a refusal to discard the outline.

One claim worth getting right: **"chunk-based retrieval can't cite pages" is
false.** Page numbers are recoverable from a span map, and most loaders keep
them anyway. What it genuinely cannot keep is *hierarchy* — which section
contains which, where a section stops, how to return one whole. A flat list of
chunks has nowhere to store that.

## Structure detection

Five conventions are understood: `A.3.2`, `1.2.3`, `Chapter 4` / `Part II`,
markdown `##`, and short ALL-CAPS lines. Each is scored against your document by
matches × depth, and the best is chosen. The report shows the scoring.

If your document uses something else, adding a matcher to `shared/headings.py`
is about fifteen lines — the existing ones are the template.

**A document with no outline gets a one-node tree, and that's correct.** Novels,
essays and forms have no structure to navigate, so tree retrieval has no
advantage on them. Chunk-and-embed does. Knowing which kind of document you're
holding is the actual decision.

## Built to extend

`Document(page_content, metadata)` and the loader's `.load()` / `.lazy_load()`
match LangChain, so swapping in the real library is an import change.

Nodes are functions `state -> dict of updates`, and `shared/graph.py` uses
LangGraph's vocabulary (`add_node`, `add_edge`, `set_entry_point`, `compile`,
`invoke`, `START`, `END`). Adding the next stage is one function and one
`add_edge` — both `pipeline.py` files show the exact lines in a comment.

Each `state.py` declares the fields for stages not yet written. Read those
first; they're the roadmap.

Both pipelines are complete: question in, answer out, end to end. What is left
is not more stages but better ones — query rewriting in front of retrieval, a
trained cross-encoder in place of the lexical reranker, and a router deciding
which questions deserve the expensive path.

## Experiments

Each is a real bug that ships regularly, and none raises an exception. Run
`verify.py` after each to see what catches it.

1. `shared/text_ops.py` — remove `"one-time"` from `KEEP_HYPHEN`. The word
   becomes `onetime`, a token no query contains.
2. `shared/text_ops.py` — make `continues_sentence` always return `False`.
   Every sentence straddling a page break silently splits in two.
3. `rag/parse.py` — in `parse_merged`, detect furniture *before* normalising.
   The running header survives into every chunk because an em-dash no longer
   matches a hyphen.
4. `shared/headings.py` — delete the `TOC_TAIL_RE` guard. Contents entries
   become sections that look entirely normal and contain nothing.
5. Compare extractors: `--extractor pdfplumber` against the pypdf default.
   Different libraries return different text for the same file, and everything
   downstream inherits the difference.
6. `chunk.py doc.pdf --compare`, then again at `--size 400` and `--size 1500`.
   Chunk size is usually tuned by feel; this turns it into a measurement. Note
   that a bigger size keeps sections together but dilutes each vector, so there
   is a real optimum rather than a bigger-is-better rule.
7. `embed.py doc.pdf --query "..."` with a question phrased in your own words
   rather than the document's. Watch how many terms get dropped. That gap is
   the entire reason dense embeddings, hybrid search and reranking exist.
8. Run the same query under `--backend tfidf` and `--backend svd`. The ranking
   changes; the maths does not.
