# FastAPI RAG + Eval Harness — Project Brief

## Goal
Build a RAG system over the FastAPI documentation, with an evaluation harness built *first* and used throughout to measure retrieval and answer quality as the system evolves (chunking strategy, hybrid search, reranking, etc.).

The point of this project is not "a chatbot that answers FastAPI questions." The point is being able to say, with numbers: *"I improved retrieval hit-rate from X% to Y% by doing Z."* Every design decision should be measurable against the eval set before/after.

---

## Phase 0 — Corpus setup

Clone the FastAPI repo and use the markdown docs as the corpus:

```bash
git clone https://github.com/fastapi/fastapi.git
```

Corpus root: `fastapi/docs/en/docs/`

- Keep the existing folder structure (tutorial/, advanced/, deployment/, etc.) — it's useful metadata for later analysis (e.g. "do we do worse on advanced/ docs than tutorial/ docs?").
- Skip non-English doc translations.
- Do a basic pass to strip any raw HTML/admonition syntax that would confuse a naive chunker if needed, but don't over-clean — real docs have some noise and that's fine for eval realism.

---

## Phase 1 — Eval set (build this before the RAG system)

### 1a. Source real questions

Pull from:
- **GitHub Discussions** on `fastapi/fastapi` (Questions category) — via GitHub API (`api.github.com`), since many have accepted answers that can double as ground truth.
- **Stack Overflow**, tag `fastapi`, sorted by votes, favoring ones with accepted answers.

Target: pull ~60 candidates, hand-curate down to ~40 final questions.

### 1b. Categorize and distribute

- **Direct lookup** (~15): answer lives in a single doc page, near-literal quote works as ground truth.
  - e.g. "How do I set a default value for a query parameter?"
- **Multi-hop** (~15): answer spans 2+ doc pages.
  - e.g. "How do I use OAuth2 with scopes AND return a custom error when a scope is missing?"
- **Conceptual/how-to** (~10): answer requires synthesis, not just a quote.
  - e.g. "What's the difference between using `Depends` and middleware for auth?"

### 1c. Ground truth schema

One JSON record per question:

```json
{
  "id": "q001",
  "question": "How do I set a default value for a query parameter?",
  "category": "direct_lookup",
  "source_docs": ["docs/en/docs/tutorial/query-params.md"],
  "answer_snippet": "Assign a default value in the function signature, e.g. `q: str = None`",
  "source_url": "https://github.com/fastapi/fastapi/discussions/XXXX"
}
```

Store all records in a single `eval_set.json` (array of these objects).

---

## Phase 2 — Scorers (build before the RAG system's "final" form, use throughout)

Two **independent** scorers — don't conflate them, since a good RAG system can fail at either stage separately.

### 2a. Retrieval scorer (objective, no LLM call — cheap, fast to iterate)

```python
def score_retrieval(question_id, retrieved_chunks, ground_truth_docs, k=5):
    retrieved_docs = {chunk.source_file for chunk in retrieved_chunks[:k]}
    hit = bool(retrieved_docs & set(ground_truth_docs))
    return {
        "question_id": question_id,
        "hit_at_k": hit,
        "retrieved_docs": list(retrieved_docs),
        "ground_truth_docs": ground_truth_docs,
    }
```

Track hit@k for k = 3, 5, 10 to see how retrieval depth affects results.

### 2b. Answer scorer (LLM-as-judge, structured output)

```python
def score_answer(question, generated_answer, ground_truth_snippet):
    judge_prompt = f"""
    Question: {question}
    Reference answer: {ground_truth_snippet}
    Generated answer: {generated_answer}

    Rate the generated answer as exactly one of:
    - correct
    - partially_correct
    - wrong
    - hallucinated  (states something not supported by the reference or docs)

    Return ONLY JSON: {{"verdict": "...", "reasoning": "..."}}
    """
    # call model with judge_prompt, parse JSON response
    ...
```

Log the judge's reasoning alongside the verdict — needed for spot-checking judge quality later, not just trusting it blindly.

### 2c. Results storage

Store every eval run as a row in a results log (CSV or JSONL), including:
- run_id / timestamp
- config used (chunk size, retrieval method, model, k)
- per-question retrieval + answer scores
- aggregate hit@k and verdict distribution

This is what lets you produce before/after comparisons later (e.g. "reranking bumped hit@5 from 61% → 84%").

---

## Phase 3 — RAG system (build iteratively against the eval set)

Suggested iteration order — rerun both scorers after each change and record deltas:

1. **Baseline**: naive fixed-size chunking (e.g. 500 tokens, no overlap) + single embedding model + top-k vector search.
2. **Chunking v2**: try structure-aware chunking (split on markdown headers instead of fixed size).
3. **Hybrid search**: add keyword/BM25 alongside vector search.
4. **Reranking**: add a reranker step on top-k candidates before generation.
5. **Generation prompt tuning**: only after retrieval is solid — tune how retrieved chunks are formatted into the generation prompt.

For each step, keep a short written note: what changed, what the eval numbers were before/after, and a hypothesis for *why* it helped or didn't. This write-up is as valuable as the code for portfolio purposes.

---

## Suggested repo structure

```
fastapi-rag-eval/
├── corpus/                  # cloned fastapi docs (or symlink/submodule)
├── eval/
│   ├── eval_set.json
│   ├── score_retrieval.py
│   ├── score_answer.py
│   └── results/             # per-run logs
├── rag/
│   ├── chunkers.py
│   ├── indexer.py
│   ├── retriever.py
│   └── generator.py
├── scripts/
│   ├── pull_github_discussions.py
│   ├── pull_stackoverflow.py
│   └── run_eval.py
└── notes.md                 # per-iteration findings
```

---

## What "done" looks like

- A working RAG pipeline over FastAPI docs.
- A 40-question eval set with real, sourced ground truth.
- At least 3-4 iterations logged with before/after retrieval + answer scores.
- A short write-up (notes.md) narrating what was tried, what worked, what didn't, and why — this is the part that reads as senior-level judgment in interviews or a portfolio, not the code itself.
