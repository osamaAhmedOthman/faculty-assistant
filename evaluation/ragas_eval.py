"""
RAGAS evaluation runner for benchmarking end-to-end pipeline performance.

Architecture & Design Notes:
- End-to-End Evaluation: Executes `Pipeline.run()` directly on `GOLDEN_SET` examples, testing the actual 
  retrieval, generation, and guardrail layers used by production entry points (`api/`, `dashboard/`).
- Zero OpenAI Footprint: Replaces RAGAS default OpenAI dependencies with the existing local 
  `sentence-transformers` model (for embeddings) and `GroqClient` (as the LLM judge).
- Core Metrics Suite: Measures Faithfulness, Answer Relevancy, Context Precision, and Context Recall to 
  comprehensively evaluate groundedness, alignment, and retrieval quality.
- Categorized & Persistent Reporting: Groups scores by `category` to uncover hidden domain-specific failure 
  modes and saves timestamped JSON outputs to `evaluation/results/` for historical comparison.
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL_NAME, require_keys
from rag.pipeline import Pipeline
from evaluation.dataset import GOLDEN_SET, GoldenExample, all_categories

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

RAGAS_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Groq enforces token-per-day (TPD) and token-per-minute (TPM) quotas
# per model, not account-wide. RAGAS judge calls can still be heavy,
# so keep the default aligned with the model the project is actively
# using: openai/gpt-oss-120b. The env var remains available for
# comparisons or temporary overrides, but the default should match the
# generation setup we are actually running.
import os
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-20b")


# ---------------------------------------------------------------------------
# Step 1: run the real pipeline over the golden set
# ---------------------------------------------------------------------------

def collect_results(pipeline: Pipeline, golden_set: list[GoldenExample]) -> list[dict]:
    """
    Run every golden example through Pipeline.run() and assemble the
    dict shape RAGAS expects (question/answer/contexts/ground_truth),
    plus the extra fields (category, citation_warnings) this file's
    own reporting needs but RAGAS doesn't consume.

    Kept as a plain function (not folded into run_evaluation) so it can
    be unit-tested against a fake/stub pipeline without a real
    Groq/Pinecone call — same pattern retriever.py and generator.py
    already use for testability.
    """
    rows = []
    for example in golden_set:
        result = pipeline.run(example.question)
        rows.append(
            {
                "id": example.id,
                "category": example.category,
                "question": example.question,
                "answer": result["answer"],
                "contexts": [c["text"] for c in result["retrieved_chunks"]],
                "ground_truth": example.ground_truth,
                "sources": result["sources"],
                "expected_sources": example.expected_sources,
                "confidence": result["confidence"],
                "citations_valid": result.get("citations_valid"),
                "citation_warnings": result.get("citation_warnings", []),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Step 2: score with RAGAS
# ---------------------------------------------------------------------------

def _build_ragas_dataset(rows: list[dict]):
    """
    Convert collected rows into the HuggingFace Dataset shape RAGAS's
    evaluate() requires. Imported lazily (not at module top) so that
    collect_results() and the reporting functions below can be
    unit-tested without the `datasets`/`ragas` packages installed.
    """
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in rows
        ]
    )


def _get_ragas_judge():
    """
    Wire RAGAS's judge LLM to a Groq model SEPARATE from GROQ_MODEL
    (see RAGAS_JUDGE_MODEL's module-level comment for why), and its
    judge embeddings to this project's existing local
    sentence-transformers model. Imported lazily for the same reason
    as _build_ragas_dataset above.
    """
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    require_keys("GROQ_API_KEY")

    judge_llm = LangchainLLMWrapper(ChatGroq(api_key=GROQ_API_KEY, model=RAGAS_JUDGE_MODEL, temperature=0.0))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    return judge_llm, judge_embeddings


def score_with_ragas(rows: list[dict]):
    """
    Run RAGAS's evaluate() over the collected rows. Returns a pandas
    DataFrame (RAGAS's own .to_pandas() output) with one row per
    example and one column per metric — kept as RAGAS's native return
    shape rather than reshaped here, so anything RAGAS adds to that
    output later is available without this file needing to change.

    THROTTLING: RunConfig defaults to max_workers=16 and max_retries=10.
    Against Groq's free-tier rate limit, 16 concurrent metric-jobs (each
    of which — faithfulness especially — issues several LLM calls
    internally to decompose and check individual claims) blow past the
    limit almost immediately, so most jobs then retry up to 10 times
    each with backoff. That's how ~48 logical jobs (12 examples x 4
    metrics) turned into 600+ actual HTTP requests and multi-minute
    TimeoutErrors in practice. max_workers=2 keeps concurrent request
    volume low enough to mostly stay under the rate limit in the first
    place, rather than relying on retries to recover from repeatedly
    blowing past it — recovering via retries is strictly more requests
    than not tripping the limit to begin with. This is a real, load-
    bearing constraint of running RAGAS against a free-tier LLM API,
    not a workaround to remove once "working" — leave it here.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.run_config import RunConfig

    dataset = _build_ragas_dataset(rows)
    judge_llm, judge_embeddings = _get_ragas_judge()

    run_config = RunConfig(
        timeout=180,       # generous ceiling — a job waiting out a TPM cooldown genuinely needs more time, not less
        max_retries=5,     # more attempts than before: a TPM limit (resets every 60s) needs enough retries for backoff to actually reach past that window, unlike a TPD limit which retries can't fix at all
        max_wait=90,       # must comfortably exceed Groq's own reported cooldown ("try again in 22.29s" observed in practice) or retries give up before the window that would have let them succeed
        max_workers=1,     # fully serialized — at max_workers=2, two ~2500-token jobs landing in the same second can blow a 6000 TPM ceiling together even though neither alone would; one at a time paces total token usage against the per-minute window instead of bursting past it
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        raise_exceptions=False,  # a single failed job (e.g. still-timed-out after retries) shouldn't crash the whole run — its metric comes back NaN instead, visible in the saved results rather than losing everything collected so far
    )
    df = result.to_pandas()
    # RAGAS's dataframe doesn't know about categories — attach them
    # positionally, since _build_ragas_dataset preserves row order.
    df["category"] = [r["category"] for r in rows]
    df["id"] = [r["id"] for r in rows]
    return df


# ---------------------------------------------------------------------------
# Step 3: report
# ---------------------------------------------------------------------------

def summarize_by_category(df) -> dict:
    """
    Average each RAGAS metric within each category. Returns a plain
    dict (not a DataFrame) so this is trivially JSON-serializable for
    the saved results file, and so it can be unit-tested without a
    real RAGAS run (see tests/test_run_eval.py) by handing it a
    hand-built DataFrame-like structure.

    NaN HANDLING: a metric cell can be NaN for two different reasons —
    (1) context_recall is only meaningful when ground_truth describes
    real expected content; the out-of-scope/adversarial examples in
    dataset.py have a ground_truth describing expected BEHAVIOR
    ("should decline"), not retrievable content, so RAGAS scoring that
    as NaN there is correct, not a failure; (2) a job that still timed
    out after RunConfig's retries (see score_with_ragas) comes back NaN
    too, which IS a failure worth knowing about. pandas' .mean()
    silently skips NaNs either way, which would hide the difference
    between "3 of 3 examples scored, average 0.9" and "1 of 3 scored,
    the other 2 failed, average 0.9" — those are very different results
    to report as the same number. scored_n makes that visible.
    """
    summary = {}
    for category in sorted(df["category"].unique()):
        subset = df[df["category"] == category]
        metric_cols = [c for c in RAGAS_METRIC_NAMES if c in subset.columns]
        category_summary = {"count": len(subset)}
        for m in metric_cols:
            scored = subset[m].dropna()
            category_summary[m] = round(float(scored.mean()), 3) if len(scored) > 0 else None
            category_summary[f"{m}_scored_n"] = len(scored)
        summary[category] = category_summary
    return summary


def _format_metric_cell(stats: dict, metric: str) -> str:
    value = stats.get(metric)
    scored_n = stats.get(f"{metric}_scored_n", 0)
    total_n = stats.get("count", 0)
    if value is None:
        return f"{'n/a':>18}"
    if scored_n < total_n:
        return f"{value:>13.3f} ({scored_n}/{total_n})"
    return f"{value:>18.3f}"


def print_summary(summary: dict) -> None:
    print(f"\n{'Category':<35}{'N':>4}  " + "  ".join(f"{m:>18}" for m in RAGAS_METRIC_NAMES))
    print("-" * 130)
    for category, stats in summary.items():
        metric_vals = "  ".join(_format_metric_cell(stats, m) for m in RAGAS_METRIC_NAMES)
        print(f"{category:<35}{stats['count']:>4}  {metric_vals}")


def save_results(rows: list[dict], df, summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"

    per_example = df.to_dict(orient="records")
    payload = {
        "timestamp": timestamp,
        "golden_set_size": len(rows),
        "summary_by_category": summary,
        "per_example": per_example,
        # Raw pipeline output (sources/guardrail flags) kept alongside
        # the RAGAS scores so a low faithfulness score and a guardrail
        # citation_warning on the SAME example can be cross-checked —
        # two independent signals catching the same failure is a
        # stronger finding than either alone.
        "pipeline_output": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(golden_set: list[GoldenExample] | None = None) -> dict:
    golden_set = golden_set if golden_set is not None else GOLDEN_SET

    print(f"Running {len(golden_set)} golden examples through the real pipeline...")
    pipeline = Pipeline()
    rows = collect_results(pipeline, golden_set)

    flagged = [r for r in rows if r["citation_warnings"]]
    if flagged:
        print(f"\n{len(flagged)} example(s) tripped the citation guardrail during collection:")
        for r in flagged:
            print(f"  - {r['id']}: {r['citation_warnings']}")

    print("\nScoring with RAGAS (this calls the judge LLM once per metric per example)...")
    df = score_with_ragas(rows)

    summary = summarize_by_category(df)
    print_summary(summary)

    out_path = save_results(rows, df, summary)
    print(f"\nFull results saved to {out_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the golden set through the pipeline and score with RAGAS."
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Only run examples from these categories (e.g. --categories answerable_course out_of_scope_topic). "
             "Default: all categories.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N examples (after category filtering, if any). "
             "Use this to fit a run inside a limited daily token budget.",
    )
    parser.add_argument(
        "--list-categories", action="store_true",
        help="Print available categories and each one's example count, then exit — no pipeline or Groq calls.",
    )
    args = parser.parse_args()

    if args.list_categories:
        for cat in all_categories():
            n = sum(1 for e in GOLDEN_SET if e.category == cat)
            print(f"  {cat}: {n} example(s)")
        sys.exit(0)

    subset = GOLDEN_SET
    if args.categories:
        subset = [e for e in subset if e.category in args.categories]
    if args.limit is not None:
        subset = subset[: args.limit]

    if not subset:
        print("No examples match the given --categories filter. Use --list-categories to see valid names.")
        sys.exit(1)

    run_evaluation(golden_set=subset)
