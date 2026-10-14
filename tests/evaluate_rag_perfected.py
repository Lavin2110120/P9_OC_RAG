#!/usr/bin/env python3
"""Ultra-safe RAG evaluation with serialized Mistral requests.

The script deliberately keeps data collection separate from scoring so the
retrieval/generation trace can be inspected and re-scored without re-calling
the application. It adds a process-wide, serialized limiter around both
ChatMistralAI generation paths because Ragas can otherwise issue requests
concurrently and trigger provider rate limits.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv

LOGGER = logging.getLogger("evaluate_rag")
T = TypeVar("T")

# Mistral rate-limit protection: serialize all evaluator requests and leave a
# hard 2.5-second minimum between request starts. Ragas can otherwise fire
# concurrent requests too quickly and trigger HTTP 429 / code 1300 responses.
_MISTRAL_REQUEST_LOCK = threading.Lock()
_MISTRAL_LAST_REQUEST = 0.0
_MISTRAL_MIN_INTERVAL = 2.5


RETRYABLE_TEXT = (
    "429", "rate limit", "rate_limit", "too many requests", "temporarily unavailable",
    "timeout", "timed out", "connection reset", "service unavailable", "overloaded",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _wait_between_mistral_requests() -> None:
    """Enforce a process-wide pause before each Mistral generation."""
    global _MISTRAL_LAST_REQUEST
    with _MISTRAL_REQUEST_LOCK:
        elapsed = time.monotonic() - _MISTRAL_LAST_REQUEST
        if elapsed < _MISTRAL_MIN_INTERVAL:
            delay = _MISTRAL_MIN_INTERVAL - elapsed
            LOGGER.debug("Throttling Mistral generation for %.2fs", delay)
            time.sleep(delay)
        _MISTRAL_LAST_REQUEST = time.monotonic()


def retry_call(fn: Callable[[], T], *, label: str, attempts: int, base_delay: float) -> T:
    """Retry transient provider failures with capped exponential backoff + jitter."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # provider SDKs use several exception classes
            message = str(exc).lower()
            retryable = any(token in message for token in RETRYABLE_TEXT)
            if not retryable or attempt == attempts:
                raise
            delay = min(60.0, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            LOGGER.warning("%s failed (attempt %d/%d): %s; retrying in %.1fs", label, attempt, attempts, exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def load_benchmarks(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Benchmark must be a non-empty JSON array")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark row {index} is not an object")
        question = item.get("question", item.get("user_input"))
        reference = item.get("ground_truth", item.get("reference"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Benchmark row {index} has no non-empty question")
        if reference is None or (isinstance(reference, str) and not reference.strip()):
            raise ValueError(f"Benchmark row {index} has no reference/ground_truth")
        # Ragas' SingleTurnSample reference is a string. Preserve multiple
        # accepted answers in a deterministic, readable form.
        if isinstance(reference, list):
            reference = "\n".join(str(value) for value in reference)
        elif not isinstance(reference, str):
            reference = str(reference)
        normalized.append({"question": question.strip(), "ground_truth": reference})
    return normalized


def text_from_response(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    return str(response).strip()


def build_run_config(RunConfig: Any, args: argparse.Namespace) -> Any:
    # Ragas has changed RunConfig fields between releases; only pass fields
    # supported by the installed version.
    candidates = {
        # Keep Ragas evaluation strictly serialized to avoid provider-side
        # concurrency and rate-limit bursts. This is intentionally fixed at 1.
        "max_workers": 1,
        "timeout": args.ragas_timeout,
        "max_retries": args.provider_attempts - 1,
        "max_wait": args.retry_base_delay * 8,
    }
    supported = inspect.signature(RunConfig).parameters
    return RunConfig(**{key: value for key, value in candidates.items() if key in supported})


def mean_scores(frame: Any) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for column in frame.columns:
        if column in {"user_input", "response", "retrieved_contexts", "reference"}:
            continue
        values = []
        for value in frame[column].tolist():
            try:
                number = float(value)
                if math.isfinite(number):
                    values.append(number)
            except (TypeError, ValueError):
                pass
        summary[column] = (sum(values) / len(values)) if values else None
    return summary


def evaluate(args: argparse.Namespace) -> int:
    load_dotenv(args.env_file)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is required")

    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    benchmark_path = args.benchmark.resolve() if args.benchmark else project_root / "annotated_qa.json"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = load_benchmarks(benchmark_path)

    # Lazy imports make --help and source-level checks usable without the full
    # ML stack, while producing a clear import error for real execution.
    from datasets import Dataset
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_mistralai import ChatMistralAI

    class RateLimitedChatMistralAI(ChatMistralAI):
        """ChatMistralAI with serialized, paced sync and async generations."""

        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            _wait_between_mistral_requests()
            return super()._generate(*args, **kwargs)

        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            # Ragas is configured with max_workers=1; this also protects an
            # async path from producing a burst of requests.
            _wait_between_mistral_requests()
            return await super()._agenerate(*args, **kwargs)

    from ragas import RunConfig, evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    try:
        from ragas.metrics import answer_correctness
    except ImportError:
        answer_correctness = None
        LOGGER.warning("answer_correctness is unavailable in this Ragas version")
    from scripts.faiss_retriever import FAISSRetriever
    from scripts.mistral_qa import generate_response

    retriever = FAISSRetriever()
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    references: list[str] = []
    errors: list[dict[str, Any]] = []

    LOGGER.info("Collecting %d benchmark examples", len(benchmarks))
    for index, item in enumerate(benchmarks):
        question = item["question"]
        try:
            docs = retry_call(
                lambda: retriever.retrieve(question, k=args.top_k),
                label=f"retrieval row {index}", attempts=args.provider_attempts,
                base_delay=args.retry_base_delay,
            )
            context_chunks = [str(getattr(doc, "page_content", doc)).strip() for doc in docs]
            context_chunks = [chunk for chunk in context_chunks if chunk]
            if not context_chunks:
                raise ValueError("retriever returned no non-empty contexts")
            answer = retry_call(
                lambda: text_from_response(generate_response(question)),
                label=f"generation row {index}", attempts=args.provider_attempts,
                base_delay=args.retry_base_delay,
            )
            if not answer:
                raise ValueError("generator returned an empty answer")
            questions.append(question)
            answers.append(answer)
            contexts.append(context_chunks)
            references.append(item["ground_truth"])
        except Exception as exc:
            error = {"row": index, "question": question, "error": repr(exc)}
            errors.append(error)
            LOGGER.error("%s", error)
            if not args.skip_errors:
                raise
        if index + 1 < len(benchmarks):
            time.sleep(args.request_interval)

    if not questions:
        raise RuntimeError("No usable benchmark rows remain after collection")
    if errors:
        LOGGER.warning("Skipped %d/%d rows", len(errors), len(benchmarks))

    # These are the canonical Ragas column names for current releases. The
    # explicit lists are important: contexts must be list[str], not one string.
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": references,
    })

    evaluator_kwargs = {
        "api_key": api_key,
        "model": args.model,
        "temperature": 0,
        "max_retries": args.provider_attempts - 1,
        "timeout": args.provider_timeout,
    }
    supported = inspect.signature(ChatMistralAI).parameters
    evaluator_llm = RateLimitedChatMistralAI(**{key: value for key, value in evaluator_kwargs.items() if key in supported})
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    if answer_correctness is not None and not args.no_answer_correctness:
        metrics.append(answer_correctness)
    LOGGER.info("Scoring %d rows with metrics: %s", len(questions), ", ".join(metric.name for metric in metrics))
    run_config = build_run_config(RunConfig, args)
    results = retry_call(
        lambda: ragas_evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm,
                               embeddings=embeddings, run_config=run_config),
        label="Ragas evaluation", attempts=args.evaluation_attempts,
        base_delay=args.retry_base_delay,
    )

    frame = results.to_pandas()
    result_csv = output_dir / args.results_csv
    frame.to_csv(result_csv, index=False)
    summary = {
        "benchmark": str(benchmark_path), "rows_total": len(benchmarks),
        "rows_scored": len(questions), "rows_skipped": len(errors),
        "metrics": [metric.name for metric in metrics], "scores": mean_scores(frame),
        "errors": errors,
    }
    summary_path = output_dir / args.summary_json
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s and %s", result_csv, summary_path)

    failed = []
    for metric_name, threshold in {
        "faithfulness": args.min_faithfulness,
        "answer_relevancy": args.min_answer_relevancy,
        "context_precision": args.min_context_precision,
        "context_recall": args.min_context_recall,
    }.items():
        score = summary["scores"].get(metric_name)
        if score is None or score < threshold:
            failed.append(f"{metric_name}={score!r} < {threshold}")
    if failed:
        LOGGER.error("Quality gate failed: %s", "; ".join(failed))
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=os.getenv("MISTRAL_EVAL_MODEL", "mistral-small-latest"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--top-k", type=positive_int, default=3)
    parser.add_argument("--workers", type=positive_int, default=1, help="Ragas workers; the global limiter also protects concurrent workers")
    parser.add_argument("--provider-attempts", type=positive_int, default=5)
    parser.add_argument("--evaluation-attempts", type=positive_int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--request-interval", type=float, default=2.5)
    parser.add_argument("--provider-timeout", type=float, default=600.0)
    parser.add_argument("--ragas-timeout", type=float, default=900.0)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--no-answer-correctness", action="store_true")
    parser.add_argument("--results-csv", default="ragas_evaluation_results.csv")
    parser.add_argument("--summary-json", default="ragas_evaluation_summary.json")
    parser.add_argument("--min-faithfulness", type=float, default=0.50)
    parser.add_argument("--min-answer-relevancy", type=float, default=0.50)
    parser.add_argument("--min-context-precision", type=float, default=0.60)
    parser.add_argument("--min-context-recall", type=float, default=0.60)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(evaluate(args))
