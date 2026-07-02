from __future__ import annotations
"""
RAGAS evaluation runner.

Runs the golden Q/A set through the RAG pipeline, scores with RAGAS,
logs to MLflow, and exits non-zero if thresholds are breached.

Usage:
    python -m eval.run_ragas [--golden eval/golden_set.json] [--fail-under-faithfulness 0.8]

CI usage (in GitHub Actions):
    python -m eval.run_ragas --ci
"""
import argparse
import asyncio
import json
import logging
import math
import os
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path

# ragas 0.4.3 eagerly imports ChatVertexAI from langchain_community.chat_models.vertexai,
# which was removed in langchain-community >= 0.3.0. Stub the module with a dummy class
# so the import succeeds; ragas never instantiates it because we pass our own judge_llm.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})  # dummy — never instantiated
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

logger = logging.getLogger(__name__)

# Quality thresholds — CI fails if any metric drops below these
THRESHOLDS = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.75,
    "context_precision": 0.7,
}


def _mean(val) -> float:
    """Average a RAGAS per-sample metric result, excluding NaN/None entries.

    ragas 0.4.3 returns a per-sample list/Series (not a scalar), and can
    return NaN for individual samples RAGAS couldn't score (e.g. empty
    context/answer). `v is not None` alone does not catch NaN — a NaN
    silently poisons the mean if included, which previously let a broken
    retrieval pipeline report a NaN "average" instead of a hard failure.
    """
    if hasattr(val, "mean"):
        val = list(val)
    vals = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float("nan")


NIM_JUDGE_MAX_RPM = 10  # stay well under NIM's 40 RPM free-tier cap during RAGAS evaluation


class _SlidingWindowRateLimiter:
    """Thread-safe rate limiter: blocks until fewer than max_calls have been made
    in the trailing period_seconds window. RAGAS dispatches metric evaluations from
    a thread pool (not asyncio), so this uses threading.Lock, not an asyncio lock."""

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self._max_calls = max_calls
        self._period = period_seconds
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                wait = self._period - (now - self._timestamps[0])
            if wait > 0:
                time.sleep(wait)


def _apply_rate_limit(client, max_calls_per_minute: int = NIM_JUDGE_MAX_RPM):
    """Monkey-patch client.chat.completions.create / client.embeddings.create so
    every call goes through a shared rate limiter.

    Patches the bound methods on the existing client object rather than wrapping
    it in a proxy — ragas's `instructor` integration does isinstance checks on the
    client, so substituting a duck-typed wrapper for the client itself would break
    that detection. Mutating the object's methods in place preserves its type.
    """
    limiter = _SlidingWindowRateLimiter(max_calls_per_minute, period_seconds=60.0)

    orig_chat_create = client.chat.completions.create

    def _rate_limited_chat_create(*args, **kwargs):
        limiter.acquire()
        return orig_chat_create(*args, **kwargs)

    client.chat.completions.create = _rate_limited_chat_create

    orig_embeddings_create = client.embeddings.create

    def _rate_limited_embeddings_create(*args, **kwargs):
        limiter.acquire()
        return orig_embeddings_create(*args, **kwargs)

    client.embeddings.create = _rate_limited_embeddings_create

    return client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAGAS evaluation over golden set")
    p.add_argument("--golden", default="eval/golden_set.json", help="Path to golden set JSON")
    p.add_argument("--ci", action="store_true", help="CI mode: exit 1 if thresholds not met")
    p.add_argument("--limit", type=int, default=20, help="Max questions to evaluate")
    p.add_argument("--judge-model", default="meta/llama-3.1-70b-instruct", help="LLM judge model for RAGAS (NVIDIA NIM by default — Gemini's 5rpm/20-per-day free tier is too small for judging multiple metrics per sample)")
    p.add_argument("--mlflow-uri", default="", help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI env)")
    p.add_argument("--experiment-name", default="rag-quality-gate", help="MLflow experiment name")
    return p.parse_args()


async def get_rag_answer(question: str, gateway) -> dict:
    """Run the RAG retrieval tool and get an LLM answer for a question."""
    from agent.tools import rag_retrieval_tool

    # Get retrieved chunks
    tool_result_json = await rag_retrieval_tool(question)
    tool_result = json.loads(tool_result_json)
    chunks = tool_result.get("results", [])

    # Build context from chunks
    context_texts = [c.get("content", "") for c in chunks[:5] if c.get("content")]

    if not context_texts:
        return {"answer": "No relevant context found.", "contexts": []}

    # Generate answer using gateway
    context_str = "\n\n".join(context_texts)
    messages = [
        {"role": "system", "content": "Answer the question based on the provided context. Be concise and accurate."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
    ]
    resp = await gateway.chat(messages, temperature=0.1, max_tokens=512, cache=False)
    return {
        "answer": resp.get("content", ""),
        "contexts": context_texts,
    }


async def run_evaluation(args: argparse.Namespace) -> dict:
    """Run the full evaluation pipeline. Returns metrics dict."""
    from db.connection import init_pool
    from agent.gateway import LLMGateway
    from agent.redis_client import create_redis_client

    # Load golden set
    golden_path = Path(args.golden)
    if not golden_path.exists():
        logger.error("Golden set not found: %s", golden_path)
        sys.exit(1)

    with open(golden_path) as f:
        golden_set = json.load(f)[: args.limit]

    logger.info("Loaded %d questions from golden set", len(golden_set))

    # Init DB + gateway
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        await init_pool(db_url)

    redis_client = None
    redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "")
    redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
    if redis_url and redis_token and redis_url.startswith("https://"):
        from urllib.parse import urlparse
        parsed = urlparse(redis_url)
        combined = f"https://default:{redis_token}@{parsed.netloc}"
        redis_client = await create_redis_client(combined)
    else:
        logger.info("Upstash not configured — running without cache")

    gateway = LLMGateway(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        nvidia_api_key=os.getenv("NVIDIA_NIM_API_KEY", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        redis_client=redis_client,
    )
    # rag_retrieval_tool fetches its gateway from the module-level registry
    # (not passed as an arg) — must register it before any retrieval runs,
    # or every call fails with "Gateway not initialized."
    from agent.registry import set_gateway
    set_gateway(gateway)

    # Collect answers + contexts
    logger.info("Running RAG pipeline on %d questions...", len(golden_set))
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden_set:
        q = item["question"]
        logger.info("  Q: %s", q[:60])
        try:
            result = await get_rag_answer(q, gateway)
            questions.append(q)
            answers.append(result["answer"])
            contexts.append(result["contexts"] if result["contexts"] else ["no context retrieved"])
            ground_truths.append(item["ground_truth"])
        except Exception as e:
            logger.warning("Failed on question %s: %s", item["id"], e)
            questions.append(q)
            answers.append("Error: could not generate answer")
            contexts.append(["error"])
            ground_truths.append(item["ground_truth"])

    # Build RAGAS dataset
    try:
        from ragas import EvaluationDataset, SingleTurnSample
        from ragas import evaluate as ragas_evaluate
        # ragas.metrics gives pre-instantiated singletons (no () needed);
        # ragas.metrics.collections gives submodules which are not callable
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
        from ragas.llms import llm_factory
        from ragas.embeddings import embedding_factory
        from openai import OpenAI

        samples = [
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=c,
                reference=g,
            )
            for q, a, c, g in zip(questions, answers, contexts, ground_truths)
        ]
        dataset = EvaluationDataset(samples=samples)

        nim_key = os.getenv("NVIDIA_NIM_API_KEY", "")
        nim_base = "https://integrate.api.nvidia.com/v1"
        nim_client = _apply_rate_limit(OpenAI(api_key=nim_key, base_url=nim_base))

        judge_llm = llm_factory(args.judge_model, client=nim_client)
        judge_embeddings = embedding_factory("openai", model="nvidia/nv-embedqa-e5-v5", client=nim_client)

        result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )

        metrics = {
            "faithfulness": _mean(result["faithfulness"]),
            "answer_relevancy": _mean(result["answer_relevancy"]),
            "context_precision": _mean(result["context_precision"]),
            "num_questions": len(questions),
        }

    except ImportError as e:
        logger.error("RAGAS import failed: %s", e)
        # Fallback: return placeholder metrics so CI doesn't crash on import issues
        metrics = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "num_questions": len(questions),
            "error": str(e),
        }

    return metrics


def log_to_mlflow(metrics: dict, args: argparse.Namespace) -> None:
    """Log evaluation metrics to MLflow on DagsHub."""
    try:
        import mlflow

        tracking_uri = (
            args.mlflow_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or f"https://dagshub.com/{os.getenv('DAGSHUB_REPO', '')}.mlflow"
        )
        mlflow.set_tracking_uri(tracking_uri)

        # DagsHub auth
        dagshub_token = os.getenv("DAGSHUB_TOKEN", "")
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = "token"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name="ragas-ci-gate"):
            mlflow.log_params(
                {
                    "judge_model": args.judge_model,
                    "num_questions": metrics.get("num_questions", 0),
                    "faithfulness_threshold": THRESHOLDS["faithfulness"],
                    "answer_relevancy_threshold": THRESHOLDS["answer_relevancy"],
                    "context_precision_threshold": THRESHOLDS["context_precision"],
                }
            )
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            logger.info("Metrics logged to MLflow: %s", tracking_uri)
    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)


def check_thresholds(metrics: dict) -> list[str]:
    """Return list of failed threshold messages, empty if all pass."""
    failures = []
    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric, 0.0)
        # NaN comparisons are always False in Python (`nan < threshold` is
        # False), so a broken pipeline that produces NaN would otherwise
        # silently bypass the gate instead of failing it.
        if math.isnan(value) or value < threshold:
            failures.append(f"{metric}={value:.3f} < threshold={threshold}")
    return failures


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    metrics = await run_evaluation(args)

    print("\n=== RAGAS Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            threshold = THRESHOLDS.get(k)
            status = "pass" if threshold is None or v >= threshold else "FAIL"
            threshold_str = f" (threshold: {threshold})" if threshold else ""
            print(f"  [{status}] {k}: {v:.3f}{threshold_str}")
        else:
            print(f"  {k}: {v}")

    log_to_mlflow(metrics, args)

    if args.ci:
        failures = check_thresholds(metrics)
        if failures:
            print("\nRAGAS quality gate FAILED:")
            for f in failures:
                print(f"   {f}")
            sys.exit(1)
        else:
            print("\nRAGAS quality gate PASSED")
            sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
