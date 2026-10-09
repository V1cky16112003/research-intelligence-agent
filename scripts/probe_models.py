"""
Daily liveness probe for every pinned LLM model.

Providers retire free-tier models without notice: NIM's llama-3.1-70b went 410 Gone,
and two dead pins each scored RAGAS 0.000 — which looked like a quality regression
rather than an outage. `/models` is no help (NIM lists models that 404/410 when
called), so this makes one tiny real completion per pin and exits non-zero if any
fails. Run on a schedule by .github/workflows/scheduled.yml.

    GROQ_API_KEY=... NVIDIA_NIM_API_KEY=... GEMINI_API_KEY=... python -m scripts.probe_models
"""
from __future__ import annotations

import asyncio
import os
import sys

from openai import AsyncOpenAI

from agent.gateway import LLMGateway

PROBES = [
    ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1",
     os.getenv("GROQ_MODEL") or LLMGateway.GROQ_MODEL),
    ("nvidia_nim", "NVIDIA_NIM_API_KEY", "https://integrate.api.nvidia.com/v1",
     os.getenv("NIM_MODEL") or LLMGateway.NIM_MODEL),
    ("gemini", "GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/",
     LLMGateway.GEMINI_MODEL),
]


async def probe(name: str, key_env: str, base_url: str, model: str) -> tuple[str, bool, str]:
    key = os.getenv(key_env)
    if not key:
        return name, True, f"skipped ({key_env} unset)"
    client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=1)
    try:
        # A 200 is the signal; content may be empty if a reasoning model spends the
        # 16 tokens thinking, which is fine — the model is servable.
        await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Reply with OK."}], max_tokens=16,
        )
        return name, True, f"{model}: ok"
    except Exception as e:
        status = getattr(e, "status_code", None)
        # 429 means the model exists and our quota is spent — not a retirement.
        if status == 429:
            return name, True, f"{model}: rate limited (alive)"
        return name, False, f"{model}: {type(e).__name__} {status or ''} {str(e)[:200]}"


async def main() -> int:
    results = await asyncio.gather(*(probe(*p) for p in PROBES))
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'} {name:<11} {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
