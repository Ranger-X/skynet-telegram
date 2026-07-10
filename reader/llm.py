"""The reader's calls into the shared local llama-server, plus the app-level contention protocol
(brief §11): ingest generation yields to live chat. All reader LLM calls go through llm_lock so a
summary never overlaps an answer; the ingest worker additionally waits for INGEST_YIELD_SECONDS of
user-facing quiet (mark_user_activity is called from the reader's answer path AND — via a one-line
hook — from the persona's local replies in openrouter_client).
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

from . import rconfig

log = logging.getLogger("t800.reader")

llm_lock = asyncio.Lock()
_last_user_activity = 0.0


def mark_user_activity() -> None:
    global _last_user_activity
    _last_user_activity = time.monotonic()


def seconds_since_user_activity() -> float:
    return time.monotonic() - _last_user_activity


async def wait_for_quiet() -> None:
    """Ingest-side: don't start a generation while the user is actively talking to the bot."""
    while seconds_since_user_activity() < rconfig.INGEST_YIELD_SECONDS:
        await asyncio.sleep(5)


async def generate(messages: list[dict], max_tokens: int, timeout: float) -> str:
    """Non-streaming call (used by ingest summaries). Caller must hold llm_lock."""
    payload = {
        "model": "gemma4",
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(rconfig.LOCAL_CHAT_URL, json=payload)
        r.raise_for_status()
        data = r.json()
    tim = data.get("timings", {})
    log.info(
        "reader gen: prefilled=%s tok, out=%s tok, pp=%.1f t/s, tg=%.1f t/s",
        tim.get("prompt_n"), data.get("usage", {}).get("completion_tokens"),
        tim.get("prompt_per_second") or 0, tim.get("predicted_per_second") or 0,
    )
    return (data["choices"][0]["message"]["content"] or "").strip()


async def generate_stream(
    messages: list[dict], max_tokens: int, timeout: float, stats: dict | None = None
) -> AsyncIterator[str]:
    """SSE-streaming call (used by /ask so the user watches the answer grow instead of staring at
    'typing' for minutes — brief §5 latency note). Yields text deltas. Caller must hold llm_lock.
    Pass `stats` to learn how the generation ENDED: stats['finish_reason'] == 'length' means the
    cap cut the answer mid-thought — callers should tell the user instead of ending on a half-word."""
    payload = {
        "model": "gemma4",
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=timeout)) as client:
        async with client.stream("POST", rconfig.LOCAL_CHAT_URL, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    choice = json.loads(chunk)["choices"][0]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if stats is not None and choice.get("finish_reason"):
                    stats["finish_reason"] = choice["finish_reason"]
                delta = (choice.get("delta") or {}).get("content") or ""
                if delta:
                    yield delta


def estimate_call_seconds(input_tokens: int, output_tokens: int) -> float:
    """Honest per-call wall-clock: on this box PREFILL DOMINATES (14.6 t/s vs 3.8 t/s decode is only
    a ~4x gap, not the 'prefill is free' the original brief assumed), so it's a first-class term."""
    return input_tokens / rconfig.PREFILL_TOK_S + output_tokens / rconfig.DECODE_TOK_S
