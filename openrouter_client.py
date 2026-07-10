import asyncio
import logging

import httpx

import config

log = logging.getLogger("t800")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


class MalformedResponseError(Exception):
    """Raised when OpenRouter returns 200 OK but the body isn't shaped like a chat completion."""


def _guard_yes(verdict: str) -> bool:
    """The guard answers YES (en) or ДА (ru). Accept either so the guard is language-agnostic."""
    v = verdict.strip().upper()
    return v.startswith("YES") or v.startswith("ДА")


async def _post(messages: list[dict], model: str) -> str:
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        log.warning("Malformed OpenRouter response body from %s: %r", model, data)
        raise MalformedResponseError(f"Unexpected response shape from {model}") from exc


async def _request(messages: list[dict], model: str) -> str:
    try:
        return await _post(messages, model)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in RETRYABLE_STATUS_CODES:
            raise
        await asyncio.sleep(2)
        return await _post(messages, model)
    except MalformedResponseError:
        await asyncio.sleep(2)
        return await _post(messages, model)


async def _post_local(
    messages: list[dict], http_timeout: float | None = None, max_tokens: int | None = None
) -> str:
    """Local llama-server via its OpenAI-compatible /v1/chat/completions, thinking OFF (Gemma 4
    otherwise burns hundreds of hidden reasoning tokens per short reply). Runs fully on CPU. Unlike
    Ollama's /api/chat, llama-server reuses the KV cache prefix across requests, so the long stable
    system prompt + history is prefilled once and only new tokens are processed on later messages."""
    payload = {
        "model": config.LOCAL_MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens or config.LOCAL_MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    timeout = (http_timeout or config.LOCAL_TIMEOUT_SECONDS) + 15
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(config.LOCAL_CHAT_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    # Breakdown from llama-server: how much of the prompt was cached vs freshly prefilled, plus speeds.
    tim = data.get("timings", {})
    usage = data.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    log.info(
        "local: prompt=%s tok (cached=%s, prefilled=%s)  gen=%s tok  pp=%.0f t/s  tg=%.1f t/s",
        usage.get("prompt_tokens"), cached, tim.get("prompt_n"),
        usage.get("completion_tokens"),
        tim.get("prompt_per_second") or 0.0, tim.get("predicted_per_second") or 0.0,
    )

    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise MalformedResponseError(f"Unexpected local response shape: {data!r}") from exc
    if not content:
        raise MalformedResponseError("Local model returned empty content")
    return content


async def local_multimodal_reply(messages: list[dict], http_timeout: float | None = None) -> str:
    """Local image/audio reply through the SAME llama-server as text — the Gemma 4 mmproj projector is
    loaded into llama-server (--mmproj <bf16>), so images (`image_url` data-URIs) and audio
    (`input_audio`) ride the OpenAI-format /v1 endpoint with no separate vision service. The model is
    already warm from text, so there's no per-photo cold load (was ~50s on the old Ollama path). Bigger
    timeout than a plain text reply because the encoder pass + generation on CPU still takes a few sec."""
    return await _post_local(messages, http_timeout=http_timeout or config.LOCAL_VISION_TIMEOUT_SECONDS)


async def get_reply(messages: list[dict]) -> str:
    # Tell the long-reader's ingest worker the user is actively chatting — it must not fire a
    # summary generation into the shared llama-server right now (reader/llm.py contention protocol).
    from reader.llm import mark_user_activity

    mark_user_activity()
    # Local model first (if enabled); fall back to OpenRouter on any failure/unavailability/timeout.
    if config.USE_LOCAL_MODEL:
        try:
            return await asyncio.wait_for(_post_local(messages), timeout=config.LOCAL_TIMEOUT_SECONDS)
        except Exception:
            log.warning("Local model unavailable/failed, falling back to OpenRouter")

    try:
        return await _request(messages, config.OPENROUTER_MODEL)
    except (httpx.HTTPStatusError, MalformedResponseError) as exc:
        is_429 = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429
        if not (is_429 or isinstance(exc, MalformedResponseError)) or not config.OPENROUTER_FALLBACK_MODEL:
            raise
        return await _request(messages, config.OPENROUTER_FALLBACK_MODEL)


async def is_adversarial(guard_prompt: str, text: str) -> bool:
    """Fails open (returns False) on any guard-model error — a broken guard must never block chat.
    Local-first when USE_LOCAL_MODEL: the same Gemma 4 does the ДА/НЕТ classification (verified 6/6 on
    a mixed attack/benign set, only 8 output tokens). Stays fully local by default — if llama-server is
    down the guard fails open, and availability is handled at the source by the llama-server watchdog
    (llama_watchdog) rather than by reaching out to the cloud. The cloud fallback is wired but gated
    behind config.GUARD_CLOUD_FALLBACK (off by default; flip on for debugging or to cover the
    crash-recovery window). Note: the guard's prompt prefix differs from the persona's, so on
    llama-server they'd evict each other's KV cache under -np 1 — that's why we run -np 2 (a slot each)."""
    messages = [
        {"role": "system", "content": guard_prompt},
        {"role": "user", "content": text},
    ]
    loop = asyncio.get_event_loop()
    t0 = loop.time()

    if config.USE_LOCAL_MODEL:
        try:
            verdict = await asyncio.wait_for(
                _post_local(messages, http_timeout=config.GUARD_LOCAL_TIMEOUT_SECONDS, max_tokens=8),
                timeout=config.GUARD_LOCAL_TIMEOUT_SECONDS,
            )
        except Exception:
            if not config.GUARD_CLOUD_FALLBACK:
                log.warning("Guard (local) failed after %.1fs, failing open", loop.time() - t0)
                return False
            log.warning("Guard (local) failed after %.1fs, trying cloud fallback", loop.time() - t0)
        else:
            log.info("guard (local): %.1fs -> %r", loop.time() - t0, verdict.strip()[:10])
            return _guard_yes(verdict)

    # Reached when: local model is off, OR the local guard failed and GUARD_CLOUD_FALLBACK is on.
    try:
        verdict = await asyncio.wait_for(_post(messages, config.GUARD_MODEL), timeout=config.GUARD_TIMEOUT_SECONDS)
    except Exception:
        log.warning("Guard (cloud) failed after %.1fs, failing open", loop.time() - t0)
        return False
    log.info("guard (cloud): %.1fs -> %r", loop.time() - t0, verdict.strip()[:10])
    return _guard_yes(verdict)


async def describe_image(image_url: str, prompt: str) -> str:
    """Plain description only — no persona, no classification. Asking the vision model to juggle
    persona + routing + generation in one multimodal call degraded badly on busy/complex images."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    try:
        return await _post(messages, config.VISION_MODEL)
    except (httpx.HTTPStatusError, httpx.TimeoutException, MalformedResponseError):
        if not config.VISION_FALLBACK_MODEL:
            raise
        return await _post(messages, config.VISION_FALLBACK_MODEL)


async def summarize_in_character(description: str, user_query: str, system_prompt: str, reminder: str) -> str:
    """Second, text-only call with a fresh context — no image, no chat history, just the
    description — asking a compact model to react briefly in character. The vision model itself
    proved too slow/unreliable for this text-only step. Falls back to a distinct model (never the
    same one twice) with its own short timeout, so a bad run never takes minutes."""
    prompt = f'Вот подробное описание изображения: "{description}".'
    if user_query:
        prompt += f' Вопрос/подпись пользователя к фото: "{user_query}".'
    prompt += " Кратко прокомментируй это в своей манере."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "system", "content": reminder},
    ]
    try:
        return await asyncio.wait_for(_post(messages, config.SUMMARY_MODEL), timeout=config.SUMMARY_TIMEOUT_SECONDS)
    except Exception:
        log.warning("Summary model (%s) failed/timed out, falling back to %s", config.SUMMARY_MODEL, config.SUMMARY_FALLBACK_MODEL)

    return await asyncio.wait_for(
        _post(messages, config.SUMMARY_FALLBACK_MODEL), timeout=config.SUMMARY_FALLBACK_TIMEOUT_SECONDS
    )
