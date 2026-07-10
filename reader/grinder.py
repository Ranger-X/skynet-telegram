"""Vision-describe calls, routable between the main 12B and a dedicated grinder VLM.

GRINDER_CHAT_URL decides where page/photo descriptions go. Default = the main llama-server
(12B, slow, polite to live chat). After the model bench picks a small VLM, point this at its
server (e.g. :8081) — calls then skip the politeness protocol entirely, because a separate
process doesn't contend with the persona for the LLM.
"""

import base64
import logging
import os

import httpx

from . import llm as reader_llm

log = logging.getLogger("t800.reader")

GRINDER_CHAT_URL = os.environ.get("GRINDER_CHAT_URL", "").strip()

MANGA_PAGE_PROMPT = (
    "Это страница манги/комикса. Сначала выпиши реплики и надписи из баблов и рамок в порядке "
    "чтения (манга читается справа налево, сверху вниз), каждую с новой строки в формате «— текст». "
    "Звуковые эффекты, японские иероглифы и повторяющиеся выкрики НЕ выписывай — только осмысленный "
    "текст, каждый бабл один раз. Затем с новой строки после метки «Сцена:» опиши в одном-двух "
    "предложениях, что происходит на панелях. Если текста нет — только «Сцена:». Ничего не выдумывай."
)


def _is_external() -> bool:
    from . import rconfig

    return bool(GRINDER_CHAT_URL) and GRINDER_CHAT_URL != rconfig.LOCAL_CHAT_URL


async def describe_image(jpeg: bytes, prompt: str, max_tokens: int = 350, timeout: float = 600) -> str:
    from . import rconfig

    url = GRINDER_CHAT_URL or rconfig.LOCAL_CHAT_URL
    b64 = base64.b64encode(jpeg).decode("ascii")
    payload = {
        "model": "grinder",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "stream": False,
        "max_tokens": max_tokens,
        # Small VLMs degenerate into «КАК! КАК! КАК!» loops on sound-effect/sparse pages (seen on
        # the GitS bench with BOTH Qwen-4B and the 12B); a firm repeat penalty kills the loop
        # without touching normal dialogue.
        "repeat_penalty": 1.25,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async def _post() -> str:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        tim = data.get("timings", {})
        log.info(
            "grinder describe: prefill=%s tok @ %.1f t/s, out=%s tok @ %.1f t/s",
            tim.get("prompt_n"), tim.get("prompt_per_second") or 0,
            data.get("usage", {}).get("completion_tokens"), tim.get("predicted_per_second") or 0,
        )
        return (data["choices"][0]["message"]["content"] or "").strip()

    if _is_external():
        try:
            return await _post()  # dedicated grinder: no contention, no locks
        except (httpx.ConnectError, httpx.ConnectTimeout):
            # Grinder server is off (it shares VRAM with the user's games and has its own
            # toggle) — fall through to the main 12B with the politeness protocol.
            log.warning("grinder server unreachable at %s, falling back to main LLM", url)
            url = rconfig.LOCAL_CHAT_URL
    # Main 12B: same politeness as book summaries — never talk over the live chat.
    await reader_llm.wait_for_quiet()
    async with reader_llm.llm_lock:
        return await _post()
