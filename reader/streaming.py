"""Throttled message-editing for streamed answers (/ask, /research). Telegram group chats allow
roughly 20 messages/edits per minute per chat; exceeding it raises RetryAfter, and a naive edit
loop that lets that propagate kills the generation it was displaying (seen live: a /research
report frozen at '▌' one paragraph before the end). Rules here:

- streaming edits NEVER raise: RetryAfter arms a skip-until backoff, transient network errors are
  ignored (the next tick catches the text up);
- the FINAL text is delivered at all costs: retry after the flood wait, and if editing still
  fails, send it as a NEW message so the result is never lost.
"""

import asyncio
import logging
import time

from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from . import rconfig

log = logging.getLogger("t800.reader")


class StreamEditor:
    def __init__(self, status_message):
        self.message = status_message
        self._last_edit = 0.0
        self._skip_until = 0.0

    async def maybe_edit(self, text: str) -> None:
        """Throttled in-stream update; silently skips when rate-limited. Never raises."""
        now = time.monotonic()
        if now < self._skip_until or now - self._last_edit < rconfig.STREAM_EDIT_INTERVAL:
            return
        self._last_edit = now
        try:
            await self.message.edit_text(text[:4090])
        except RetryAfter as e:
            self._skip_until = now + e.retry_after + 1.0
            log.info("stream edit flood-limited, pausing edits for %.0fs", e.retry_after + 1.0)
        except (TimedOut, NetworkError):
            pass  # transient; the next tick resends the accumulated text anyway
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                log.warning("stream edit rejected: %s", e)

    async def finalize(self, text: str) -> None:
        """Deliver the final text: wait out a flood limit and retry; fall back to a new message."""
        text = text[:4090]
        for attempt in (1, 2):
            try:
                await self.message.edit_text(text)
                return
            except RetryAfter as e:
                if attempt == 1:
                    await asyncio.sleep(e.retry_after + 1.0)
                    continue
            except BadRequest as e:
                if "not modified" in str(e).lower():
                    return
                break
            except (TimedOut, NetworkError):
                await asyncio.sleep(3)
                continue
        try:
            await self.message.reply_text(text)
        except Exception:
            log.exception("finalize failed even as a new message")
