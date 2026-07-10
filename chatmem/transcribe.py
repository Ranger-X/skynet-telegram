"""faster-whisper wrapper for batch voice transcription (chat memory grind).

Measured on this box (2026-07-09, real chat voices): small/int8 on CPU does 4.5x realtime on a 29s
voice and 2.7x on a 5-minute one, Russian quality on par with Gemma-native — while the LLM stays
free. Gemma's native audio costs ~27 context tokens per second of audio (a 5-min voice would blow
the 4096 slot), so whisper IS the batch path; Gemma keeps the live listen-and-reply path.

Model files cache under D:\\hf-cache (HF_HOME). Lazy-loaded; ~2.5 GB RAM while resident — load on
first grind, keep for the batch, freed only with the process.
"""

import asyncio
import logging
import time

log = logging.getLogger("t800.chatmem")

_model = None
_wlock = asyncio.Lock()  # one transcription at a time — whisper saturates the cores by itself


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        log.info("loading faster-whisper small/int8 (first use)...")
        t0 = time.monotonic()
        _model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=6)
        log.info("whisper loaded in %.0fs", time.monotonic() - t0)
    return _model


def _transcribe(path: str) -> str:
    model = _get_model()
    t0 = time.monotonic()
    segments, info = model.transcribe(path, language="ru", vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    log.info("whisper: %.0fs audio in %.1fs (%.1fx)", info.duration, time.monotonic() - t0,
             info.duration / max(0.1, time.monotonic() - t0))
    return text


async def transcribe(path: str) -> str:
    async with _wlock:
        return await asyncio.to_thread(_transcribe, path)
