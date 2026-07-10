"""Reader-specific knobs, .env-overridable like the main config. The speed numbers are REAL Phase 0
measurements on this box (Ryzen 5600H, CPU-only llama-server, 2026-07-08):

    prefill (fresh)  ~14.6 tok/s      <-- NOT negligible: it DOMINATES both ingest and answers
    decode           ~3.7-4.3 tok/s
    per-slot context 4096 tok (-c 8192 -np 2) — a whole reader request must fit ~4000 tok

The tier ETA and the per-answer budget are computed from these; re-measure (bench_phase0) and update
.env if the server flags or the machine change.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Measured llama-server speeds (tok/s).
PREFILL_TOK_S = float(os.environ.get("READER_PREFILL_TOK_S", "14.6"))
DECODE_TOK_S = float(os.environ.get("READER_DECODE_TOK_S", "3.8"))

# ~chars per token for Russian prose through Gemma's tokenizer (checked against Phase 0: 8000 chars
# of the HABR draft = 2494 tok ≈ 3.2). Used only for budgeting/ETA, never for offsets.
CHARS_PER_TOKEN = 3.2

# Per-slot context is 4096 (-c 8192 / -np 2). Budget a whole answer request inside it:
#   system prompt ~250 + question ~100 + context CONTEXT_TOKENS + answer ANSWER_MAX_TOKENS + slack.
CONTEXT_TOKENS = int(os.environ.get("READER_CONTEXT_TOKENS", "2200"))     # retrieved parents, total
ANSWER_MAX_TOKENS = int(os.environ.get("READER_ANSWER_MAX_TOKENS", "300"))
# Global questions (plot retellings over chapter summaries) legitimately need longer answers —
# a 300-token cap cut a live "tell me the plot" mid-word. ~2 min of decode at measured speeds.
ANSWER_MAX_TOKENS_GLOBAL = int(os.environ.get("READER_ANSWER_MAX_TOKENS_GLOBAL", "550"))

# Retrieval.
RETRIEVE_LIMIT = int(os.environ.get("READER_RETRIEVE_LIMIT", "24"))       # hybrid candidates before rerank
RERANK_ENABLED = os.environ.get("READER_RERANK", "true").strip().lower() in ("1", "true", "yes", "on")
RERANK_KEEP = 8                                                            # leaves kept after rerank
# bge-reranker-v2-m3 normalized score below which we honestly refuse instead of answering
# (brief §5 step 6: a hallucination about the book is worse than a refusal). Tune on the golden set.
MIN_SCORE = float(os.environ.get("READER_MIN_SCORE", "0.30"))
MAX_PARENTS = 3                                                            # parents fed to generation

# Summaries (medium tier).
SUMMARY_INPUT_CHARS = 9000      # ~2800 tok of chapter text per summary call (slot budget!)
SUMMARY_MAX_TOKENS = 250        # short output on purpose — decode is the expensive direction

# Ingest niceness: seconds of user-facing LLM quiet time required before the worker fires the next
# summary call (the 12B is shared; brief §11 contention note).
INGEST_YIELD_SECONDS = int(os.environ.get("READER_INGEST_YIELD_SECONDS", "20"))

# Telegram file limit for getFile (brief §4): bigger uploads can't be downloaded by a standard bot.
MAX_FILE_BYTES = 19 * 1024 * 1024

LOCAL_CHAT_URL = os.environ.get("LOCAL_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")

# Streaming answer: how often (s) the telegram message is edited with accumulated text.
# Groups are flood-limited to ~20 messages+edits/min per chat; 4s ≈ 15/min leaves headroom for
# the user's own messages (2.5s tripped RetryAfter mid-stream in live use).
STREAM_EDIT_INTERVAL = 4.0
