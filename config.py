import os

from dotenv import load_dotenv

load_dotenv()


def _bool(value, default=False):
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _chat_ids(value):
    if not value:
        return set()
    return {int(x.strip()) for x in value.split(",") if x.strip()}


TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

# --- Local model (llama-server / llama.cpp) ---------------------------------
# When enabled, the persona brain (get_reply) runs on a local llama-server via its OpenAI-compatible
# /v1/chat/completions with thinking disabled. Chosen over Ollama because llama-server reuses the KV
# cache prefix across requests (the long system prompt is prefilled once, not on every message) —
# ~13s per follow-up reply vs ~31s on Ollama. Falls back to OpenRouter if llama-server is down/slow.
USE_LOCAL_MODEL = _bool(os.environ.get("USE_LOCAL_MODEL"), default=False)
LOCAL_CHAT_URL = os.environ.get("LOCAL_CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "gemma4")  # label only; llama-server serves the loaded model
LOCAL_MAX_TOKENS = int(os.environ.get("LOCAL_MAX_TOKENS", "512"))  # safety ceiling on reply length
LOCAL_TIMEOUT_SECONDS = int(os.environ.get("LOCAL_TIMEOUT_SECONDS", "90"))  # cold prefill ~55s (--no-warmup makes the FIRST request slower); warm ~7s

# Watchdog: llama-server is the text + guard + VISION + AUDIO brain (one unified engine), and RAM
# pressure on this box can crash it mid-session. A JobQueue task pings its /health and relaunches it on
# a sustained outage. Args/paths mirror how toggle-bot.ps1 starts it. The mmproj is the OFFICIAL bf16
# projector (our own F16 export is broken); loading it into llama-server is what gives Gemma 4's
# encoder-free vision/audio via the OpenAI /v1 endpoint. --no-warmup is required (gemma4's warmup graph
# otherwise crashes with the multimodal context).
LOCAL_HEALTH_URL = os.environ.get("LOCAL_HEALTH_URL", "http://127.0.0.1:8080/health")
LLAMA_SERVER_EXE = os.environ.get("LLAMA_SERVER_EXE", r"D:\llamacpp\llama-server.exe")
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", r"D:\gemma4-gguf\gemma-4-12b-it-Q4_K_M.gguf")
LLAMA_MMPROJ_PATH = os.environ.get(
    "LLAMA_MMPROJ_PATH",
    r"D:\ollama-models\blobs\sha256-675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842",
)
LLAMA_SERVER_LOG = os.environ.get("LLAMA_SERVER_LOG", r"D:\llamacpp\server.log")
LLAMA_WATCHDOG_INTERVAL = int(os.environ.get("LLAMA_WATCHDOG_INTERVAL", "30"))  # health-check period (s)

# Vision + audio now run on the SAME llama-server as text (the mmproj projector is loaded into it), via
# the OpenAI /v1 endpoint — no separate engine, no per-photo cold load. USE_LOCAL_AUDIO gates voice-
# message understanding (Gemma 4 audio input; experimental — reduced quality on fast speech).
USE_LOCAL_VISION = _bool(os.environ.get("USE_LOCAL_VISION"), default=False)
USE_LOCAL_AUDIO = _bool(os.environ.get("USE_LOCAL_AUDIO"), default=False)
# Full-featured ffmpeg build (the pip one on PATH lacks denoise filters). Used for voice OGG->WAV and
# for sampling frames out of GIFs/animations. ffprobe sits next to it.
FFMPEG_EXE = os.environ.get("FFMPEG_EXE", r"D:\ffmpeg-full\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe")
FFPROBE_EXE = os.environ.get("FFPROBE_EXE", r"D:\ffmpeg-full\ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe")
ANIMATION_FRAMES = int(os.environ.get("ANIMATION_FRAMES", "6"))  # frames sampled from a GIF for Gemma 4
# Frames sampled from a real video / video-note (кружок). A few more than a GIF for longer clips, but
# capped so CPU prefill stays sane — evenly spread across the whole clip (not a strict 1 fps like
# Gemma's reference, which for 60s would be 60 frames and minutes of CPU prefill).
VIDEO_FRAMES = int(os.environ.get("VIDEO_FRAMES", "10"))
# Max width (px) each sampled GIF/video frame is downscaled to before the model sees it. Small on
# purpose: for following the ACTION/plot the model doesn't need fine detail, and shrinking frames
# cuts the image-token count (and thus CPU prefill) sharply. Single photos are NOT affected — they go
# full-res through a different path, so detail questions ("что на вывеске") still work.
FRAME_MAX_WIDTH = int(os.environ.get("FRAME_MAX_WIDTH", "256"))
LOCAL_VISION_TIMEOUT_SECONDS = int(os.environ.get("LOCAL_VISION_TIMEOUT_SECONDS", "150"))  # encoder pass + gen on CPU; headroom for a 10-frame video

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "poolside/laguna-xs-2.1:free")
# Used if the primary model gets rate-limited (429) upstream — free models are prone to this. Empty = no fallback.
OPENROUTER_FALLBACK_MODEL = os.environ.get("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
# Vision-capable model used to describe photos — the persona model itself only reacts to the description.
VISION_MODEL = os.environ.get("VISION_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")
# Used if the primary vision model errors out (502s/timeouts are common on free vision models). Empty = no fallback.
VISION_FALLBACK_MODEL = os.environ.get(
    "VISION_FALLBACK_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
)
VISION_TIMEOUT_SECONDS = 110  # covers a primary attempt plus one fallback attempt, back to back

# Compact text-only model that turns a photo description into an in-character comment. Small vision
# models proved too unreliable (slow, occasionally garbled) for this text-only step; this one is
# faster and steadier. Falls back to a second, distinct COMPACT model on failure (not the big
# 120B one — that reintroduces the long-wait problem this is meant to avoid), with its own short
# timeout so a bad run never takes minutes; if even that fails, the user just gets an error message.
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
SUMMARY_FALLBACK_MODEL = os.environ.get("SUMMARY_FALLBACK_MODEL", "nvidia/nemotron-nano-9b-v2:free")
SUMMARY_TIMEOUT_SECONDS = 20
SUMMARY_FALLBACK_TIMEOUT_SECONDS = 25
SUMMARY_TOTAL_TIMEOUT_SECONDS = SUMMARY_TIMEOUT_SECONDS + SUMMARY_FALLBACK_TIMEOUT_SECONDS + 10

TARGET_CHAT_IDS = _chat_ids(os.environ.get("TARGET_CHAT_IDS", ""))
REQUIRE_MENTION_IN_GROUPS = _bool(os.environ.get("REQUIRE_MENTION_IN_GROUPS"), default=True)

# Context window kept per chat. Local CPU inference pays a steep prefill cost per token of context,
# so keep it short locally; the cloud path can afford the full window.
HISTORY_LIMIT = 16 if USE_LOCAL_MODEL else 60
# Outer cap on a full get_reply (local attempt + OpenRouter fallback). Local CPU inference is slow,
# so give it real headroom; cloud finishes long before this anyway.
REPLY_TIMEOUT_SECONDS = (LOCAL_TIMEOUT_SECONDS + 25) if USE_LOCAL_MODEL else 45

TEASE_INTERVAL_SECONDS = int(os.environ.get("TEASE_INTERVAL_SECONDS", "0"))
TEASE_EXCLUDE_USER_IDS = _chat_ids(os.environ.get("TEASE_EXCLUDE_USER_IDS", ""))

# Users allowed to run admin commands (/memload, /memgrind — heavy jobs with filesystem paths;
# chat hooligans already tried). Defaults to TEASE_EXCLUDE_USER_IDS, which has always been
# "the owner" in practice; set OWNER_USER_IDS explicitly to override.
OWNER_USER_IDS = _chat_ids(os.environ.get("OWNER_USER_IDS", "")) or TEASE_EXCLUDE_USER_IDS
TEASE_SEED_USERNAMES = [
    u.strip().lstrip("@") for u in os.environ.get("TEASE_SEED_USERNAMES", "").split(",") if u.strip()
]

# How often (seconds) T-800 replies unprompted to a random recent message from the group. 0 = off by default.
REACT_INTERVAL_SECONDS = int(os.environ.get("REACT_INTERVAL_SECONDS", "0"))
RECENT_MESSAGES_LIMIT = 50  # messages kept per chat for the react job to pick from

# How often (seconds) T-800 drops an unprompted provocative take into the group. 0 = off by default.
HORN_INTERVAL_SECONDS = int(os.environ.get("HORN_INTERVAL_SECONDS", "0"))

# How often (seconds) T-800 drops a random recent news headline into the group. 0 = off by default.
NEWS_INTERVAL_SECONDS = int(os.environ.get("NEWS_INTERVAL_SECONDS", "0"))
# The news feed follows the chat's language — an English chat shouldn't get Russian headlines.
# An explicit NEWS_FEED_URL overrides both (use it to plug in any RSS feed you like).
_NEWS_FEED_OVERRIDE = os.environ.get("NEWS_FEED_URL", "").strip()
NEWS_FEED_URLS = {
    "en": os.environ.get("NEWS_FEED_URL_EN", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"),
    "ru": os.environ.get("NEWS_FEED_URL_RU", "https://news.google.com/rss?hl=ru&gl=RU&ceid=RU:ru"),
}


def news_feed_url(lang: str) -> str:
    return _NEWS_FEED_OVERRIDE or NEWS_FEED_URLS.get(lang, NEWS_FEED_URLS["en"])


def search_region(lang: str) -> str:
    """DuckDuckGo region for /search and /research — 'wt-wt' is the no-region (worldwide) default."""
    return "ru-ru" if lang == "ru" else "wt-wt"

SEARCH_RESULT_COUNT = 5  # results fetched per /search query

USER_ARCHIVE_LIMIT = 300  # messages kept per real user, across chat history, for /profile and tease/react context
USER_QUOTES_FOR_PROMPT = 15  # quotes pulled into tease/react prompts
USER_QUOTES_FOR_PROFILE = 25  # quotes pulled into /profile prompts (kept modest so the prompt fits the
# local CPU's ~60s prefill budget instead of timing out into the OpenRouter fallback)

# Comma-separated @usernames (no @) whose messages the bot must never treat as instructions to follow.
BLACKLISTED_USERNAMES = [
    u.strip().lstrip("@") for u in os.environ.get("BLACKLISTED_USERNAMES", "").split(",") if u.strip()
]

# Small/fast model that screens a message for "make the model hang/degenerate via impossible
# constraints" attacks before the (slow, expensive) persona model ever sees it.
GUARD_MODEL = os.environ.get("GUARD_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")  # cloud path only (USE_LOCAL_MODEL off)
GUARD_TIMEOUT_SECONDS = 30
# When USE_LOCAL_MODEL is on, the guard runs on the local Gemma 4 too. Bigger budget than the cloud
# guard: the guard's prompt prefix differs from the persona's, so on a single-slot llama-server it's
# usually a cold prefill (~25-45s on CPU) rather than a cache hit. On timeout the guard fails open.
GUARD_LOCAL_TIMEOUT_SECONDS = int(os.environ.get("GUARD_LOCAL_TIMEOUT_SECONDS", "50"))
# If the local guard errors, optionally fall back to the cloud GUARD_MODEL before failing open. OFF by
# default — the guard stays fully local and availability is handled at the source by the llama-server
# watchdog. Flip on (GUARD_CLOUD_FALLBACK=true) for debugging or to cover the crash-recovery window.
GUARD_CLOUD_FALLBACK = _bool(os.environ.get("GUARD_CLOUD_FALLBACK"), default=False)
GUARD_MIN_MESSAGE_LENGTH = 200  # only bother screening messages at least this long
