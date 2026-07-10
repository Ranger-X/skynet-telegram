# skynet-telegram

A Telegram bot that role-plays the T-800 in a group chat with friends — and gradually grew a local
LLM stack that has no business fitting on a gaming laptop, yet does:

- **Persona chat** on a local Gemma 4 12B (llama.cpp, CPU) with a fallback to free OpenRouter
  models: jabs, reactions, provocative takes, news, web search, dossiers on chat members.
- **Native multimodality**: photos, albums, voice messages, GIFs, video, video notes, stickers — one
  model, one engine.
- **Long-reader**: RAG over books (FB2/EPUB) with a **spoiler filter** — the bot answers only from
  the part of the book you have actually read; it physically cannot peek ahead (`/pos`, `/ask`).
- **Manga/comics** (CBZ/CBR/PDF): a vision model reads the pages (lines from speech bubbles + scenes),
  and from there it is the same page-based, spoiler-filtered RAG.
- **Chat memory**: the entire message history (Telegram Desktop export) in a vector index — the bot
  recalls on its own when you say "remember when...", `/recall` answers "who, what and when" with
  links back to the original messages, and `/profile` builds a dossier from a person's whole history.
- **Deep research** (`/research`): multi-angle web search + source analysis + a verdict with
  citations — for settling group-chat arguments.

## Model layout (laptop: Ryzen 5600H, 16 GB RAM, RTX 3050 4 GB)

| Job | Model | Hardware |
|---|---|---|
| Persona, replies, chapter summaries | Gemma 4 12B Q4 (heretic) | CPU, llama.cpp, KV q8_0 + FA |
| Bulk image/manga reading | Qwen3-VL-4B Q4 | whole GPU (CUDA build), port 8081 |
| Bulk voice transcription | faster-whisper small int8 | CPU |
| Embeddings (books, chat) | bge-m3 fp16 | CPU, resident |
| Reranker | bge-reranker-v2-m3 | CPU, lazily under a RAM guard |
| Vector store | Qdrant embedded | disk, no server |

Both LLMs run **simultaneously** (each on its own hardware). The vision grinder is optional: without
it, image batches fall back gracefully to the main model.

## Installation

1. **Python 3.13+**, `python -m venv .venv`, `pip install -r requirements.txt`.
2. **llama.cpp**: a regular build for the persona; a CUDA build if you want the GPU grinder.
3. **Models**: GGUF Gemma 4 12B (Q4_K_M) + the **official bf16 mmproj** (third-party F16 projector
   exports produce garbage); optionally Qwen3-VL-4B GGUF + mmproj for the grinder.
4. **Tools**: a full ffmpeg (the pip build is missing filters), 7-Zip (for CBR).
5. `cp .env.example .env`, fill in the tokens and paths.
6. Run: `toggle-bot.cmd` (brings up llama-server + the bot; run it again to shut them down),
   `toggle-grinder.cmd` — a separate switch for the GPU grinder (shut it down before gaming — VRAM
   is shared).

The first run downloads bge-m3 (~2.3 GB) and, if used, whisper (~0.5 GB) from HuggingFace.

## Commands

**Chat:** `/tease` `/react` `/horn` `/new` `/search` `/research` `/profile` `/time` `/help`
**Books/manga:** send a file → pick the analysis depth → `/pos` (bookmark) → `/ask`, `/chapters`,
`/books`, `/book`, `/tier`
**Chat memory:** `/recall`, `/memstat`; owner-only — `/memload`, `/memgrind`, `/memwipe`
**Language:** `/lang en|ru` — switch the bot's language for this chat (English by default, per-chat).

## Architecture notes

- `long-reader-architecture.md` — the RAG module brief, with a section of corrections for the real
  hardware (measured speeds, context budgets, why prefill on CPU is not free).
- `HABR_DRAFT.md` — a draft article covering the whole development story: the saga of Gemma 4's
  vision, KV quantization, RAM guards, the "4B reads better than 12B" benchmark, and the other
  pitfalls, with the lessons they taught.

Private data (tokens, chat history, reading positions, books) lives in `.env`, `state.json`,
`reader.db`, `qdrant_data/`, `reader_files/`, `chatmem_*.json` — all of it is in `.gitignore` and
never reaches the repository.

## License

MIT.
