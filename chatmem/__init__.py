# Chat-memory RAG: long-term, attributed memory of the group chat. Sources: Telegram Desktop
# HTML export (backfill) + the live message stream. Media becomes text (grind pipeline), text
# becomes dialogue windows, windows become embeddings in Qdrant. Design notes: HABR_DRAFT.md
# and the reader/ module this reuses (embedder, store patterns, contention protocol).
