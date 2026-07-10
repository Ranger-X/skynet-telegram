"""Semantic route classifier for /ask: global (chapter summaries) vs local (leaf retrieval).

The regex markers stay as the high-precision first tier (an explicit «по всей книге» must never
depend on a threshold). This adds a second tier for everything else: the question's dense
embedding — already computed for retrieval, so free at query time — is compared against two
anchor sets of prototype questions. Anchors are embedded once per process and cached.

bge-m3 dense vectors are L2-normalized, so cosine = dot product. Score per class = mean of the
top-3 anchor similarities (robust to any single odd anchor). Decision by margin; near-ties fall
back to local — the cheaper and more honest path (it can refuse, summaries can't cite verbatim).

Anchors come as per-language sets (English primary, Russian retained) and are embedded once per
language per process, then cached. classify() picks the set matching the chat's language so the
question is scored against prototypes in the same language.
"""

import asyncio
import logging

log = logging.getLogger("t800.reader")

GLOBAL_ANCHORS = {
    "en": [
        "tell me the whole plot of the book",
        "briefly retell the entire story",
        "how did the main character change over the course of the story",
        "what themes does the author raise",
        "what is this work about",
        "how did the characters' relationships develop through the book",
        "what is the main message of the book",
        "describe the atmosphere and style of the work",
        "how does it all end and what do the characters arrive at",
        "what are the key events in the book",
        "who are all the main characters in this story",
        "how does the author explore the theme of death throughout the book",
    ],
    "ru": [
        "расскажи сюжет книги целиком",
        "перескажи кратко всю историю",
        "как менялся главный герой на протяжении истории",
        "какие темы поднимает автор",
        "о чём это произведение",
        "как развивались отношения героев по ходу книги",
        "какая главная мысль книги",
        "опиши атмосферу и стиль произведения",
        "чем всё закончилось и к чему пришли герои",
        "какие ключевые события происходят в книге",
        "кто все главные персонажи этой истории",
        "как автор раскрывает тему смерти через всю книгу",
    ],
}
LOCAL_ANCHORS = {
    "en": [
        "what did the hero do in this scene",
        "who is this character",
        "what happened in the church",
        "what was the name of the protagonist's dog",
        "where did the characters first meet",
        "what did X say when he found out about Y",
        "what color was the sword",
        "why did he hit him in that chapter",
        "what happened at the start of the third chapter",
        "where did they go after the conversation",
        "quote what was written in the letter",
        "how old was the heroine when they met",
    ],
    "ru": [
        "что сделал герой в этой сцене",
        "кто такой этот персонаж",
        "что было в церкви",
        "как звали собаку главного героя",
        "где произошла первая встреча героев",
        "что сказал X, когда узнал про Y",
        "какого цвета был меч",
        "почему он ударил его в той главе",
        "что случилось в начале третьей главы",
        "куда они поехали после разговора",
        "процитируй, что было написано в письме",
        "сколько лет было героине в момент знакомства",
    ],
}

# Calibrated 2026-07-10 on a 12-question set: true globals scored >= +0.089, true locals
# <= +0.028 — 0.055 splits them cleanly (12/12). Below the margin -> local: the cheaper path,
# and the one that can honestly refuse.
MARGIN = 0.055

_anchor_cache: dict[str, tuple[list[list[float]], list[list[float]]]] = {}
_alock = asyncio.Lock()


async def _anchors(lang: str) -> tuple[list[list[float]], list[list[float]]]:
    global _anchor_cache
    async with _alock:
        if lang not in _anchor_cache:
            from . import embedder

            g_anchors = GLOBAL_ANCHORS.get(lang, GLOBAL_ANCHORS["en"])
            l_anchors = LOCAL_ANCHORS.get(lang, LOCAL_ANCHORS["en"])
            embs = await embedder.embed_texts(g_anchors + l_anchors)
            dense = [e["dense"] for e in embs]
            _anchor_cache[lang] = (dense[: len(g_anchors)], dense[len(g_anchors):])
            log.info("router anchors embedded for %s (%d global, %d local)", lang, len(g_anchors), len(l_anchors))
        return _anchor_cache[lang]


def _top3_mean(q: list[float], anchors: list[list[float]]) -> float:
    sims = sorted((sum(a * b for a, b in zip(q, v)) for v in anchors), reverse=True)
    return sum(sims[:3]) / 3


async def classify(q_dense: list[float], lang: str = "en") -> tuple[str, float]:
    """-> (route, margin). Call only when the regex tier didn't already decide."""
    ga, la = await _anchors(lang)
    g = _top3_mean(q_dense, ga)
    l = _top3_mean(q_dense, la)
    margin = g - l
    route = "global" if margin > MARGIN else "local"
    log.info("router: semantic %s (global=%.3f local=%.3f margin=%+.3f)", route, g, l, margin)
    return route, margin
