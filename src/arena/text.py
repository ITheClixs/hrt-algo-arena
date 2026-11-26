"""Headline parsing: company detection, template extraction, keyword scores, documents.

HRT's synthetic headlines are '<Company> <event phrase>' built from a few hundred templates
with numbers filled in. Turning each headline into its template (company and numbers masked)
is the single most useful step: impacts are then estimated per template.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from arena.data import Sessions

MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:[KMBT]\b|million\b|billion\b|bn\b|mn\b)?", re.I)
PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")

POSITIVE_WORDS = (
    "secures", "beats", "record", "raises", "breakthrough", "wins", "award", "expands",
    "launches", "improvement", "strong", "upgrade", "approval", "partnership", "buyback",
    "surge", "exceeds", "growth",
)
NEGATIVE_WORDS = (
    "misses", "decline", "recall", "lawsuit", "regulatory review", "loses", "drop", "cuts",
    "downgrade", "delay", "investigation", "weak", "warning", "layoffs", "resigns", "breach",
    "concerns", "probe", "fine",
)


def mask_numbers(text: pd.Series) -> pd.Series:
    """'$120M' -> '$X', '25%' -> 'N%', other numbers -> 'N'."""
    return (
        text.str.replace(MONEY, "$X", regex=True)
        .str.replace(PERCENT, "N%", regex=True)
        .str.replace(NUMBER, "N", regex=True)
    )


def detect_entities(headlines: pd.Series, base_tokens: int = 2, max_tokens: int = 4) -> pd.Series:
    """Company = first `base_tokens` words, extended while that prefix is always followed by
    the same next word ('Jovik Asset' -> 'Jovik Asset Group')."""
    words = headlines.str.split()
    entity = words.str[:base_tokens].str.join(" ")
    for k in range(base_tokens, max_tokens):
        nxt = words.str[k]
        frame = pd.DataFrame({"entity": entity, "nxt": nxt})
        stats = frame.groupby("entity")["nxt"].agg(["nunique", "size"])
        fixed = stats.index[(stats["nunique"] == 1) & (stats["size"] >= 2)]
        extend = entity.isin(fixed) & (words.str.len() > k + 1)
        if not extend.any():
            break
        entity = entity.where(~extend, entity + " " + nxt.fillna(""))
    return entity


def templates(headlines: pd.Series, entities: pd.Series | None = None) -> pd.Series:
    """Headline -> event template with the company and the numbers masked."""
    entity = detect_entities(headlines) if entities is None else entities
    rest = [
        h[len(e) :].strip() if isinstance(h, str) and h.startswith(e) else h
        for h, e in zip(headlines, entity, strict=True)
    ]
    return mask_numbers(pd.Series(rest, index=headlines.index, dtype="object")).str.strip()


def keyword_score(
    headlines: pd.Series,
    positive: tuple[str, ...] = POSITIVE_WORDS,
    negative: tuple[str, ...] = NEGATIVE_WORDS,
) -> pd.Series:
    """+1 per positive word, -1 per negative word. A sanity baseline, not a strategy."""
    low = headlines.str.lower()

    def hits(words: tuple[str, ...]) -> pd.Series:
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")"
        return low.str.count(pattern)

    return (hits(positive) - hits(negative)).astype(float)


def session_documents(s: Sessions, first_bar: int = 0, last_bar: int | None = None) -> list[str]:
    """One string per session (all its headlines in the bar window) for TF-IDF models."""
    hl = s.headlines
    mask = hl["bar_ix"] >= first_bar
    if last_bar is not None:
        mask &= hl["bar_ix"] <= last_bar
    docs = hl[mask].groupby("sidx")["headline"].agg(" . ".join)
    return docs.reindex(np.arange(s.n), fill_value="").tolist()


# Slot phrases that vary inside otherwise identical templates, listed from the practice data
# (research/r1_generator.py --vocab). Replacing them gives coarser event types with more data each.
SLOT_VOCAB: dict[str, tuple[str, ...]] = {
    "SECTOR": ("automated logistics", "cloud infrastructure", "data analytics", "digital payments",
               "enterprise software", "precision manufacturing", "process automation",
               "renewable storage", "supply chain optimization", "wireless connectivity"),
    "REGION": ("Asia Pacific", "Central Europe", "Europe", "Latin America", "Middle East",
               "North America", "Scandinavia", "Southeast Asia"),
    "PARTNER": ("a global retailer", "a leading cloud platform", "a major logistics provider",
                "a multinational manufacturer", "a national infrastructure agency",
                "a top-tier research institute", "an international consortium"),
    "OFFICER": ("Chief Operating Officer", "Chief Strategy Officer", "CEO", "CFO", "CTO"),
    "EVENT": ("Annual Industry Forum", "Capital Markets Day", "Corporate Strategy Symposium",
              "Emerging Markets Forum", "Global Innovation Summit", "Tech Leadership Conference"),
}


def coarse_types(templates: pd.Series) -> pd.Series:
    """Template -> coarse event type: slot phrases (sector, region, partner, ...) become tags."""
    out = templates.astype("object")
    for tag, phrases in SLOT_VOCAB.items():
        longest_first = sorted(phrases, key=len, reverse=True)
        pattern = r"\b(?:" + "|".join(re.escape(p) for p in longest_first) + r")\b"
        out = out.str.replace(pattern, tag, regex=True)
    return out
