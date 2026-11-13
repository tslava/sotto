"""Runtime venue selection (PLAN.md, stage 5; CLAUDE.md, "Stage 5 interfaces").

Pure synchronous functions with no I/O. Filling the cache (Google Places, LLM compression
of reviews) is done by a separate offline script (``places_fill.py``, another agent); this
module only reads ready ``db.PlaceRow`` objects and decides what to mix into the prompt
based on the trigger message text:

- ``select_places`` / ``render_places_block``: the old path (regex detection + a filtered
  list of 1-2 places); kept for ``replay.py`` and tests, ``responder._generate_and_send``
  no longer calls them;
- ``render_places_menu``: the runtime path on direct addressing: the whole cache at once,
  leaving the decision "was a place asked about" and the choice of 1-2 suitable ones to the
  model (an owner decision after a live test: regex does not cover "where's a decent beer
  around here");
- ``validate_fact``: the same validation ``places_fill.py`` runs on the compressed review
  before writing to the DB. A review is foreign text from the internet, a channel for
  indirect injection (PLAN.md stage 5 item 3): a command in a review could travel through
  LLM compression into the fact and from there into every venue prompt, so the filter is
  strict (only Cyrillic, spaces, comma, hyphen, up to 40 characters) and is the last
  automatic defence before manual review.
"""

from __future__ import annotations

import random

from sotto.config_models import PlacesConfig
from sotto.db import PlaceRow
from sotto.fragments import Fragments
from sotto.lang import Language

_OUTSKIRTS_CATEGORY = "outskirts"

_FACT_MAX_LEN = 40


def _normalize(text: str, lang: Language) -> str:
    """``lang.fold`` (case, "yo" letter), for the case-insensitive comparison from the contract."""
    return lang.fold(text.strip())


def _district_lookup(cfg: PlacesConfig, lang: Language) -> dict[str, str]:
    """Spelling variant (case- and yo/ye-insensitive) -> canonical district name from
    ``district_aliases``."""
    lookup: dict[str, str] = {}
    for canonical, variants in cfg.district_aliases.items():
        lookup[_normalize(canonical, lang)] = canonical
        for variant in variants:
            lookup[_normalize(variant, lang)] = canonical
    return lookup


def _district_group(name: str, lookup: dict[str, str], lang: Language) -> str | None:
    """Canonical district for an arbitrary spelling (PlaceRow.district), or None."""
    return lookup.get(_normalize(name, lang))


def _detect_district(request_text: str, lookup: dict[str, str], lang: Language) -> str | None:
    """First district mentioned in the text (any spelling variant), or None."""
    normalized = _normalize(request_text, lang)
    for variant, canonical in lookup.items():
        if variant and variant in normalized:
            return canonical
    return None


def _mentions_any(request_text: str, keywords: list[str], lang: Language) -> bool:
    normalized = _normalize(request_text, lang)
    return any(_normalize(keyword, lang) in normalized for keyword in keywords)


def select_places(
    rows: list[PlaceRow],
    cfg: PlacesConfig,
    request_text: str,
    rng: random.Random,
    *,
    lang: Language,
) -> list[PlaceRow]:
    """Filters ``rows`` by the request text and picks up to ``cfg.max_per_reply``.

    Order (CLAUDE.md, "Stage 5 interfaces"):

    1. Base: ``operational``, ``rating >= cfg.min_rating``, ``reviews >= cfg.min_reviews``.
    2. If the request has words about quiet: a hard ``quiet=1`` filter (empty if none of the
       remaining places is quiet).
    3. If a district is mentioned it takes priority: narrow to district matches, but only if
       there are any; otherwise (the district was not met in the cache) use what there was
       before this step rather than an empty list.
    4. Otherwise, if the request has words about going out of town: a hard filter
       ``category == "outskirts"``.
    5. Otherwise any category.

    From the resulting set, ``rng.sample`` up to ``max_per_reply`` (or fewer if there are
    not enough suitable ones). An empty list at any step -> ``[]``.
    """
    candidates = [
        row
        for row in rows
        if row.operational and row.rating >= cfg.min_rating and row.reviews >= cfg.min_reviews
    ]
    if not candidates:
        return []

    if _mentions_any(request_text, cfg.quiet_keywords, lang):
        candidates = [row for row in candidates if row.quiet]
        if not candidates:
            return []

    lookup = _district_lookup(cfg, lang)
    district = _detect_district(request_text, lookup, lang)
    if district is not None:
        district_matches = [
            row for row in candidates if _district_group(row.district, lookup, lang) == district
        ]
        if district_matches:
            candidates = district_matches
    elif _mentions_any(request_text, cfg.outskirts_keywords, lang):
        candidates = [row for row in candidates if row.category == _OUTSKIRTS_CATEGORY]
        if not candidates:
            return []

    if not candidates:
        return []

    k = min(cfg.max_per_reply, len(candidates))
    if k <= 0:
        return []
    return rng.sample(candidates, k)


def render_places_block(rows: list[PlaceRow], *, frag: Fragments) -> str:
    """``{places}`` block: name, district, category, fact. Empty -> prompt.PLACES_NONE.

    Opening hours are never printed (they are not in PlaceRow) and the source (Google) is not
    mentioned: "been there, nothing special", not "according to Google reviews".
    """
    if not rows:
        return str(frag.text("prompt.places_none"))
    lines = [str(frag.text("render.places_block_header"))]
    for row in rows:
        parts = [row.name, row.district, row.category]
        if row.fact:
            parts.append(row.fact)
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def render_places_menu(rows: list[PlaceRow], *, frag: Fragments) -> str:
    """The whole venue cache for the prompt on direct addressing (mention/reply/name).

    An owner decision after a live test (CLAUDE.md, "Stage 5 interfaces"): detecting a place
    request by regex does not cover live speech ("where's a decent beer around here", "any
    quiet place in the neighbourhood?"), so on any direct addressing the list goes into the
    prompt whole, and the decision "was a place asked about" is made by the model following
    the instruction in the same block, not by ``select_places``.
    The regexes (``patterns.places_request``) remain for statistics and ambient.

    Empty -> the ``prompt.places_none`` fragment. Otherwise lines by ``name``, quiet/noisy
    per ``quiet``, ``fact`` only if non-empty. No ratings, prices, opening hours or the word
    Google (the source is not given away, as in ``render_places_block``).
    """
    if not rows:
        return str(frag.text("prompt.places_none"))
    lines = [str(frag.text("render.places_menu_header"))]
    quiet, noisy = frag.text("render.places_quiet"), frag.text("render.places_noisy")
    for row in sorted(rows, key=lambda r: r.name):
        parts = [row.name, row.district, row.category, quiet if row.quiet else noisy]
        if row.fact:
            parts.append(row.fact)
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def validate_fact(raw: str, *, lang: Language) -> str | None:
    """Only Cyrillic, spaces, comma, hyphen; <= 40 characters; otherwise None.

    A Google review is untrusted external text (PLAN.md stage 5 item 3): Latin, digits and
    any other punctuation fail the whole fact rather than being merely stripped from it:
    partial cleaning would leave a chance to smuggle an injection through in fragments.
    """
    text = raw.strip()
    if not text or len(text) > _FACT_MAX_LEN:
        return None
    if not lang.fact_re().fullmatch(text):
        return None
    return text
