"""Offline fill of the venue cache (CLAUDE.md/PLAN.md, stage 5).

``python -m sotto.places_fill [--dry-run] [--no-llm] [--queries "q1;q2"]
    [--manual places_manual.yaml] [--manual-only]``

Not part of the bot runtime: a separate manual run (monthly, PLAN.md stage 7). It calls the
Google Places API (New) ``places:searchText`` for the queries in ``cfg.places.queries``
(or ``--queries``), filters (``businessStatus == OPERATIONAL``, ``rating``,
``userRatingCount``), dedups by ``place.id``, derives the district from
``formattedAddress`` and the category from review text/``primaryType`` (``price_level`` is
unreliable, PLAN.md stage 5 item 4, so the category is not derived from it), compresses the
reviews into one fact via the LLM, validates it, and writes to ``places``
(``Database.upsert_place``). Raw reviews never reach the DB, only the compressed and
validated ``fact``.

``quiet``, ``category`` (if set) and ``district`` (if set) come from
``places_manual.yaml``: Google does not know whether a place is good for conversation
(PLAN.md stage 5 item 5). A Google row is matched to a manual entry by ``name``,
case-insensitively and on word boundaries, not by raw substring (``find_manual_override``):
an exact match of the normalized names OR all words of ``name`` appearing as separate words
in ``displayName``. An applied manual entry is logged at INFO and marked in the report
table with the status "manual".

``--manual-only`` does not call Google at all: it writes only the manual list into
``places`` with ``rating=5.0``, ``reviews=999`` (honestly flagged in the log as "manual
data") and ``operational=1``, so the bot can work from the starter list until a Google
Places key exists (see README, section "Venues").

Dependencies needed by tests are injected explicitly: ``httpx.AsyncClient`` (tests supply
``httpx.MockTransport``) and an object shaped like ``LLMClient.call`` (``LLMLike`` below;
tests supply a fake that does not count budget/money).
``db.PlaceRow``/``Database.upsert_place`` and ``places.validate_fact`` were written in
parallel by another agent (CLAUDE.md, "Stage 5 interfaces"), so they are imported lazily
inside the functions that use them, the same way as in ``replay.py`` (which also imports
``gate``/``patterns`` lazily), so the order in which modules appeared on disk cannot break
test collection for this module.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time as time_module
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from sotto.config import load_config
from sotto.config_models import Config, PlacesConfig
from sotto.fragments import Fragments
from sotto.lang import Language
from sotto.llm import LLMClient, LLMError, LLMResult
from sotto.persona import Persona, persona_from_env
from sotto.prompt import CHAT_CLOSE, CHAT_OPEN
from sotto.sanitize import normalize_text
from sotto.settings import Settings

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_FIELD_MASK = (
    "places.id,places.displayName,places.rating,places.userRatingCount,"
    "places.priceLevel,places.businessStatus,places.formattedAddress,"
    "places.reviews,places.primaryType"
)
# Search bias radius around cfg.places.search_center, in metres.
_SEARCH_RADIUS_M = 15000

_CHEAP_MARKERS = ("cheap", "tanio", "fair price")

_PRICE_LEVELS = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}

_MAX_REVIEWS = 5
_REVIEW_TEXT_LIMIT = 1500
_FACT_MAX_TOKENS = 40
_MANUAL_ONLY_RATING = 5.0
_MANUAL_ONLY_REVIEWS = 999
_MANUAL_ONLY_PREFIX = "manual:"

# A single-word manual name of length <= this counts as a match only on exact equality of
# the whole normalized strings (see find_manual_override): short words like "ARK" would
# otherwise match any place where that word merely appears in passing ("Ark's Pub"),
# not only ARK itself.
_SHORT_MANUAL_NAME_MAX_LEN = 3

_NAME_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_NAME_WS_RE = re.compile(r"\s+")


class LLMLike(Protocol):
    """The subset of LLMClient this module needs; tests supply a fake that does not count
    the bot's budget/accounting (see the module docstring)."""

    async def call(
        self, messages: list[dict[str, str]], *, model: str, max_tokens: int, now: int
    ) -> LLMResult: ...


class _InMemoryState:
    """Minimal state store for the real LLMClient inside places_fill.

    Filling the cache is an offline one-off action: the live bot's daily limits, budget and
    circuit breaker (the same state table in the shared DB) are not needed here and must not
    depend on it, so the LLMClient counters live only for the duration of one run."""

    def __init__(self) -> None:
        self._state: dict[str, str] = {}

    async def get_state(self, key: str) -> str | None:
        return self._state.get(key)

    async def set_state(self, key: str, value: str) -> None:
        self._state[key] = value

    async def increment_state(self, key: str, by: int = 1) -> int:
        new_value = int(self._state.get(key, "0")) + by
        self._state[key] = str(new_value)
        return new_value

    async def add_state_float(self, key: str, by: float) -> float:
        new_value = float(self._state.get(key, "0")) + by
        self._state[key] = str(new_value)
        return new_value


@dataclass(frozen=True, slots=True)
class ManualEntry:
    name: str
    quiet: bool
    category: str | None
    district: str | None


@dataclass(frozen=True, slots=True)
class FetchedPlace:
    place_id: str
    name: str
    district: str
    category: str
    rating: float
    reviews: int
    price_level: int | None
    quiet: bool
    fact: str
    operational: bool
    # True if a places_manual.yaml entry was found for this place (find_manual_override);
    # render_report shows such a row with the status "manual" rather than operational/closed
    # (code review). manual-only entries (build_manual_only_rows) do not set this field:
    # they are fully manual anyway and have their own place_id prefix "manual:".
    manual_override: bool = False


def load_manual(path: Path) -> list[ManualEntry]:
    """places_manual.yaml -> list of ManualEntry. No file -> empty list."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw:
        return []
    entries: list[ManualEntry] = []
    for item in raw:
        entries.append(
            ManualEntry(
                name=str(item["name"]),
                quiet=bool(item.get("quiet", False)),
                category=item.get("category"),
                district=item.get("district"),
            )
        )
    return entries


def _normalize_name(name: str) -> str:
    """casefold + punctuation -> space + collapsed spaces, for comparing names on word
    boundaries (find_manual_override) rather than by raw substring."""
    folded = name.casefold()
    no_punct = _NAME_PUNCT_RE.sub(" ", folded)
    return _NAME_WS_RE.sub(" ", no_punct).strip()


def find_manual_override(display_name: str, manual: list[ManualEntry]) -> ManualEntry | None:
    """Match against the Google displayName on word boundaries, not by substring
    (code review): both names are normalized (casefold, punctuation -> space, collapsed
    spaces); a match is an exact equality of the normalized names OR all words of the manual
    name appearing as separate words in displayName.

    "Komora" matches "Sklep Pub Komora" (the word is contained whole) and "Komora pod
    Baranami" (all the more), but not "Komorka" (a different word, not a substring).
    A single-word manual name of length <= 3 (for example "ARK") is an exception: only an
    exact whole match works for it, otherwise it would match any place where that short word
    appears in passing ("Arkada Wolsanski" does not contain it as a separate word, but
    accidental matches of short words are closer to noise than to signal)."""
    display_normalized = _normalize_name(display_name)
    if not display_normalized:
        return None
    display_words = set(display_normalized.split())
    for entry in manual:
        entry_normalized = _normalize_name(entry.name)
        if not entry_normalized:
            continue
        if entry_normalized == display_normalized:
            return entry
        entry_words = entry_normalized.split()
        if len(entry_words) == 1 and len(entry_words[0]) <= _SHORT_MANUAL_NAME_MAX_LEN:
            continue
        if all(word in display_words for word in entry_words):
            return entry
    return None


def pick_district(*, address: str, manual: ManualEntry | None, cfg: PlacesConfig) -> str:
    if manual is not None and manual.district:
        return manual.district
    folded = address.casefold()
    for district in cfg.district_aliases:
        if district.casefold() in folded:
            return district
    return cfg.default_district


def pick_category(
    *,
    manual: ManualEntry | None,
    review_text: str,
    address: str,
    primary_type: str,
    outskirts_markers: Sequence[str] = (),
) -> str:
    """category: manual > "cheap"/"tanio"/"fair price" in reviews > outskirts by
    address > craft/pub by primaryType (PLAN.md stage 5 item 4, NOT by price_level)."""
    if manual is not None and manual.category:
        return manual.category
    lowered_reviews = review_text.casefold()
    if any(marker in lowered_reviews for marker in _CHEAP_MARKERS):
        return "cheap"
    lowered_address = address.casefold()
    if any(marker.casefold() in lowered_address for marker in outskirts_markers):
        return "outskirts"
    if "bar" in (primary_type or "").casefold():
        return "pub"
    return "craft"


def parse_price_level(raw: object) -> int | None:
    """price_level is unreliable (PLAN.md stage 5 item 4): it is stored as is and the
    category is never derived from it. The New API returns a string enum."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        return _PRICE_LEVELS.get(raw)
    return None


def _display_name(place: dict[str, Any]) -> str:
    display = place.get("displayName")
    if isinstance(display, dict):
        return str(display.get("text", ""))
    return str(display or "")


def build_review_text(reviews: list[dict[str, Any]]) -> str:
    """Up to 5 reviews, joined, normalized and truncated to 1500 characters
    (PLAN.md stage 5 item 3). The raw result is not written to the DB, only fed to the LLM."""
    texts: list[str] = []
    for review in reviews[:_MAX_REVIEWS]:
        text_obj = review.get("text") or review.get("originalText") or {}
        text = text_obj.get("text", "") if isinstance(text_obj, dict) else ""
        if text:
            texts.append(str(text))
    joined = normalize_text(" ".join(texts))
    return joined[:_REVIEW_TEXT_LIMIT]


def passes_filters(place: dict[str, Any], cfg: PlacesConfig) -> bool:
    if cfg.require_operational and place.get("businessStatus") != "OPERATIONAL":
        return False
    rating = place.get("rating")
    if rating is None or float(rating) < cfg.min_rating:
        return False
    reviews = place.get("userRatingCount")
    return reviews is not None and int(reviews) >= cfg.min_reviews


async def compress_fact(
    review_text: str, *, llm: LLMLike | None, model: str, now: int, frag: Fragments, lang: Language
) -> str:
    """Compresses the joined reviews into a fact via the LLM + ``places.validate_fact``.

    ``llm=None`` or an empty ``review_text`` -> ``""`` without a model call (this is
    ``--no-llm``). An invalid model answer -> ``""`` and a WARNING in the log: a review is a
    channel for indirect injection (PLAN.md stage 5 item 3), the fact is reviewed by hand
    before loading, and until then it must be empty rather than contain whatever.
    """
    if llm is None or not review_text:
        return ""
    from sotto.places import validate_fact  # lazy import, see the module docstring

    review_clean = normalize_text(review_text)
    messages = [
        {"role": "system", "content": frag.text("tools.places_fact_prompt")},
        {
            "role": "user",
            "content": (
                f"{frag.get('tools.places_reviews_preamble')}"
                f"\n{CHAT_OPEN}\n{review_clean}\n{CHAT_CLOSE}"
            ),
        },
    ]
    try:
        result = await llm.call(messages, model=model, max_tokens=_FACT_MAX_TOKENS, now=now)
    except LLMError as exc:
        logger.warning("places_fill: llm error while compressing fact: reason=%s", exc.reason)
        return ""

    fact = validate_fact(result.text, lang=lang)
    if fact is None:
        logger.warning("places_fill: invalid fact from LLM, raw=%r", result.text)
        return ""
    return fact


async def search_places(
    http: httpx.AsyncClient, *, query: str, api_key: str, cfg: PlacesConfig
) -> list[dict[str, Any]]:
    latitude, longitude = cfg.search_center
    body = {
        "textQuery": query,
        "languageCode": cfg.language_code,
        "locationBias": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": _SEARCH_RADIUS_M,
            }
        },
    }
    if cfg.region_code:
        body["regionCode"] = cfg.region_code
    headers = {"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": _FIELD_MASK}
    try:
        response = await http.post(_SEARCH_URL, json=body, headers=headers)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("places_fill: search failed for query=%r: %s", query, exc)
        return []
    data = response.json()
    places = data.get("places") if isinstance(data, dict) else None
    return places if isinstance(places, list) else []


async def fetch_and_build(
    *,
    http: httpx.AsyncClient,
    api_key: str,
    queries: list[str],
    manual: list[ManualEntry],
    cfg: PlacesConfig,
    llm: LLMLike | None,
    model: str,
    now: int,
    frag: Fragments,
    lang: Language,
) -> list[FetchedPlace]:
    """Runs all queries, filters, dedups by ``place.id`` (keeping the first occurrence:
    the config's query order matters more than a repeated find)."""
    seen: dict[str, FetchedPlace] = {}
    for query in queries:
        places = await search_places(http, query=query, api_key=api_key, cfg=cfg)
        for place in places:
            place_id = str(place.get("id") or "")
            if not place_id or place_id in seen:
                continue
            if not passes_filters(place, cfg):
                continue

            name = _display_name(place)
            address = str(place.get("formattedAddress", ""))
            manual_entry = find_manual_override(name, manual)
            if manual_entry is not None:
                logger.info("manual override: %s → %s", manual_entry.name, name)
            review_text = build_review_text(place.get("reviews") or [])
            category = pick_category(
                manual=manual_entry,
                review_text=review_text,
                address=address,
                primary_type=str(place.get("primaryType", "")),
                outskirts_markers=cfg.address_outskirts_markers,
            )
            district = pick_district(address=address, manual=manual_entry, cfg=cfg)
            fact = await compress_fact(
                review_text, llm=llm, model=model, now=now, frag=frag, lang=lang
            )

            seen[place_id] = FetchedPlace(
                place_id=place_id,
                name=name,
                district=district,
                category=category,
                rating=float(place.get("rating", 0.0)),
                reviews=int(place.get("userRatingCount", 0)),
                price_level=parse_price_level(place.get("priceLevel")),
                quiet=bool(manual_entry.quiet) if manual_entry is not None else False,
                fact=fact,
                operational=True,
                manual_override=manual_entry is not None,
            )
    return list(seen.values())


def build_manual_only_rows(
    manual: list[ManualEntry], now: int, *, cfg: PlacesConfig
) -> list[FetchedPlace]:
    """``--manual-only``: no Google. rating/reviews are honestly inflated (see README,
    section "Venues"): this is a deliberate bypass of min_rating/min_reviews in
    ``select_places``, not real data, so every row is logged."""
    rows: list[FetchedPlace] = []
    for index, entry in enumerate(manual):
        logger.info(
            "places_fill: manual-only row (manual data, rating/reviews are not real): %s",
            entry.name,
        )
        rows.append(
            FetchedPlace(
                place_id=f"manual:{index}:{entry.name.casefold().replace(' ', '-')}",
                name=entry.name,
                district=entry.district or cfg.default_district,
                category=entry.category or "craft",
                rating=_MANUAL_ONLY_RATING,
                reviews=_MANUAL_ONLY_REVIEWS,
                price_level=None,
                quiet=entry.quiet,
                fact="",
                operational=True,
            )
        )
    del now  # kept in the signature for symmetry with fetch_and_build/write_rows
    return rows


def render_report(rows: list[FetchedPlace]) -> str:
    lines = ["name | district | category | rating/reviews | quiet | fact | status"]
    for row in rows:
        status = "manual" if row.manual_override else "operational" if row.operational else "closed"
        lines.append(
            f"{row.name} | {row.district} | {row.category} | "
            f"{row.rating:.1f}/{row.reviews} | {'yes' if row.quiet else 'no'} | "
            f"{row.fact or '-'} | {status}"
        )
    lines.append("")
    lines.append(
        f"Total: {len(rows)}. Check the fact column by hand before rollout; Google reviews"
    )
    lines.append("are not stored in the database, only this condensed field.")
    return "\n".join(lines)


async def write_rows(
    db_path: Path,
    rows: list[FetchedPlace],
    now: int,
    *,
    seen_ids: Sequence[str] | None = None,
) -> int:
    """Upsert ``rows`` into ``places``. ``seen_ids`` (not ``None``) means a real run
    against Google (not ``--dry-run``, not ``--manual-only``): after writing, every place_id
    that was in ``places`` but was not seen in this run is marked ``operational=0``
    (``Database.mark_places_not_seen``), except ``--manual-only`` entries (``place_id``
    prefix "manual:"), which Google never finds, so "seen in this run or not" is moot for
    them. Returns the number of marked rows (0 if ``seen_ids`` is not passed).
    """
    # PlaceRow/Database.upsert_place were written in parallel by another agent: the import
    # is deferred until the call, see the module docstring.
    from sotto.db import Database, PlaceRow

    db = Database(db_path)
    await db.connect()
    try:
        for row in rows:
            place_row = PlaceRow(
                place_id=row.place_id,
                name=row.name,
                district=row.district,
                category=row.category,
                rating=row.rating,
                reviews=row.reviews,
                price_level=row.price_level,
                quiet=row.quiet,
                fact=row.fact,
                operational=row.operational,
                refreshed_at=now,
            )
            await db.upsert_place(place_row)
        if seen_ids is None:
            return 0
        return await db.mark_places_not_seen(seen_ids, now)
    finally:
        await db.close()


def _split_queries(raw: str) -> list[str]:
    return [q.strip() for q in raw.split(";") if q.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sotto.places_fill",
        description="Offline fill of the venue cache (PLAN.md, stage 5).",
    )
    parser.add_argument("--dry-run", action="store_true", help="do not write to the database")
    parser.add_argument(
        "--no-llm", action="store_true", help="fact is always empty, the model is not called"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="queries separated by ';' instead of cfg.places.queries",
    )
    parser.add_argument(
        "--manual",
        type=Path,
        default=None,
        help="manual fields (quiet/category/district); default: the pack's places_manual.yaml",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="do not call Google; write only the manual list (no key needed)",
    )
    return parser.parse_args(argv)


async def run(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    http: httpx.AsyncClient | None = None,
    llm: LLMLike | None = None,
    persona: Persona | None = None,
) -> list[FetchedPlace]:
    """Entry point shared by the CLI and tests.

    ``settings``/``http``/``llm`` are injectable dependencies (see the module
    docstring): tests supply a temporary ``Settings.db_path``, an ``httpx.AsyncClient``
    with ``httpx.MockTransport`` and a fake ``LLMLike``.
    ``main()`` calls it without them, in which case real objects are created.
    """
    args = parse_args(argv)
    settings = settings if settings is not None else Settings()
    persona = persona if persona is not None else persona_from_env()
    cfg: Config = load_config(persona.config_layers())
    manual = load_manual(args.manual if args.manual is not None else persona.places_manual_path)
    now = int(time_module.time())

    if args.manual_only:
        rows = build_manual_only_rows(manual, now, cfg=cfg.places)
    else:
        if settings.google_places_key is None:
            raise SystemExit(
                "GOOGLE_PLACES_KEY is not set in .env. Set a Google Places (New) key "
                "or run with --manual-only to fill the cache from the manual "
                "list in places_manual.yaml only."
            )
        api_key = settings.google_places_key.get_secret_value()
        queries = _split_queries(args.queries) if args.queries else cfg.places.queries

        use_llm: LLMLike | None
        owns_llm = False
        if args.no_llm:
            use_llm = None
        elif llm is not None:
            use_llm = llm
        elif settings.openrouter_api_key is not None and cfg.llm.main_model:
            use_llm = LLMClient(
                api_key=settings.openrouter_api_key.get_secret_value(),
                cfg_getter=lambda: cfg,
                db=_InMemoryState(),
            )
            owns_llm = True
        else:
            use_llm = None
            logger.warning(
                "places_fill: LLM unavailable (no OPENROUTER_API_KEY or llm.main_model "
                "is not set); fact will be empty for all places"
            )

        owns_http = http is None
        http_client = http if http is not None else httpx.AsyncClient()
        try:
            rows = await fetch_and_build(
                http=http_client,
                api_key=api_key,
                queries=queries,
                manual=manual,
                cfg=cfg.places,
                llm=use_llm,
                model=cfg.llm.main_model,
                now=now,
                frag=persona.frag,
                lang=persona.lang,
            )
        finally:
            if owns_http:
                await http_client.aclose()
            if owns_llm:
                assert isinstance(use_llm, LLMClient)
                await use_llm.aclose()

    print(render_report(rows))  # CLI script output (table + reminder), not logging

    if not args.dry_run:
        # seen_ids only for a real run against Google: --manual-only does not call Google at
        # all, so "not seen in this run" is meaningless for it and would hit the whole rest
        # of the cache (see write_rows/Database.mark_places_not_seen).
        seen_ids = [row.place_id for row in rows] if not args.manual_only else None
        marked = await write_rows(settings.db_path, rows, now, seen_ids=seen_ids)
        if marked:
            logger.info(
                "places_fill: marked %d place(s) as not operational (not seen this run)", marked
            )

    return rows


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
