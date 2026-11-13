"""Weather as background colour for the persona's life (CLAUDE.md, "Interfaces: weather" and
"Interfaces: weather somewhere else").

The persona used to answer a weather question with a shrug; now it has real data. The source
is Open-Meteo: a public API with no key and no registration.

Weather is background, not a lookup service: one or two lines go into the system prompt and
the model decides whether they are relevant. The home point (coordinates and name) lives in
``.env``, not in the config or the code: the repository is public and where the owner lives
is personal data. No coordinates -> ``get()`` without a place returns ``None``, the weather
block is empty and everything else works as before.

When a message asks about another place ("and in Warsaw?"), a cheap model extracts the name
from the text (``WeatherPlaceExtractor``, the same approach as ``followup.FollowupChecker``:
one call, only when there is a reason), the Open-Meteo geocoder supplies coordinates, and a
second line is added to the prompt.

The network is used only in this module and its errors never escape: any failure is a
``logger.warning`` plus the previous (even stale) cache, or ``None``. Saying nothing about
the weather is cheaper than a broken reply.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sotto.config_models import Config, WeatherConfig
from sotto.fragments import Fragments
from sotto.llm import LLMClient, LLMError
from sotto.timeutil import local_date

logger = logging.getLogger(__name__)

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Own daily call counter (like vision_calls/followup_calls): a cheap call must not eat
# the main model's call cap.
_COUNTER_KEY = "weather_calls"

# A longer name is no longer a place but a paraphrase of the message: the model misread
# the task and the geocoder would not find it anyway.
_PLACE_MAX_LEN = 60

_SLOT_RE = re.compile(r"\{(text)\}")
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Weather:
    temp_now: float
    code_now: int
    today_min: float
    today_max: float
    today_code: int
    tomorrow_min: float
    tomorrow_max: float
    tomorrow_code: int
    fetched_at: int


@dataclass(frozen=True, slots=True)
class Place:
    """A point on the map. ``name`` is how the prompt refers to it: for the home point it is
    ``Settings.weather_home_name`` (may be empty), for a geocoded place whatever the
    geocoder returned."""

    name: str
    latitude: float
    longitude: float


def describe(code: int, *, frag: Fragments) -> str:
    """WMO code -> description from the package's ``weather.wmo`` table; unknown code -> ""
    (a line without a description).

    The table holds exactly the codes Open-Meteo actually returns; YAML keys are integers."""
    return str(frag.table("weather.wmo").get(code, ""))


def format_temp(value: float) -> str:
    """Temperature as a signed integer: "+9", "-3", "0" (zero has no sign).

    Public: the weather line in ``/status`` (commands.py) uses the same form, so "+9" looks
    the same in the prompt and in the status."""
    rounded = round(value)
    if rounded == 0:
        return "0"
    return f"{rounded:+d}"


def _same_local_day(fetched_at: int, tz: str, now: int) -> bool:
    """The snapshot was taken on the same local day as ``now``.

    Keeps "today" and "tomorrow" honest: after a long network outage the client serves the
    previous cache (yesterday's weather beats none), but if it was taken yesterday its
    "tomorrow" is already today and the line would lie. Such a snapshot is not rendered at
    all, so the prompt simply has no weather paragraph.
    """
    return local_date(fetched_at, tz) == local_date(now, tz)


def _with_desc(prefix: str, code: int, *, joiner: str, frag: Fragments) -> str:
    text = describe(code, frag=frag)
    if not text:
        return prefix
    return f"{prefix}{joiner}{text}"


def render_weather(
    w: Weather | None, tz: str, now: int, *, home_name: str = "", frag: Fragments
) -> str:
    """Home-weather block for the system prompt. ``None`` -> "" (the paragraph disappears).

    The home point's name is NOT put into the text: Russian names would need declension
    (a wrong case form of the city) and the model already knows from the prompt where the
    persona lives. ``home_name`` is only needed to skip the second block when someone asks
    about the persona's own town (``responder``). Lines do NOT start with "- ":
    ``filters.regex:prompt_leak`` treats bullet lines as the instructional part of the
    prompt, and a bullet weather line would cut the persona's reply as a leak.
    There are no times or dates in the line, so the model does not recite "data as of 12:30".
    """
    if w is None or not _same_local_day(w.fetched_at, tz, now):
        return ""
    now_part = _with_desc(
        f"{frag.get('weather.now_word')}{format_temp(w.temp_now)}",
        w.code_now,
        joiner=frag.text("weather.and_word"),
        frag=frag,
    )
    range_to = frag.text("weather.range_to")
    first = (
        f"{frag.get('weather.home_prefix')}"
        f"{now_part}{frag.get('weather.today_from')}{format_temp(w.today_min)}"
        f"{range_to}{format_temp(w.today_max)}."
    )
    second = _with_desc(
        f"{frag.get('weather.tomorrow_from')}{format_temp(w.tomorrow_min)}"
        f"{range_to}{format_temp(w.tomorrow_max)}",
        w.tomorrow_code,
        joiner=", ",
        frag=frag,
    )
    return f"{first}\n{second}."


def render_weather_place(
    place_name: str, w: Weather | None, tz: str, now: int, *, frag: Fragments
) -> str:
    """Second line of the block: weather at the place that was asked about. ``None`` -> ""."""
    if w is None or not _same_local_day(w.fetched_at, tz, now):
        return ""
    now_part = _with_desc(
        f"{frag.get('weather.now_word')}{format_temp(w.temp_now)}",
        w.code_now,
        joiner=frag.text("weather.and_word"),
        frag=frag,
    )
    range_to = frag.text("weather.range_to")
    tomorrow_part = _with_desc(
        f"{frag.get('weather.tomorrow_from_lower')}{format_temp(w.tomorrow_min)}"
        f"{range_to}{format_temp(w.tomorrow_max)}",
        w.tomorrow_code,
        joiner=", ",
        frag=frag,
    )
    # The name is in the nominative case: "Asking about Warszawa" would be wrong in Russian
    # and geocoder names cannot be declined, hence the colon after "place".
    return (
        f"{frag.get('weather.place_prefix')}{place_name}. {now_part.capitalize()}, {tomorrow_part}."
    )


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


def _parse_weather(data: Any, now: int) -> Weather:
    """Open-Meteo response -> Weather. Any mismatch with the format is an exception
    (KeyError/IndexError/TypeError/ValueError) that the caller catches."""
    current = data["current"]
    daily = data["daily"]
    mins = daily["temperature_2m_min"]
    maxs = daily["temperature_2m_max"]
    codes = daily["weather_code"]
    return Weather(
        temp_now=float(current["temperature_2m"]),
        code_now=int(current["weather_code"]),
        today_min=float(mins[0]),
        today_max=float(maxs[0]),
        today_code=int(codes[0]),
        tomorrow_min=float(mins[1]),
        tomorrow_max=float(maxs[1]),
        tomorrow_code=int(codes[1]),
        fetched_at=now,
    )


def _parse_place(data: Any, cfg: WeatherConfig) -> Place:
    """Geocoder response -> Place. The country is appended to the name unless it is the home
    country (``home_country_codes``/``home_country_names``): the home country is understood
    anyway, while "Cologne, Germany" is another story."""
    results = data["results"]
    first = results[0]
    name = str(first["name"]).strip()
    country = str(first.get("country") or "").strip()
    country_code = str(first.get("country_code") or "").strip()
    home_codes = {code.upper() for code in cfg.home_country_codes}
    home_names = {item.casefold() for item in cfg.home_country_names}
    if (
        name
        and country
        and country_code.upper() not in home_codes
        and country.casefold() not in home_names
    ):
        name = f"{name}, {country}"
    return Place(name=name, latitude=float(first["latitude"]), longitude=float(first["longitude"]))


@dataclass(frozen=True, slots=True)
class _GeocodeEntry:
    """Geocoding result in the cache. ``place`` = None means "searched, not found": a negative
    answer is cached too, so the geocoder is not hammered with the same nonexistent place."""

    place: Place | None
    fetched_at: int


class WeatherClient:
    """Open-Meteo public API with an in-memory cache. The only place in the project that
    goes to the network for weather.

    ``home`` is the home point from ``.env`` (app.py builds it from
    ``WEATHER_LATITUDE``/``WEATHER_LONGITUDE``/``WEATHER_HOME_NAME``). ``None`` means
    "no home point": ``get()`` without an argument returns ``None``, while weather for a
    place that was asked about still works.
    """

    def __init__(
        self,
        cfg_getter: Callable[[], Config],
        home: Place | None = None,
        http: httpx.AsyncClient | None = None,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self._cfg_getter = cfg_getter
        self._home = home
        self._http = (
            http
            if http is not None
            else httpx.AsyncClient(timeout=cfg_getter().behaviour.weather.timeout_sec)
        )
        self._clock = clock
        # Cache keyed by coordinates rounded to hundredths: the home point and every
        # place asked about live separately and do not overwrite each other.
        self._cache: dict[tuple[float, float], Weather] = {}
        self._geocode_cache: dict[str, _GeocodeEntry] = {}
        # Two separate locks: weather and geocoding are called in sequence (geocode first,
        # then get), so there is no need to take one lock twice, and nesting them would be
        # easy to break with a future edit.
        self._lock = asyncio.Lock()
        self._geocode_lock = asyncio.Lock()

    @property
    def home_name(self) -> str:
        """What the home point is called in the prompt; "" means no name or no point."""
        return self._home.name if self._home is not None else ""

    @property
    def home(self) -> Place | None:
        return self._home

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get(self, place: Place | None = None) -> Weather | None:
        """Weather at ``place`` (or at the home point). A fresh cache means no network.

        Network/parse error -> the previous cache of the same point, even stale: yesterday's
        weather beats none (``render_weather`` keeps "today/tomorrow" honest and does not
        render a yesterday snapshot).
        """
        cfg = self._cfg_getter()
        weather_cfg = cfg.behaviour.weather
        if not weather_cfg.enabled:
            return None

        target = place if place is not None else self._home
        if target is None:
            return None
        key = (round(target.latitude, 2), round(target.longitude, 2))

        cached = self._cache.get(key)
        if cached is not None and self._clock() - cached.fetched_at < weather_cfg.ttl_min * 60:
            return cached

        async with self._lock:
            # Re-check under the lock: while waiting for our turn a parallel call may
            # already have gone to the network, so a second request is unnecessary.
            cached = self._cache.get(key)
            now = self._clock()
            if cached is not None and now - cached.fetched_at < weather_cfg.ttl_min * 60:
                return cached

            fetched = await self._fetch(cfg, target, now)
            if fetched is None:
                return self._cache.get(key)
            self._cache[key] = fetched
            return fetched

    async def _fetch(self, cfg: Config, place: Place, now: int) -> Weather | None:
        weather_cfg = cfg.behaviour.weather
        params: dict[str, str | float | int] = {
            "latitude": place.latitude,
            "longitude": place.longitude,
            "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_min,temperature_2m_max,weather_code",
            "timezone": cfg.persona.timezone,
            "forecast_days": 2,
        }
        try:
            response = await self._http.get(
                _FORECAST_URL, params=params, timeout=weather_cfg.timeout_sec
            )
        except httpx.HTTPError as exc:
            # TimeoutException is an HTTPError too, no separate branch needed.
            logger.warning("weather request failed: error=%s", type(exc).__name__)
            return None

        if not (200 <= response.status_code < 300):
            logger.warning("weather http error: status=%s", response.status_code)
            return None

        try:
            return _parse_weather(response.json(), now)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("weather parse failed: error=%s", type(exc).__name__)
            return None

    async def geocode(self, query: str) -> Place | None:
        """Place name -> coordinates. Cached by the query string, TTL
        ``geocode_ttl_days``; a negative result is cached too."""
        cfg = self._cfg_getter()
        weather_cfg = cfg.behaviour.weather
        key = query.strip().lower()
        if not key:
            return None
        ttl_sec = weather_cfg.geocode_ttl_days * 86400

        entry = self._geocode_cache.get(key)
        if entry is not None and self._clock() - entry.fetched_at < ttl_sec:
            return entry.place

        async with self._geocode_lock:
            entry = self._geocode_cache.get(key)
            now = self._clock()
            if entry is not None and now - entry.fetched_at < ttl_sec:
                return entry.place

            place = await self._geocode_request(query.strip(), weather_cfg)
            self._geocode_cache[key] = _GeocodeEntry(place=place, fetched_at=now)
            return place

    async def _geocode_request(self, query: str, weather_cfg: WeatherConfig) -> Place | None:
        params: dict[str, str | int] = {
            "name": query,
            "count": 1,
            "language": weather_cfg.geocode_language,
            "format": "json",
        }
        try:
            response = await self._http.get(
                _GEOCODE_URL, params=params, timeout=weather_cfg.timeout_sec
            )
        except httpx.HTTPError as exc:
            logger.warning("geocode request failed: error=%s", type(exc).__name__)
            return None

        if not (200 <= response.status_code < 300):
            logger.warning("geocode http error: status=%s", response.status_code)
            return None

        try:
            return _parse_place(response.json(), weather_cfg)
        except (KeyError, IndexError, TypeError, ValueError):
            # Nothing found: the response simply has no "results" key. That is
            # routine (a misspelled or obscure place), not a service error.
            logger.warning("geocode: place not found")
            return None


class WeatherPlaceExtractor:
    """Extracts the place name from a weather question with one cheap call.

    The geocoder does not understand Russian declensions ("in Warsaw" in the locative) and
    regexes cannot undo them, so the model restores the nominative case. The call is made
    only when the trigger regex (``patterns.weather_request``) matched and only for direct
    addressing: see ``responder._generate_and_send``.

    Any failure (empty model, ``LLMError``, not JSON, too long an answer) gives ``None``:
    then only the home weather goes into the prompt.
    """

    def __init__(
        self, llm: LLMClient, cfg_getter: Callable[[], Config], prompt_template: str
    ) -> None:
        self._llm = llm
        self._cfg_getter = cfg_getter
        self._prompt_template = prompt_template

    async def extract(self, text: str, *, now: int) -> str | None:
        cfg = self._cfg_getter()
        weather_cfg = cfg.behaviour.weather
        model = weather_cfg.lookup_model or cfg.llm.judge_model
        if not model:
            return None

        slot_values = {"text": _strip_fake_delimiters(text).strip()}
        system = _SLOT_RE.sub(lambda m: slot_values[m.group(1)], self._prompt_template)

        try:
            result = await self._llm.call(
                [{"role": "system", "content": system}],
                model=model,
                max_tokens=weather_cfg.lookup_max_tokens,
                now=now,
                counter_key=_COUNTER_KEY,
                calls_cap=weather_cfg.lookup_daily_cap,
            )
        except LLMError as exc:
            logger.warning("weather place llm error: reason=%s", exc.reason)
            return None

        place = _parse_place_name(result.text)
        if place is None:
            logger.info("weather place: no place in the message")
            return None
        logger.info("weather place: %s", place)
        return place


def _parse_place_name(raw: str) -> str | None:
    """``{"place": "Warsaw"|null}`` -> the name or None.

    Parsing is as strict and tolerant as ``judge._parse_verdict``: strip the ```json```
    fence, find the first "{" and ``raw_decode`` from it. An empty string, a non-string
    and a too long answer are also None (the model misread the task).
    """
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None

        value = data.get("place")
        if not isinstance(value, str):
            return None
        place = _strip_fake_delimiters(value).strip()
        if not place or len(place) > _PLACE_MAX_LEN:
            return None
        return place
    except Exception:
        # The model's answer is untrusted external text: any parse failure means
        # "no place", not a crash of the whole reply.
        return None
