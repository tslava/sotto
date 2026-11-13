"""Рантайм-выбор заведений (PLAN.md, этап 5; CLAUDE.md, "Интерфейсы этапа 5").

Чистые синхронные функции без I/O — наполнение кэша (Google Places, LLM-сжатие
отзывов) делает отдельный офлайн-скрипт (``places_fill.py``, другой агент),
этот модуль только читает уже готовые ``db.PlaceRow`` и решает, что подмешать
в промпт по тексту сообщения-триггера:

- ``select_places`` / ``render_places_block`` — старый путь (regex-детект +
  фильтрованный список из 1-2 мест); оставлены для ``replay.py`` и тестов,
  ``responder._generate_and_send`` их больше не зовёт;
- ``render_places_menu`` — рантайм-путь при прямом обращении: весь кэш целиком,
  решение «спрашивали ли про место» и выбор 1-2 подходящих отдаёт модели (решение
  владельца после живого теста — regex не покрывает «где тут пиво приличное»);
- ``validate_fact`` — та же валидация, что ``places_fill.py`` прогоняет по
  сжатому отзыву перед записью в БД. Отзыв — чужой текст из интернета, канал
  непрямой инъекции (PLAN.md, этап 5, п.3): команда в отзыве может проехать
  через LLM-сжатие в fact и оттуда в каждый промпт про места, поэтому фильтр
  строгий (только кириллица, пробелы, запятая, дефис, до 40 символов) и это
  последняя автоматическая защита перед ручным просмотром.
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
    """``lang.fold`` (регистр, «ё»), для сравнения без учёта регистра, принятого в контракте."""
    return lang.fold(text.strip())


def _district_lookup(cfg: PlacesConfig, lang: Language) -> dict[str, str]:
    """Вариант написания (без регистра, ё/е) -> каноническое имя района из ``district_aliases``."""
    lookup: dict[str, str] = {}
    for canonical, variants in cfg.district_aliases.items():
        lookup[_normalize(canonical, lang)] = canonical
        for variant in variants:
            lookup[_normalize(variant, lang)] = canonical
    return lookup


def _district_group(name: str, lookup: dict[str, str], lang: Language) -> str | None:
    """Канонический район для произвольного написания (PlaceRow.district), либо None."""
    return lookup.get(_normalize(name, lang))


def _detect_district(request_text: str, lookup: dict[str, str], lang: Language) -> str | None:
    """Первый упомянутый в тексте район (любой вариант написания), либо None."""
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
    """Фильтрует ``rows`` по тексту запроса и выбирает до ``cfg.max_per_reply``.

    Порядок (CLAUDE.md, "Интерфейсы этапа 5"):

    1. База: ``operational``, ``rating >= cfg.min_rating``, ``reviews >= cfg.min_reviews``.
    2. Если в запросе слова про тишину — жёсткий фильтр ``quiet=1`` (пусто, если
       среди оставшихся нет ни одного тихого).
    3. Если упомянут район — он в приоритете: сужаем до совпадений по району,
       но только если такие совпадения есть, иначе (район в кэше не встретился)
       используем то, что было до этого шага, а не пустой список.
    4. Иначе, если в запросе слова про выезд за город — жёсткий фильтр
       ``category == "outskirts"``.
    5. Иначе — любая категория.

    Из итогового набора — ``rng.sample`` до ``max_per_reply`` (или меньше, если
    подходящих не хватает). Пустой список на любом шаге -> ``[]``.
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
    """Блок ``{places}``: название, район, категория, факт. Пусто -> prompt.PLACES_NONE.

    Часы работы не выводятся никогда (их нет в PlaceRow), источник (Google) не
    упоминается — "был, ничего особенного", а не "по отзывам в Гугле".
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
    """Весь кэш заведений для промпта при прямом обращении (mention/reply/name).

    Решение владельца после живого теста (CLAUDE.md, "Интерфейсы этапа 5"): детект
    запроса про место регулярками не покрывает живую речь («где тут пиво
    приличное», «тихое место есть на Рожицах?»), поэтому при любом прямом
    обращении список уходит в промпт целиком, а решение «спрашивали ли про место»
    принимает модель по инструкции в этом же блоке — не ``select_places``.
    Регулярки (``patterns.places_request``) остаются для статистики и ambient.

    Пусто -> фрагмент ``prompt.places_none``. Иначе строки по ``name``, ``тихо``/``шумно``
    по ``quiet``, ``fact`` — только если непустой. Никаких рейтингов, цен, часов
    работы, слова Google (источник не палится, как и в ``render_places_block``).
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
    """Только кириллица, пробелы, запятая, дефис; <= 40 символов; иначе None.

    Отзыв Google — недоверенный внешний текст (PLAN.md, этап 5, п.3): латиница,
    цифры и любая другая пунктуация валят факт целиком, а не только вырезаются
    из него — частичная очистка оставила бы шанс протащить инъекцию обрывками.
    """
    text = raw.strip()
    if not text or len(text) > _FACT_MAX_LEN:
        return None
    if not lang.fact_re().fullmatch(text):
        return None
    return text
