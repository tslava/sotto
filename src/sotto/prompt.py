"""Сборка сообщений для LLM и разбор её ответа (этап 3).

Чистые функции, без I/O и без aiogram. Ключевое правило — данные отдельно
от инструкций (PLAN.md, этап 3, "Данные отдельно от инструкций"):

- Системный промпт (``template``) уходит ролью ``system``, слоты ``{age}`` и
  ``{few_shot}`` в нём подставляются реальными значениями, а ``{context}``,
  ``{recent_replies}``, ``{places}``, ``{situation}`` — короткими маркерами
  "см. ниже", потому что сами данные уезжают вторым сообщением ролью ``user``,
  внутри разделителей ``<<<CHAT ... >>>``, с явной оговоркой, что это данные,
  а не команды.
- Подстановка слотов — один проход ``re.sub`` по известным именам слотов,
  никогда ``str.format`` и никогда цепочка ``str.replace``: фигурная скобка в
  сообщении участника («{context}», «{'a': 1}») либо уронила бы вызов
  (``str.format``), либо, оказавшись внутри уже подставленного слота (например
  «{context}» внутри few_shot), сама попала бы под следующую замену в цепочке
  ``.replace()`` — один проход ``re.sub`` сканирует только исходный template и
  такого не делает.
- Разделители ``<<<``/``>>>`` вырезаются из данных ещё раз здесь, дублируя
  ``sanitize.normalize_text`` — по счастливой случайности normalize_text уже
  чистит текст отдельных сообщений, но собранные блоки (context, места,
  ситуация) дополнительно проходят ту же чистку на всякий случай.

``parse_reply`` разбирает ответ модели так же строго и терпимо к обёрткам
(```json ... ```` и преамбулам вроде "Вот ответ: {...}"), но любой сбой —
это ``None``, то есть молчание, а не исключение.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from sotto.db import LifeEventRow, MessageRow
from sotto.fragments import Fragments
from sotto.timeutil import local_date

CHAT_OPEN = "<<<CHAT"
CHAT_CLOSE = ">>>"

# Тексты ситуаций, каркаса сообщения и заголовков блоков живут в пакете персоны
# (``fragments.yaml``, группы situations/prompt/render/placeholders) и приходят
# аргументом ``frag``; здесь только сборка.

_ADDRESSED_TEXT_MAX_LEN = 300
_ADDRESSED_ITEMS_MAX = 5

_LIFE_TEXT_MAX_LEN = 300
_SITUATION_MARKER = ""

# Дублирует вырезание разделителей из sanitize.normalize_text (см. докстринг модуля):
# собранные блоки (context/recent_replies/places/situation) чистятся ещё раз здесь,
# перед тем как сами стать обёрнутыми в <<<CHAT ... >>>.
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

_SLOT_RE = re.compile(
    r"\{(age|few_shot|life|diary|vocab|chat_memory|jokes|weather|context|recent_replies|places|avoid"
    r"|situation)\}"
)


def _strip_fake_delimiters(text: str) -> str:
    return _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))


def _clean_addressed_text(text: str) -> str:
    return _strip_fake_delimiters(text).strip()[:_ADDRESSED_TEXT_MAX_LEN]


def situation_addressed(items: list[tuple[str, str]], *, frag: Fragments) -> str:
    """Ситуация «к тебе обратился(-ись)» из накопленных обращений (display_name, text).

    ``items`` — все обращения, накопившиеся к моменту срабатывания отложенного
    ответа (схлопывание дебаунс-буфера может собрать несколько обращений от
    разных людей за время задержки), в хронологическом порядке. Только
    последние ``_ADDRESSED_ITEMS_MAX`` используются — вызывающий (responder.py)
    уже должен был обрезать список сам, но обрезаем и здесь на всякий случай.

    Одно обращение — короткая форма (``situations.addressed_single``, подстановка
    через .replace, не re.sub/format: тут только два слота и они не могут
    провоцировать повторную подстановку друг друга). Несколько — маркированный
    список с общей инструкцией (``situations.addressed_multi_*``).

    Имя и текст каждого обращения — недоверенный ввод участника: разделители
    ``<<<``/``>>>`` вырезаются тем же способом, что и в build_messages, а текст
    обрезается до 300 символов, чтобы не раздувать промпт длинной репликой.
    Пустой ``items`` -> пустая строка (значит, ситуацию добавлять не нужно).
    """
    trimmed = items[-_ADDRESSED_ITEMS_MAX:]
    cleaned = [
        (_strip_fake_delimiters(name).strip(), _clean_addressed_text(text))
        for name, text in trimmed
    ]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        name, text = cleaned[0]
        single: str = frag.text("situations.addressed_single")
        return single.replace("{name}", name).replace("{text}", text)

    quote_open, quote_close = frag.text("render.quote_open"), frag.text("render.quote_close")
    lines = [frag.text("situations.addressed_multi_header")]
    lines.extend(f"- {name}: {quote_open}{text}{quote_close}" for name, text in cleaned)
    lines.append(frag.text("situations.addressed_multi_footer"))
    return "\n".join(lines)


def situation_followup(items: list[tuple[str, str]], *, frag: Fragments) -> str:
    """Ситуация для триггера ``followup`` (CLAUDE.md, "внимание как у живого
    человека") — та же форма, что ``situation_addressed`` (может накопиться
    несколько поводов за время дебаунс-схлопывания одного pending), но с явной
    оговоркой, что адресность не точная, а вероятная: обычный гейт такое
    сообщение обращением не признал, это решение дешёвой проверки.

    Потолок ``_ADDRESSED_ITEMS_MAX``, обрезка текста до ``_ADDRESSED_TEXT_MAX_LEN``,
    вырезание поддельных разделителей — как у ``situation_addressed``.
    """
    trimmed = items[-_ADDRESSED_ITEMS_MAX:]
    cleaned = [
        (_strip_fake_delimiters(name).strip(), _clean_addressed_text(text))
        for name, text in trimmed
    ]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        name, text = cleaned[0]
        single: str = frag.text("situations.followup_single")
        return single.replace("{name}", name).replace("{text}", text)

    quote_open, quote_close = frag.text("render.quote_open"), frag.text("render.quote_close")
    lines = [frag.text("situations.followup_multi_header")]
    lines.extend(f"- {name}: {quote_open}{text}{quote_close}" for name, text in cleaned)
    lines.append(frag.text("situations.followup_multi_footer"))
    return "\n".join(lines)


def render_numbered_messages(rows: Sequence[MessageRow]) -> str:
    """ "1. Имя: текст" по строке, нумерация с 1, имя и текст чистятся от
    поддельных разделителей и обрезаются тем же лимитом, что и
    ``situation_addressed`` (``_ADDRESSED_TEXT_MAX_LEN`` символов).

    Общий хелпер для ``situation_checkin`` и дешёвого предфильтра "вернулся
    проверить" (``followup.FollowupChecker.check_batch``, CLAUDE.md,
    "Интерфейсы: дешёвый предфильтр для «вернулся проверить»") — оба показывают
    модели одну и ту же пронумерованную пачку сообщений, форматирование не
    дублируется. Пустой ``rows`` -> пустая строка.
    """
    lines = []
    for index, row in enumerate(rows, start=1):
        name = _strip_fake_delimiters(row.display_name or "").strip()
        text = _clean_addressed_text(row.text or "")
        lines.append(f"{index}. {name}: {text}")
    return "\n".join(lines)


def situation_checkin(rows: Sequence[MessageRow], *, frag: Fragments) -> str:
    """Ситуация для триггера ``checkin`` (CLAUDE.md, "внимание как у живого
    человека: вернулся проверить") — пронумерованный список сообщений,
    накопившихся после последней реплики персонажа, плюс инструкция ответить
    одной фразой на выбранное (номер уходит в поле ``reply_to`` JSON-ответа)
    либо промолчать. Пустой ``rows`` -> пустая строка — вызывающий
    (responder.py) не должен звать генерацию без сообщений вовсе, но на всякий
    случай это тоже "ничего не отвечать".
    """
    if not rows:
        return ""
    lines = [
        frag.text("situations.checkin_header"),
        render_numbered_messages(rows),
        frag.text("situations.checkin_footer"),
    ]
    return "\n".join(lines)


def render_context(rows: list[MessageRow]) -> str:
    """ "Имя: текст" по строке, в хронологическом порядке (rows уже отсортированы)."""
    return "\n".join(f"{row.display_name}: {row.text}" for row in rows)


def render_life(rows: Sequence[LifeEventRow], tz: str, *, frag: Fragments) -> str:
    """Блок памяти о событиях жизни персонажа (CLAUDE.md, "события жизни").

    Пусто -> "" (тогда весь абзац со слотом {life} в system.txt пропадает).
    Иначе — заголовок и по строке на событие, дата локальная (timeutil.local_date,
    tz), формат dd.mm.yyyy: "12.09.2026: продал тележку, взял Hyundai".

    Строки НЕ начинаются с "- " и не содержат слова JSON: filters.regex:prompt_leak
    считает инструктивной частью системного промпта именно строки-буллеты «- …»
    и блок про JSON, а пересказ события персонажем утечкой не является.

    Событие подставляется прямо в system (как few_shot, не как context/places) —
    это память владельца о персонаже, а не недоверенный ввод участников чата, но
    _strip_fake_delimiters всё равно применяется на всякий случай, чтобы поддельные
    ``<<<``/``>>>`` внутри заметки не путали разметку user-сообщения.
    """
    if not rows:
        return ""
    lines = [frag.text("render.life_header")]
    for row in rows:
        day = local_date(row.created_at, tz).strftime("%d.%m.%Y")
        lines.append(f"{day}: {_strip_fake_delimiters(row.text)}")
    return "\n".join(lines)


def situation_life(text: str, *, frag: Fragments) -> str:
    """Ситуация для announce_life: подстановка в ``situations.life``.

    Один слот {text} — через .replace, не re.sub/format (как в situation_addressed):
    единственная подстановка не может спровоцировать повторную замену самой себя.
    Текст обрезается до 300 символов и чистится от поддельных разделителей — тот
    же приём, что и для обращений (_clean_addressed_text), но со своей константой
    длины, потому что источник другой (заметка владельца, не сообщение участника).
    """
    cleaned = _strip_fake_delimiters(text).strip()[:_LIFE_TEXT_MAX_LEN]
    life: str = frag.text("situations.life")
    return life.replace("{text}", cleaned)


def situation_callback(name: str, text: str, *, frag: Fragments) -> str:
    """Ситуация для триггера ``callback``: подстановка в ``situations.callback``.

    Имя и текст чистятся от поддельных разделителей и режутся до 300 символов
    (``_clean_addressed_text``); подстановка за один проход ``re.sub``, чтобы
    ``{name}``/``{text}`` внутри чужих данных повторно не заменялись.
    """
    values = {"name": _clean_addressed_text(name), "text": _clean_addressed_text(text)}
    template: str = frag.text("situations.callback")
    return re.sub(r"\{(name|text)\}", lambda m: values[m.group(1)], template)


def build_messages(
    template: str,
    *,
    age: int,
    few_shot: str,
    context: str,
    recent_replies: str,
    places: str,
    situation: str,
    avoid: str = "",
    life: str = "",
    diary: str = "",
    chat_memory: str = "",
    jokes: str = "",
    weather: str = "",
    vocab: str = "",
    json_reminder: str | None = None,
    frag: Fragments,
) -> list[dict[str, str]]:
    """Собирает [system, user] для LLMClient.call().

    Слоты подставляются через один проход re.sub (не str.format и не цепочку
    str.replace): цепочка последовательных .replace() сканирует уже подставленный
    текст заново на каждом шаге, поэтому "{context}", случайно оказавшийся внутри
    few_shot (или любого другого уже подставленного слота), тоже заменился бы на
    следующем шаге. Один проход re.sub сканирует только исходный template — то,
    что подставлено, повторно не трогается. Данные людей
    (context/recent_replies/places/situation) в system не попадают — только
    маркеры "см. ниже"; сами данные уходят вторым сообщением role=user,
    context и recent_replies — внутри разделителей <<<CHAT ... >>>. {life} —
    исключение: это память владельца о персонаже (как few_shot), а не ввод
    участников чата, поэтому подставляется в system напрямую, реальным значением.
    {diary} (CLAUDE.md, "дневник дня") — тоже в system: факты о собственных делах
    персонажа, достанные из его же реплик (diary.render_diary), а не ввод участников.
    {chat_memory} — такое же исключение: это уже сжатый моделью пересказ прошедших
    недель (chat_memory.py), память персонажа, а не сырые сообщения участников.
    {jokes} (CLAUDE.md, "шутки чата и истории людей") — тоже в system: общие шутки
    чата, достанные недельным ChatMemorizer (jokes.render_jokes), память персонажа.
    {avoid} (CLAUDE.md, "меньше и разнообразнее", мера 5) — тоже в system: это не
    данные участников, а инструкция, собранная motifs.render_avoid из того, что
    персонаж сам уже наговорил («жену и гараж ты уже поминал, сейчас без них»).
    {weather} (CLAUDE.md, "Интерфейсы: погода") — тоже в system: это факт из
    открытого API (weather.render_weather), фон жизни персонажа, а не сообщение
    участника; название места внутри блока приходит от геокодера, не из чата.
    {vocab} (CLAUDE.md, "словечки из словаря") — тоже в system: словари грузит
    владелец (vocab.render_vocab), это его данные, как {life}, а не ввод участников.
    """
    slot_values = {
        "age": str(age),
        "few_shot": few_shot,
        "life": life,
        "diary": _strip_fake_delimiters(diary).strip(),
        "chat_memory": chat_memory,
        "jokes": _strip_fake_delimiters(jokes).strip(),
        "weather": _strip_fake_delimiters(weather).strip(),
        "vocab": _strip_fake_delimiters(vocab).strip(),
        "context": frag.text("prompt.context_marker"),
        "recent_replies": frag.text("prompt.recent_replies_marker"),
        "places": frag.text("prompt.places_marker"),
        "avoid": _strip_fake_delimiters(avoid).strip(),
        "situation": _SITUATION_MARKER,
    }
    system = _SLOT_RE.sub(lambda m: slot_values[m.group(1)], template)

    clean_context = _strip_fake_delimiters(context)
    clean_recent = _strip_fake_delimiters(recent_replies).strip() or frag.text(
        "placeholders.recent_replies_empty"
    )
    clean_places = _strip_fake_delimiters(places).strip() or frag.text("prompt.places_none")
    clean_situation = _strip_fake_delimiters(situation).strip()

    lines = [
        frag.text("prompt.data_notice"),
        f"{CHAT_OPEN}\n{clean_context}\n{CHAT_CLOSE}",
        "",
        frag.text("prompt.recent_replies_label"),
        f"{CHAT_OPEN}\n{clean_recent}\n{CHAT_CLOSE}",
        "",
        clean_places,
    ]
    if clean_situation:
        lines.append("")
        lines.append(clean_situation)
    lines.append("")
    lines.append(json_reminder if json_reminder is not None else frag.text("prompt.json_reminder"))

    user = "\n".join(lines)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass(frozen=True, slots=True)
class Reply:
    speak: bool
    text: str
    # Номер сообщения (1-based, индекс в checkin_rows), на которое отвечает модель —
    # только для триггера "checkin" (CLAUDE.md, "вернулся проверить"); для остальных
    # триггеров модель это поле не заполняет, и оно остаётся None.
    reply_to: int | None = None


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


def parse_reply(raw: str) -> Reply | None:
    """Разбирает ответ модели в Reply, либо None при любом сбое (значит — молчание).

    Срезает ```json ... ``` / ``` ... ``` обёртки; ищет первую "{" и разбирает
    JSON от неё через ``json.JSONDecoder().raw_decode`` вместо ``json.loads`` —
    raw_decode останавливается сразу после первого валидного объекта и не требует,
    чтобы после него ничего не было. Это терпимо к хвосту, который модели любят
    дописывать после JSON (например "\\n\\nПояснение: не уверен" или второй JSON-объект)
    — json.loads на таком хвосте упал бы, raw_decode его просто игнорирует.
    speak обязан быть bool, text — строкой; при speak=true пустой text — сбой
    (None), при speak=false отсутствующий/пустой text — это норма ("").
    Лишние ключи в JSON игнорируются.
    """
    try:
        candidate = _strip_code_fence(raw.strip())
        start = candidate.find("{")
        if start == -1:
            return None
        data, _end = json.JSONDecoder().raw_decode(candidate, start)
        if not isinstance(data, dict):
            return None

        speak = data.get("speak")
        if not isinstance(speak, bool):
            return None

        text_value = data.get("text", "")
        if text_value is None:
            text_value = ""
        if not isinstance(text_value, str):
            return None
        text = text_value.strip()

        # "reply_to": int, null или отсутствует -> используется как есть/None;
        # любой другой тип (строка, float, bool — bool исключён явно, т.к.
        # isinstance(True, int) в Python истинно) -> None, это не срыв всего
        # разбора, только этого необязательного поля (CLAUDE.md, "вернулся проверить").
        reply_to_value = data.get("reply_to")
        reply_to: int | None
        if reply_to_value is None or isinstance(reply_to_value, bool):
            reply_to = None
        elif isinstance(reply_to_value, int):
            reply_to = reply_to_value
        else:
            reply_to = None

        if speak:
            if not text:
                return None
            return Reply(speak=True, text=text, reply_to=reply_to)
        return Reply(speak=False, text="", reply_to=reply_to)
    except Exception:
        # Ответ модели — недоверенный внешний текст: любой сбой разбора (не JSON,
        # оборванная обёртка, неожиданный тип) означает "молчание", а не падение.
        return None
