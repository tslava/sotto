"""Router с командами управления ботом из телеграма (PLAN.md, этап 6).

`build_commands_router` возвращает Router с одним хендлером на все сообщения,
начинающиеся с "/". Интегратор (app.py) обязан включить этот роутер в Dispatcher
ПЕРВЫМ, перед основным роутером сообщений (sotto.bot.build_router) — иначе
команды в разрешённом чате попадут в гейт и запишутся в messages как обычный текст.
aiogram останавливает propagation на первом хендлере, вернувшем не-None/не
skip_this_handler, так что второй раз то же сообщение до основного роутера не дойдёт.

Хендлер сам решает, кому что можно — гейт "чужой чат" из bot.py сюда не
распространяется, роутер команд стоит раньше него.

stores.py (ConfigStore/PromptStore) пишет параллельно другой агент по контракту
CLAUDE.md ("Интерфейсы этапа 6"). Чтобы не зависеть от его файла до готовности,
здесь определены структурные Protocol'ы (`_ConfigStoreLike`, `_PromptStoreLike`,
`_DbLike` и вспомогательные row-протоколы) с ровно теми методами, которые нужны
командам. Реальные классы (ConfigStore/PromptStore/Database) подойдут под них
структурно, без явного наследования.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, BinaryIO, Protocol

from aiogram import F, Router
from aiogram.types import Message, User

from sotto import __version__
from sotto.changelog import Release, chat_releases, owner_releases
from sotto.chat_memory import format_period
from sotto.config import KeyInfo
from sotto.config_models import Config
from sotto.db import (
    ChatMemoryRow,
    JokeRow,
    LifeEventRow,
    PeopleThreadRow,
    SelfFactRow,
)
from sotto.errors import OwnerError
from sotto.few_shot import FewShot
from sotto.fragments import Fragments
from sotto.i18n import Catalog, engine_catalog
from sotto.motifs import render_self_share, self_share
from sotto.patterns import Patterns
from sotto.persona import Persona
from sotto.sanitize import normalize_text, sanitize_display_name
from sotto.settings import Settings
from sotto.timeutil import day_key, day_start, local_dt
from sotto.weather import Place, Weather, describe, format_temp

if TYPE_CHECKING:
    # Только для аннотаций: "from __future__ import annotations" делает их строками,
    # так что этот импорт не исполняется в рантайме. Прямой (не TYPE_CHECKING) импорт
    # sotto.responder сюда нежелателен — тянет aiogram-цепочку в commands.py и
    # рискует циклическим импортом (bot.py и так импортирует и responder, и commands).
    from sotto.db import VocabDictRow
    from sotto.responder import PreviewOutcome, SendOutcome
    from sotto.vocab import Vocab, VocabDict

logger = logging.getLogger(__name__)

# Ограничение длины ответа владельцу (PLAN.md, этап 6: "не длиннее 3500 символов").
_MAX_REPLY_LEN = 3500
# Превью тела промпта в /prompt без "full" (CLAUDE.md, "Интерфейсы этапа 6").
_PROMPT_PREVIEW_LEN = 1500
# Сколько пересказов показывать в /memory list (ответ всё равно режется до 3500 символов).
_MEMORY_LIST_LIMIT = 20
# Сколько символов свежесозданных пересказов показать в ответе на /memory run.
_MEMORY_RUN_PREVIEW_LEN = 1500

# /status: окно метрики «про себя» в сутках (CLAUDE.md, "про себя через раз").
_SELF_SHARE_DAYS = 3

# /diary: окно списка (CLAUDE.md, "дневник дня": «за неделю»).
_DIARY_LIST_DAYS = 7

# /jokes и /people (CLAUDE.md, "шутки чата и истории людей"): потолок списка шуток
# (ответ всё равно режется до 3500 символов).
_JOKES_LIST_LIMIT = 1000

# /words (CLAUDE.md, "словечки из словаря"): потолок файла словаря.
_WORDS_MAX_FILE_BYTES = 512 * 1024


class _MessageRowLike(Protocol):
    """Подмножество db.MessageRow, нужное /ex add.

    Члены объявлены через ``@property`` (а не как обычные атрибуты), потому что
    db.MessageRow — frozen dataclass: его поля read-only, а обычный атрибут в
    Protocol требует settable-совместимости (инвариантно в обе стороны) и не
    матчится на read-only поле. ``@property`` в Protocol — это ровно read-only
    член, который frozen dataclass удовлетворяет структурно.
    """

    @property
    def display_name(self) -> str | None: ...
    @property
    def text(self) -> str | None: ...


class _BotReplyRowLike(Protocol):
    """Подмножество db.BotReplyRow, нужное /ex add и /last. См. _MessageRowLike про @property."""

    @property
    def trigger(self) -> str: ...
    @property
    def trigger_tg_message_id(self) -> int | None: ...
    @property
    def text(self) -> str: ...
    @property
    def prompt_version(self) -> int: ...
    @property
    def few_shot_version(self) -> int: ...
    @property
    def delay_sec(self) -> int: ...
    @property
    def created_at(self) -> int: ...


class _VersionRowLike(Protocol):
    """Подмножество db.VersionRow, нужное /prompt (посчитать "всего M"). См. _MessageRowLike."""

    @property
    def version(self) -> int: ...


class _DbLike(Protocol):
    """Методы Database, которые использует commands.py.

    Часть из них (add_mute/remove_mute/last_bot_replies/audit_stop/
    message_by_tg_id/bot_reply_by_tg_id и версии промпта) пишет параллельно
    другой агент — здесь только сигнатуры из контракта CLAUDE.md, без импорта db.py.
    """

    async def get_state(self, key: str) -> str | None: ...
    async def set_state(self, key: str, value: str) -> None: ...
    async def delete_state(self, key: str) -> None: ...
    async def audit_stop(
        self, key: str, changed_by: int, now: int, new_value: str | None = None
    ) -> None: ...
    async def add_mute(self, user_id: int, display_name: str, muted_by: int, now: int) -> None: ...
    async def remove_mute(self, user_id: int) -> bool: ...
    # Sequence (не list) в возвращаемом типе: list инвариантен по своему параметру,
    # так что list[_BotReplyRowLike] не принял бы Database.last_bot_replies(), чей
    # реальный тип — list[BotReplyRow] (BotReplyRow — подтип _BotReplyRowLike, но
    # List[BotReplyRow] всё равно не подтип List[_BotReplyRowLike]). Sequence
    # объявлен ковариантным по элементу — этой проблемы не создаёт.
    async def last_bot_replies(self, n: int) -> Sequence[_BotReplyRowLike]: ...
    async def message_by_tg_id(
        self, chat_id: int, tg_message_id: int
    ) -> _MessageRowLike | None: ...
    async def bot_reply_by_tg_id(self, tg_message_id: int) -> _BotReplyRowLike | None: ...
    async def filter_log_summary(self, since: int) -> list[tuple[str, int]]: ...
    # -- потолок присутствия (CLAUDE.md, "меньше и разнообразнее") -----------
    async def count_messages_today(self, chat_id: int, day_start: int) -> int: ...
    async def count_bot_replies_today(self, day_start: int) -> int: ...
    async def load_pending(self) -> Sequence[object]: ...
    async def night_unanswered(self) -> Sequence[object]: ...
    async def prompt_versions(self) -> Sequence[_VersionRowLike]: ...
    # -- события жизни (/life, CLAUDE.md "события жизни и /say") -------------
    async def insert_life_event(self, *, text: str, created_at: int) -> int: ...
    async def life_events(self) -> Sequence[LifeEventRow]: ...
    async def life_event(self, event_id: int) -> LifeEventRow | None: ...
    async def delete_life_event(self, event_id: int) -> bool: ...
    # -- долгая память чата (/memory, CLAUDE.md "долгая память чата") --------
    async def chat_memories(self, limit: int) -> Sequence[ChatMemoryRow]: ...
    async def chat_memory(self, memory_id: int) -> ChatMemoryRow | None: ...
    async def chat_memory_count(self) -> int: ...
    async def delete_chat_memory(self, memory_id: int) -> bool: ...
    async def last_chat_memory_end(self) -> int | None: ...
    # -- дневник дня (/diary, CLAUDE.md "дневник дня") -----------------------
    async def self_facts_since(self, since: int) -> Sequence[SelfFactRow]: ...
    async def bot_replies_between(self, start: int, end: int) -> Sequence[_BotReplyRowLike]: ...
    async def delete_self_fact(self, fact_id: int) -> bool: ...
    # -- шутки чата и истории людей (/jokes, /people) ------------------------
    async def jokes(self, limit: int) -> Sequence[JokeRow]: ...
    async def delete_joke(self, joke_id: int) -> bool: ...
    async def people_threads_all(self) -> Sequence[PeopleThreadRow]: ...
    async def delete_people_thread(self, thread_id: int) -> bool: ...
    # -- /ask: имя владельца, каким его видит контекст чата ---------------------
    async def display_names(self, user_ids: Sequence[int]) -> dict[int, str]: ...


class _ResponderLike(Protocol):
    """Подмножество Responder, нужное /life и /say (CLAUDE.md, «события жизни»).

    Только эти два метода — не полный Responder. SendOutcome в аннотациях
    приходит из TYPE_CHECKING-импорта выше (см. его комментарий про то, почему
    commands.py не импортирует sotto.responder напрямую)."""

    async def announce_life(self, event: LifeEventRow) -> SendOutcome: ...
    async def say(self, text: str) -> SendOutcome: ...
    async def preview(self, text: str, display_name: str, user_id: int = 0) -> PreviewOutcome: ...


class _ChatMemorizerLike(Protocol):
    """Подмножество chat_memory.ChatMemorizer, нужное /memory run.

    Только ``run_due`` — список и удаление идут прямо через БД. None означает,
    что LLM-часть выключена (нет ключа): тогда /memory run отвечает об этом и
    ничего не запускает."""

    async def run_due(self, *, now: int) -> Sequence[ChatMemoryRow]: ...


class _VocabStoreLike(Protocol):
    """Подмножество vocab.VocabStore, нужное /words и /status."""

    def get(self) -> Vocab: ...
    async def put(self, raw: str, now: int) -> VocabDict: ...
    async def set_enabled(self, name: str, enabled: bool) -> bool: ...
    async def remove(self, name: str) -> bool: ...
    async def dicts(self) -> list[tuple[VocabDictRow, VocabDict | None]]: ...


class _DownloaderLike(Protocol):
    """Скачивание файла по file_id — подмножество aiogram.Bot.download (как bot.BotLike)."""

    async def download(self, file: str, destination: BinaryIO | None = None) -> BinaryIO | None: ...


class _ConfigStoreLike(Protocol):
    """Методы stores.ConfigStore, которые использует commands.py."""

    def get(self) -> Config: ...
    async def set(
        self, key: str, raw_value: str, changed_by: int, now: int
    ) -> tuple[str | None, str]: ...
    async def unset(self, key: str, changed_by: int, now: int) -> str | None: ...
    def flat(self) -> list[tuple[str, str, bool]]: ...
    def describe(self, key: str) -> KeyInfo | None: ...
    def patterns(self) -> Patterns: ...


class _PromptStoreLike(Protocol):
    """Методы stores.PromptStore, которые использует commands.py."""

    def system_prompt(self) -> str: ...
    def prompt_version(self) -> int: ...
    def few_shot_version(self) -> int: ...
    async def rollback_prompt(self, version: int) -> bool: ...
    async def add_example(self, name: str, user: str, text: str, now: int) -> int: ...
    async def remove_example(self, index: int, now: int) -> int: ...
    def examples(self) -> list[FewShot]: ...


class _WeatherLike(Protocol):
    """Подмножество WeatherClient, нужное /status: один снимок погоды. Как и
    остальные протоколы здесь — структурный, чтобы тестам не собирать настоящий
    клиент с httpx."""

    async def get(self, place: Place | None = None) -> Weather | None: ...


class _CommandsDeps(Protocol):
    """Зависимости build_commands_router. bot.Deps потом дополнят этими полями.

    responder используется /life и /say (announce_life/say) через _ResponderLike —
    структурный протокол, а не прямой тип Responder, чтобы commands.py не тянул
    responder.py (и вместе с ним aiogram-цепочку) в рантайме. responder может быть
    None (LLM-часть выключена, CLAUDE.md "Settings.openrouter_api_key None").

    Члены объявлены через ``@property``, а не как обычные атрибуты: commands.py
    только читает deps.* и никогда не присваивает — по PEP 544 обычный (settable)
    атрибут Protocol требует инвариантного совпадения типа в обе стороны, а
    ``@property`` — только совместимости на чтение (ковариантно). bot.Deps —
    обычный (не frozen) dataclass с более конкретными типами полей (ConfigStore,
    PromptStore, Database, Responder | None), и без ``@property`` он не матчился
    бы структурно на эти протокольные типы.
    """

    @property
    def settings(self) -> Settings: ...
    @property
    def persona(self) -> Persona | None: ...
    @property
    def frag(self) -> Fragments: ...
    @property
    def config_store(self) -> _ConfigStoreLike: ...
    @property
    def prompt_store(self) -> _PromptStoreLike: ...
    @property
    def db(self) -> _DbLike: ...
    @property
    def responder(self) -> _ResponderLike | None: ...
    @property
    def bot_user_id(self) -> int: ...
    @property
    def bot_username(self) -> str: ...
    @property
    def sticker_catalog_enabled(self) -> int: ...
    @property
    def memorizer(self) -> _ChatMemorizerLike | None: ...
    @property
    def weather(self) -> _WeatherLike | None: ...
    @property
    def vocab_store(self) -> _VocabStoreLike | None: ...
    @property
    def bot(self) -> _DownloaderLike | None: ...


def _truncate(text: str) -> str:
    if len(text) <= _MAX_REPLY_LEN:
        return text
    return text[: _MAX_REPLY_LEN - 1] + "…"


async def _reply(message: Message, text: str) -> None:
    await message.answer(_truncate(text), parse_mode=None)


def _catalog(deps: _CommandsDeps) -> Catalog:
    """Owner-UI texts of the pack's locale; a deps without a persona reads the engine's English."""
    return deps.persona.catalog if deps.persona is not None else engine_catalog()


def _t(deps: _CommandsDeps, key: str, /, **params: object) -> str:
    return _catalog(deps).t(key, **params)


def _is_owner(user: User | None, deps: _CommandsDeps) -> bool:
    return (
        user is not None
        and deps.settings.admin_user_id != 0
        and user.id == deps.settings.admin_user_id
    )


def _resolve_mention_target(message: Message) -> tuple[int, str] | None:
    """Цель /mute или /unmute: реплай или text_mention-упоминание в entities.

    Обычный @username (entity type "mention") не поддерживается — username
    в БД не хранится, найти по нему пользователя нечем, игнорируем молча.
    """
    reply = message.reply_to_message
    if reply is not None and reply.from_user is not None:
        return reply.from_user.id, reply.from_user.full_name
    for entity in message.entities or ():
        if entity.type == "text_mention" and entity.user is not None:
            return entity.user.id, entity.user.full_name
    return None


async def _cmd_stop(message: Message, deps: _CommandsDeps, now: int, user: User | None) -> None:
    changed_by = user.id if user is not None else 0
    await deps.db.set_state("stop_until", str(now + 86400))
    await deps.db.audit_stop("stop", changed_by, now)
    logger.info("stop: silenced until %s by user_id=%s", now + 86400, changed_by)


async def _cmd_mute(message: Message, deps: _CommandsDeps, now: int, user: User | None) -> None:
    """Без реплая/упоминания — мьютит самого отправителя, доступно всем. Реплаем
    или упоминанием на ЧУЖОЕ сообщение — только владелец (иначе это готовый
    инструмент травли: «тебя теперь бот не видит»). Цель-владелец или цель-бот —
    молча ничего, кем бы ни была вызвана команда."""
    if user is None:
        return
    target = _resolve_mention_target(message)
    if target is None:
        target_id, target_name = user.id, user.full_name
    else:
        target_id, target_name = target
        if target_id != user.id and not _is_owner(user, deps):
            return
    if target_id == deps.settings.admin_user_id or target_id == deps.bot_user_id:
        return
    display_name = sanitize_display_name(target_name, target_id, set(), frag=deps.frag)
    await deps.db.add_mute(target_id, display_name, user.id, now)
    await deps.db.audit_stop(f"mute:{target_id}", user.id, now)
    logger.info("mute: user_id=%s by user_id=%s", target_id, user.id)


async def _cmd_unmute(message: Message, deps: _CommandsDeps, now: int, user: User | None) -> None:
    """Без реплая/упоминания — снимает мьют с самого отправителя, доступно всем.
    Реплаем или упоминанием на чужого — только владелец."""
    if user is None:
        return
    target = _resolve_mention_target(message)
    target_id = user.id if target is None else target[0]
    if target is not None and target_id != user.id and not _is_owner(user, deps):
        return
    await deps.db.remove_mute(target_id)
    await deps.db.audit_stop(f"unmute:{target_id}", user.id, now)
    logger.info("unmute: user_id=%s by user_id=%s", target_id, user.id)


async def _cmd_ex_add(message: Message, deps: _CommandsDeps, now: int) -> None:
    reply = message.reply_to_message
    if reply is None or reply.from_user is None or reply.from_user.id != deps.bot_user_id:
        return
    bot_reply = await deps.db.bot_reply_by_tg_id(reply.message_id)
    if bot_reply is None:
        return

    trigger_msg = None
    if bot_reply.trigger_tg_message_id is not None:
        trigger_msg = await deps.db.message_by_tg_id(
            message.chat.id, bot_reply.trigger_tg_message_id
        )

    if trigger_msg is not None and trigger_msg.display_name and trigger_msg.text:
        name, user_text = trigger_msg.display_name, trigger_msg.text
    else:
        # Chat-facing: the pair is stored as few-shot data the model sees, so the pack words it.
        name = deps.frag.text("placeholders.ex_add_name")
        user_text = deps.frag.text("placeholders.ex_add_user")

    await deps.prompt_store.add_example(name=name, user=user_text, text=bot_reply.text, now=now)
    logger.info("ex add: %r -> %r", user_text, bot_reply.text)


async def _cmd_panic(message: Message, deps: _CommandsDeps, now: int, user: User) -> None:
    await deps.db.set_state("panic", "1")
    await deps.db.audit_stop("panic", user.id, now)
    logger.info("panic set")
    await _reply(message, _t(deps, "commands.panic_done"))


async def _cmd_resume(message: Message, deps: _CommandsDeps, now: int, user: User) -> None:
    """Снимает panic, stop_until и предохранитель LLM (llm_circuit_until/
    llm_error_streak). Действия безусловны (как раньше для panic/stop_until), но
    в ответе перечисляется только то, что реально было выставлено — иначе
    владелец решит, что бот стоял на паузе, которой не было."""
    panic_was_set = await deps.db.get_state("panic") is not None
    stop_was_set = await deps.db.get_state("stop_until") is not None
    streak_before = await deps.db.get_state("llm_error_streak")
    circuit_was_open = await deps.db.get_state("llm_circuit_until") is not None

    await deps.db.delete_state("panic")
    await deps.db.delete_state("stop_until")
    await deps.db.delete_state("llm_circuit_until")
    await deps.db.set_state("llm_error_streak", "0")
    await deps.db.audit_stop("resume", user.id, now)
    logger.info(
        "resume: panic=%s stop=%s circuit=%s", panic_was_set, stop_was_set, circuit_was_open
    )

    cleared: list[str] = []
    if panic_was_set:
        cleared.append("panic")
    if stop_was_set:
        cleared.append("stop")
    if circuit_was_open:
        streak = streak_before if streak_before is not None else "?"
        cleared.append(_t(deps, "commands.resume_item_circuit", streak=streak))

    if cleared:
        text = _t(deps, "commands.resume_cleared", items=", ".join(cleared))
    else:
        text = _t(deps, "commands.resume_done")
    await _reply(message, text)


def _llm_last_refusal(deps: _CommandsDeps, raw: str | None, now: int, tz: str) -> str:
    """«402 в 13:28 26.10» за последние сутки, иначе «нет за сутки». «Ошибок подряд»
    этого не покажет: 22–26.10 дешёвые модели проходили и сбрасывали серию, пока
    основная получала 402 на каждый вызов."""
    try:
        at_raw, status = (raw or "").split("|", 1)
        at = int(at_raw)
    except ValueError:
        return _t(deps, "status.refusal_none")
    if now - at >= 86400:
        return _t(deps, "status.refusal_none")
    return _t(deps, "status.refusal", status=status, time=local_dt(at, tz).strftime("%H:%M %d.%m"))


async def _cmd_status(message: Message, deps: _CommandsDeps, now: int) -> None:
    cfg = deps.config_store.get()
    tz = cfg.persona.timezone

    panic = await deps.db.get_state("panic") == "1"
    stop_until_raw = await deps.db.get_state("stop_until")
    # Ключ stop_until после истечения никто не удаляет (гейт просто сравнивает его с now),
    # поэтому показываем дату только пока стоп действует — иначе /status врал бы про стоп,
    # которого уже нет (живой случай 16.09.2026).
    stop_until = _t(deps, "status.none")
    if stop_until_raw is not None and int(stop_until_raw) > now:
        stop_until = local_dt(int(stop_until_raw), tz).strftime("%H:%M %d.%m")

    circuit_until_raw = await deps.db.get_state("llm_circuit_until")
    circuit_state = _t(deps, "status.circuit_closed")
    if circuit_until_raw is not None:
        circuit_until = int(circuit_until_raw)
        if circuit_until > now:
            circuit_state = _t(
                deps,
                "status.circuit_open",
                time=local_dt(circuit_until, tz).strftime("%H:%M"),
            )
    llm_error_streak = await deps.db.get_state("llm_error_streak") or "0"
    llm_last_refusal = _llm_last_refusal(deps, await deps.db.get_state("llm_last_refusal"), now, tz)

    ambient = int(await deps.db.get_state(day_key("ambient_count", now, tz)) or "0")
    mention = int(await deps.db.get_state(day_key("mention_count", now, tz)) or "0")
    llm_calls = int(await deps.db.get_state(day_key("llm_calls", now, tz)) or "0")
    llm_spent = float(await deps.db.get_state(day_key("llm_spent_usd", now, tz)) or "0")
    reactions = int(await deps.db.get_state(day_key("reaction_count", now, tz)) or "0")
    reaction_calls = int(await deps.db.get_state(day_key("reaction_calls", now, tz)) or "0")
    stickers = int(await deps.db.get_state(day_key("sticker_count", now, tz)) or "0")
    followup_calls = int(await deps.db.get_state(day_key("followup_calls", now, tz)) or "0")
    vision_count = int(await deps.db.get_state(day_key("vision_count", now, tz)) or "0")
    vocab_line = await _vocab_status_line(deps, cfg, now)

    # Потолок присутствия (CLAUDE.md, "меньше и разнообразнее", мера 1) — считается по
    # таблицам, как в гейте, а не по state-счётчику.
    presence_cfg = cfg.behaviour.presence
    midnight = day_start(now, tz)
    human_today = await deps.db.count_messages_today(deps.settings.allowed_chat_id, midnight)
    bot_today = await deps.db.count_bot_replies_today(midnight)
    presence_line = _t(
        deps,
        "status.presence",
        bot=bot_today,
        allowance=presence_cfg.allowance(human_today),
        people=human_today,
    )

    # Метрика «про себя» (CLAUDE.md, "про себя через раз") — только смотреть, ничего не режет.
    self_rows = await deps.db.bot_replies_between(now - _SELF_SHARE_DAYS * 86400, now + 1)
    self_line = render_self_share(
        self_share(
            ((row.trigger, row.text) for row in self_rows),
            deps.config_store.patterns().self_markers,
            frag=deps.frag,
        ),
        _SELF_SHARE_DAYS,
        frag=deps.frag,
    )

    diary_calls = int(await deps.db.get_state(day_key("diary_calls", now, tz)) or "0")
    diary_today = len(await deps.db.self_facts_since(midnight))

    jokes_total = len(await deps.db.jokes(_JOKES_LIST_LIMIT))
    people_open = sum(1 for thread in await deps.db.people_threads_all() if not thread.closed)
    callback_last_raw = await deps.db.get_state("callback_last_at")
    try:
        callback_last = int(callback_last_raw) if callback_last_raw is not None else None
    except ValueError:
        callback_last = None
    callback_text = (
        _t(deps, "status.callback_last", date=local_dt(callback_last, tz).strftime("%d.%m"))
        if callback_last is not None
        else _t(deps, "status.none")
    )

    pending = len(await deps.db.load_pending())
    night_queue = len(await deps.db.night_unanswered())

    life_events = await deps.db.life_events()
    life_unsent = sum(1 for event in life_events if event.announced_at is None)

    memory_total = await deps.db.chat_memory_count()
    memory_end = await deps.db.last_chat_memory_end()
    memory_line = (
        _t(
            deps,
            "status.memory",
            total=memory_total,
            date=local_dt(memory_end, tz).strftime("%d.%m.%Y"),
        )
        if memory_end is not None
        else _t(deps, "status.memory_none")
    )

    hot_until_raw = await deps.db.get_state("hot_until")
    try:
        hot_until = int(hot_until_raw) if hot_until_raw is not None else None
    except ValueError:
        hot_until = None
    if hot_until is not None and hot_until > now:
        hot_ambient_count = int(await deps.db.get_state("hot_ambient_count") or "0")
        hot_line = _t(
            deps,
            "status.hot",
            time=local_dt(hot_until, tz).strftime("%H:%M"),
            count=hot_ambient_count,
            cap=cfg.behaviour.hot_window.ambient_cap,
        )
    else:
        hot_line = _t(deps, "status.hot_none")

    checkin_due_raw = await deps.db.get_state("checkin_due")
    try:
        checkin_due = int(checkin_due_raw) if checkin_due_raw is not None else None
    except ValueError:
        checkin_due = None
    checkin_line = (
        _t(deps, "status.checkin", time=local_dt(checkin_due, tz).strftime("%H:%M"))
        if checkin_due is not None
        else _t(deps, "status.checkin_none")
    )

    # Погода (CLAUDE.md, "Интерфейсы: погода"): get() может сходить в сеть, если
    # кэш протух, — /status зовут редко, это нормально. Нет домашней точки
    # (координаты не заданы в .env) или клиент выключен -> «погода: нет».
    weather_line = _t(deps, "status.weather_none")
    weather_client = deps.weather
    snapshot = await weather_client.get() if weather_client is not None else None
    if snapshot is not None:
        now_desc = describe(snapshot.code_now, frag=deps.frag)
        weather_line = _t(
            deps,
            "status.weather",
            temp=format_temp(snapshot.temp_now),
            desc=f", {now_desc}" if now_desc else "",
            time=local_dt(snapshot.fetched_at, tz).strftime("%H:%M"),
        )

    yes, no = _t(deps, "status.yes"), _t(deps, "status.no")
    lines = [
        _t(deps, "status.header", version=__version__),
        _t(deps, "status.panic", value=yes if panic else no),
        _t(deps, "status.stop_until", value=stop_until),
        _t(
            deps,
            "status.circuit",
            state=circuit_state,
            streak=llm_error_streak,
            refusal=llm_last_refusal,
        ),
        _t(
            deps,
            "status.prompt",
            prompt=deps.prompt_store.prompt_version(),
            few_shot=deps.prompt_store.few_shot_version(),
        ),
        _t(
            deps,
            "status.models",
            main=cfg.llm.main_model or "-",
            judge=cfg.llm.judge_model or "-",
        ),
        _t(deps, "status.shadow", value=yes if cfg.filters.shadow else no),
        _t(
            deps,
            "status.today",
            ambient=ambient,
            mention=mention,
            llm_calls=llm_calls,
            llm_spent=llm_spent,
            reactions=reactions,
            reactions_cap=cfg.behaviour.reactions.daily_cap,
            reaction_calls=reaction_calls,
            semantic_cap=cfg.behaviour.reactions.semantic_daily_cap,
            stickers=stickers,
            stickers_cap=cfg.behaviour.stickers.daily_cap,
            catalog=deps.sticker_catalog_enabled,
        ),
        _t(deps, "status.pending", count=pending),
        _t(deps, "status.night_queue", count=night_queue),
        _t(deps, "status.life_events", total=len(life_events), unsent=life_unsent),
        memory_line,
        hot_line,
        _t(
            deps,
            "status.followup",
            calls=followup_calls,
            cap=cfg.behaviour.followup.daily_cap,
        ),
        _t(deps, "status.vision", count=vision_count, cap=cfg.behaviour.vision.daily_cap),
        _t(
            deps,
            "status.diary",
            today=diary_today,
            calls=diary_calls,
            cap=cfg.behaviour.diary.daily_cap,
        ),
        _t(
            deps,
            "status.jokes",
            jokes=jokes_total,
            people=people_open,
            callback=callback_text,
        ),
        checkin_line,
        presence_line,
        self_line,
        vocab_line,
        weather_line,
    ]
    await _reply(message, "\n".join(lines))


async def _vocab_status_line(deps: _CommandsDeps, cfg: Config, now: int) -> str:
    """«словечки: 1/2 словарей (100 слов), сегодня 1/2, последнее сырюка 16.10»."""
    store = deps.vocab_store
    if store is None:
        return _t(deps, "status.vocab_none")
    tz = cfg.persona.timezone
    dicts = await store.dicts()
    enabled = sum(1 for row, _ in dicts if row.enabled)
    words = len(store.get().entries)
    today = int(await deps.db.get_state(day_key("vocab_count", now, tz)) or "0")
    last = _t(deps, "status.vocab_last_none")
    last_raw = await deps.db.get_state("vocab_last")
    if last_raw is not None:
        at_raw, _, word = last_raw.partition("|")
        try:
            last = f"{word} {local_dt(int(at_raw), tz).strftime('%d.%m')}"
        except ValueError:
            last = _t(deps, "status.vocab_last_none")
    return _t(
        deps,
        "status.vocab",
        enabled=enabled,
        total=len(dicts),
        words=words,
        today=today,
        cap=cfg.behaviour.vocab.daily_cap,
        last=last,
    )


async def _cmd_last(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    n = 5
    if args:
        try:
            n = int(args[0])
        except ValueError:
            n = 5
    n = max(1, min(n, 20))

    tz = deps.config_store.get().persona.timezone
    rows = await deps.db.last_bot_replies(n)
    lines = [
        _t(
            deps,
            "commands.last_line",
            time=local_dt(row.created_at, tz).strftime("%H:%M %d.%m"),
            trigger=row.trigger,
            prompt=row.prompt_version,
            few_shot=row.few_shot_version,
            delay=row.delay_sec,
            text=row.text,
        )
        for row in rows
    ]
    await _reply(message, "\n".join(lines) if lines else _t(deps, "commands.empty"))


async def _cmd_why(message: Message, deps: _CommandsDeps, now: int, args: list[str]) -> None:
    hours = 1
    if args:
        try:
            hours = int(args[0])
        except ValueError:
            hours = 1
    hours = max(1, min(hours, 168))

    summary = await deps.db.filter_log_summary(now - hours * 3600)
    lines = [f"{reason} {count}" for reason, count in summary]
    await _reply(message, "\n".join(lines) if lines else _t(deps, "commands.empty"))


def _render_key_card(deps: _CommandsDeps, info: KeyInfo) -> str:
    lines = [f"{info.key}: {info.value}"]
    if info.bounds:
        lines.append(_t(deps, "commands.key_type_bounds", type=info.type_name, bounds=info.bounds))
    else:
        lines.append(_t(deps, "commands.key_type", type=info.type_name))
    if info.description:
        lines.append(info.description)
    if info.overridden:
        lines.append(_t(deps, "commands.key_overridden", default=info.default))
    if not info.settable:
        lines.append(_t(deps, "commands.key_yaml_only"))
    return "\n".join(lines)


async def _cmd_get(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    arg = args[0] if args else ""
    if arg:
        info = deps.config_store.describe(arg)
        if info is not None:
            await _reply(message, _render_key_card(deps, info))
            return

    lines = [
        f"{key}{'*' if overridden else ''}: {value}"
        for key, value, overridden in deps.config_store.flat()
        if key.startswith(arg)
    ]
    if not lines:
        await _reply(message, _t(deps, "commands.empty"))
        return
    lines.append(_t(deps, "commands.get_hint"))
    await _reply(message, "\n".join(lines))


async def _cmd_set(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    if len(args) < 2:
        await _reply(message, _t(deps, "commands.set_usage"))
        return
    key, value = args[0], " ".join(args[1:])
    old, new = await deps.config_store.set(key, value, admin_user_id, now)
    await _reply(message, _t(deps, "commands.set_done", key=key, old=old, new=new))


async def _cmd_unset(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    if not args:
        await _reply(message, _t(deps, "commands.unset_usage"))
        return
    key = args[0]
    default = await deps.config_store.unset(key, admin_user_id, now)
    if default is None:
        await _reply(message, _t(deps, "commands.unset_not_overridden", key=key))
    else:
        await _reply(message, _t(deps, "commands.unset_done", key=key, default=default))


async def _cmd_prompt(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    full = bool(args) and args[0].lower() == "full"
    body = deps.prompt_store.system_prompt()
    total = len(await deps.db.prompt_versions())
    header = _t(
        deps, "commands.prompt_header", version=deps.prompt_store.prompt_version(), total=total
    )

    if full:
        text = f"{header}\n\n{body}"
    else:
        preview = body[:_PROMPT_PREVIEW_LEN]
        if len(body) > _PROMPT_PREVIEW_LEN:
            preview += "…"
        text = f"{header}\n\n{preview}"
    await _reply(message, text)


async def _cmd_rollback(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    if not args:
        await _reply(message, _t(deps, "commands.rollback_usage"))
        return
    try:
        version = int(args[0])
    except ValueError:
        await _reply(message, _t(deps, "commands.rollback_bad_number", arg=args[0]))
        return
    if await deps.prompt_store.rollback_prompt(version):
        await _reply(message, _t(deps, "commands.rollback_done", version=version))
    else:
        await _reply(message, _t(deps, "commands.rollback_not_found", version=version))


async def _cmd_ex_last(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    n = 5
    if len(args) > 1:
        try:
            n = int(args[1])
        except ValueError:
            n = 5
    n = max(1, n)

    lines = []
    for example in deps.prompt_store.examples()[-n:]:
        payload = json.dumps(
            {"speak": example.speak, "text": example.text},
            ensure_ascii=False,
            separators=(", ", ": "),
        )
        lines.append(
            _t(deps, "commands.ex_line", name=example.name, user=example.user, payload=payload)
        )
    await _reply(message, "\n".join(lines) if lines else _t(deps, "commands.empty"))


async def _cmd_ex_rm(message: Message, deps: _CommandsDeps, now: int, args: list[str]) -> None:
    n = 1
    if len(args) > 1:
        try:
            n = int(args[1])
        except ValueError:
            n = 1
    n = max(1, n)
    version = await deps.prompt_store.remove_example(n, now)
    await _reply(message, _t(deps, "commands.ex_rm_done", version=version, n=n))


def _parse_int(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def _command_arg_text(raw_text: str, tokens: list[str]) -> str:
    """Текст после команды, с исходными пробелами/переносами внутри (для /say —
    "дословно"), только внешние пробелы срезаны. args (tokens.split()) для этого
    не годится — схлопывает внутренние пробелы."""
    if not tokens:
        return ""
    return raw_text[len(tokens[0]) :].strip()


def _is_blocked(outcome: SendOutcome) -> bool:
    return outcome.reason in ("blocked:panic", "blocked:stop")


def _life_outcome_text(deps: _CommandsDeps, outcome: SendOutcome) -> str:
    """Общая часть ответа /life <текст> и /life post N после вызова announce_life."""
    if outcome.sent:
        return _t(deps, "commands.life_sent", text=outcome.text)
    if _is_blocked(outcome):
        return _t(deps, "commands.blocked")
    text = _t(deps, "commands.life_not_sent", reason=outcome.reason)
    if outcome.text:
        text += "\n" + _t(deps, "commands.candidate", text=outcome.text)
    return text


async def _cmd_life(
    message: Message,
    deps: _CommandsDeps,
    now: int,
    admin_user_id: int,
    args: list[str],
    raw_text: str,
    tokens: list[str],
) -> None:
    sub = args[0].lower() if args else ""
    if sub == "list":
        await _cmd_life_list(message, deps)
        return
    if sub == "rm":
        await _cmd_life_rm(message, deps, now, admin_user_id, args)
        return
    if sub == "post":
        await _cmd_life_post(message, deps, now, admin_user_id, args)
        return

    text = normalize_text(_command_arg_text(raw_text, tokens))
    if not text:
        await _reply(message, _t(deps, "commands.life_usage"))
        return

    event_id = await deps.db.insert_life_event(text=text, created_at=now)
    await deps.db.audit_stop("life:add", admin_user_id, now, text)
    logger.info("life add: #%s %r", event_id, text)

    if deps.responder is None:
        await _reply(message, _t(deps, "commands.life_no_llm", id=event_id))
        return

    event = LifeEventRow(
        id=event_id, text=text, created_at=now, announced_at=None, announced_tg_message_id=None
    )
    outcome = await deps.responder.announce_life(event)
    await _reply(
        message,
        _t(deps, "commands.life_saved", id=event_id, outcome=_life_outcome_text(deps, outcome)),
    )


async def _cmd_life_list(message: Message, deps: _CommandsDeps) -> None:
    tz = deps.config_store.get().persona.timezone
    events = await deps.db.life_events()
    if not events:
        await _reply(message, _t(deps, "commands.life_none"))
        return
    lines = [
        _t(
            deps,
            "commands.life_line",
            id=event.id,
            date=local_dt(event.created_at, tz).strftime("%d.%m.%Y"),
            mark=(
                _t(deps, "commands.life_mark_sent")
                if event.announced_at is not None
                else _t(deps, "commands.dash")
            ),
            text=event.text,
        )
        for event in events
    ]
    await _reply(message, "\n".join(lines))


async def _cmd_life_rm(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    event_id = _parse_int(args[1]) if len(args) > 1 else None
    if event_id is None:
        await _reply(message, _t(deps, "commands.life_usage"))
        return
    event = await deps.db.life_event(event_id)
    if event is None:
        await _reply(message, _t(deps, "commands.life_not_found", id=event_id))
        return
    await deps.db.delete_life_event(event_id)
    await deps.db.audit_stop("life:rm", admin_user_id, now, event.text)
    logger.info("life rm: #%s %r", event_id, event.text)
    await _reply(message, _t(deps, "commands.life_removed", id=event_id))


async def _cmd_life_post(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    event_id = _parse_int(args[1]) if len(args) > 1 else None
    if event_id is None:
        await _reply(message, _t(deps, "commands.life_usage"))
        return
    event = await deps.db.life_event(event_id)
    if event is None:
        await _reply(message, _t(deps, "commands.life_not_found", id=event_id))
        return
    await deps.db.audit_stop("life:post", admin_user_id, now, event.text)
    logger.info("life post: #%s %r", event_id, event.text)

    if deps.responder is None:
        await _reply(message, _t(deps, "commands.life_post_no_llm"))
        return

    outcome = await deps.responder.announce_life(event)
    await _reply(message, _life_outcome_text(deps, outcome))


async def _cmd_memory(
    message: Message,
    deps: _CommandsDeps,
    now: int,
    admin_user_id: int,
    args: list[str],
) -> None:
    """/memory [list|rm N|run] — долгая память чата (CLAUDE.md, "долгая память чата").

    Голое /memory — это /memory list: смотреть, что бот о вас помнит, владелец
    будет чаще, чем что-то менять.
    """
    sub = args[0].lower() if args else "list"
    if sub == "list":
        await _cmd_memory_list(message, deps)
        return
    if sub == "rm":
        await _cmd_memory_rm(message, deps, now, admin_user_id, args)
        return
    if sub == "run":
        await _cmd_memory_run(message, deps, now, admin_user_id)
        return
    await _reply(message, _t(deps, "commands.memory_usage"))


async def _cmd_memory_list(message: Message, deps: _CommandsDeps) -> None:
    tz = deps.config_store.get().persona.timezone
    rows = await deps.db.chat_memories(_MEMORY_LIST_LIMIT)
    if not rows:
        await _reply(message, _t(deps, "commands.memory_none"))
        return
    blocks = [
        _t(
            deps,
            "commands.memory_item",
            id=row.id,
            period=format_period(row.period_start, row.period_end, tz),
            text=row.text,
        )
        for row in rows
    ]
    await _reply(message, "\n\n".join(blocks))


async def _cmd_memory_rm(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    memory_id = _parse_int(args[1]) if len(args) > 1 else None
    if memory_id is None:
        await _reply(message, _t(deps, "commands.memory_usage"))
        return
    row = await deps.db.chat_memory(memory_id)
    if row is None:
        await _reply(message, _t(deps, "commands.memory_not_found", id=memory_id))
        return
    await deps.db.delete_chat_memory(memory_id)
    await deps.db.audit_stop("memory:rm", admin_user_id, now, row.text)
    logger.info("memory rm: #%s, символов %d", memory_id, len(row.text))
    await _reply(message, _t(deps, "commands.memory_removed", id=memory_id))


async def _cmd_diary(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    """/diary [rm N] — факты о себе, достанные из реплик (CLAUDE.md, "дневник дня").

    Голое /diary — список за неделю «#N dd.mm HH:MM текст»; /diary rm N удаляет факт
    (если модель вытащила не то, он перестаёт попадать в слот {diary})."""
    if not args:
        tz = deps.config_store.get().persona.timezone
        rows = await deps.db.self_facts_since(now - _DIARY_LIST_DAYS * 86400)
        if not rows:
            await _reply(message, _t(deps, "commands.diary_none"))
            return
        lines = [
            _t(
                deps,
                "commands.diary_line",
                id=row.id,
                time=local_dt(row.created_at, tz).strftime("%d.%m %H:%M"),
                text=row.text,
            )
            for row in rows
        ]
        await _reply(message, "\n".join(lines))
        return
    if args[0].lower() != "rm":
        await _reply(message, _t(deps, "commands.diary_usage"))
        return
    fact_id = _parse_int(args[1]) if len(args) > 1 else None
    if fact_id is None:
        await _reply(message, _t(deps, "commands.diary_usage"))
        return
    rows = await deps.db.self_facts_since(0)
    fact = next((row for row in rows if row.id == fact_id), None)
    if fact is None:
        await _reply(message, _t(deps, "commands.diary_not_found", id=fact_id))
        return
    await deps.db.delete_self_fact(fact_id)
    await deps.db.audit_stop("diary:rm", admin_user_id, now, fact.text)
    logger.info("diary rm: #%s", fact_id)
    await _reply(message, _t(deps, "commands.diary_removed", id=fact_id))


async def _cmd_jokes(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    """/jokes [rm N] — шутки чата, которые он помнит (CLAUDE.md, "шутки чата и истории
    людей"). /jokes rm N удаляет шутку из слота {jokes}."""
    rows = await deps.db.jokes(_JOKES_LIST_LIMIT)
    if not args:
        if not rows:
            await _reply(message, _t(deps, "commands.jokes_none"))
            return
        tz = deps.config_store.get().persona.timezone
        lines = [
            _t(
                deps,
                "commands.jokes_line",
                id=row.id,
                text=row.text,
                used=(
                    local_dt(row.last_used_at, tz).strftime("%d.%m")
                    if row.last_used_at
                    else _t(deps, "commands.dash")
                ),
            )
            for row in rows
        ]
        await _reply(message, "\n".join(lines))
        return
    joke_id = _parse_int(args[1]) if args[0].lower() == "rm" and len(args) > 1 else None
    if joke_id is None:
        await _reply(message, _t(deps, "commands.jokes_usage"))
        return
    joke = next((row for row in rows if row.id == joke_id), None)
    if joke is None:
        await _reply(message, _t(deps, "commands.jokes_not_found", id=joke_id))
        return
    await deps.db.delete_joke(joke_id)
    await deps.db.audit_stop("jokes:rm", admin_user_id, now, joke.text)
    logger.info("jokes rm: #%s", joke_id)
    await _reply(message, _t(deps, "commands.jokes_removed", id=joke_id))


async def _cmd_people(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int, args: list[str]
) -> None:
    """/people [rm N] — незакрытые истории людей для «ну как там?» (CLAUDE.md, "шутки
    чата и истории людей"). Это заметки о конкретных людях: владелец видит их все и
    может удалить любую."""
    rows = await deps.db.people_threads_all()
    if not args:
        if not rows:
            await _reply(message, _t(deps, "commands.people_none"))
            return
        tz = deps.config_store.get().persona.timezone
        lines = [
            _t(
                deps,
                "commands.people_line",
                id=row.id,
                name=row.display_name,
                text=row.text,
                state=(
                    _t(
                        deps,
                        "commands.people_asked",
                        date=local_dt(row.asked_at, tz).strftime("%d.%m"),
                    )
                    if row.asked_at is not None
                    else _t(deps, "commands.people_open")
                ),
            )
            for row in rows
        ]
        await _reply(message, "\n".join(lines))
        return
    thread_id = _parse_int(args[1]) if args[0].lower() == "rm" and len(args) > 1 else None
    if thread_id is None:
        await _reply(message, _t(deps, "commands.people_usage"))
        return
    thread = next((row for row in rows if row.id == thread_id), None)
    if thread is None:
        await _reply(message, _t(deps, "commands.people_not_found", id=thread_id))
        return
    await deps.db.delete_people_thread(thread_id)
    await deps.db.audit_stop("people:rm", admin_user_id, now, thread.text)
    logger.info("people rm: #%s", thread_id)
    await _reply(message, _t(deps, "commands.people_removed", id=thread_id))


async def _cmd_memory_run(
    message: Message, deps: _CommandsDeps, now: int, admin_user_id: int
) -> None:
    """Принудительный прогон, вне run_window — чтобы не ждать до ночи."""
    if deps.memorizer is None:
        await _reply(message, _t(deps, "commands.memory_run_off"))
        return
    await deps.db.audit_stop("memory:run", admin_user_id, now)
    tz = deps.config_store.get().persona.timezone
    created = await deps.memorizer.run_due(now=now)
    logger.info("memory run: добавлено %d", len(created))
    if not created:
        await _reply(message, _t(deps, "commands.memory_run_none"))
        return
    blocks = [
        _t(
            deps,
            "commands.memory_item",
            id=row.id,
            period=format_period(row.period_start, row.period_end, tz),
            text=row.text,
        )
        for row in created
    ]
    preview = "\n\n".join(blocks)[:_MEMORY_RUN_PREVIEW_LEN]
    await _reply(message, _t(deps, "commands.memory_run_done", count=len(created), preview=preview))


async def _cmd_say(
    message: Message,
    deps: _CommandsDeps,
    now: int,
    admin_user_id: int,
    raw_text: str,
    tokens: list[str],
) -> None:
    text = _command_arg_text(raw_text, tokens)
    if not text:
        await _reply(message, _t(deps, "commands.say_usage"))
        return

    await deps.db.audit_stop("say", admin_user_id, now, text)
    logger.info("say: %r", text)

    if deps.responder is None:
        await _reply(message, _t(deps, "commands.say_off"))
        return

    outcome = await deps.responder.say(text)
    if outcome.sent:
        await _reply(message, _t(deps, "commands.say_sent"))
    else:
        await _reply(message, _t(deps, "commands.blocked"))


def _preview_text(deps: _CommandsDeps, outcome: PreviewOutcome) -> str:
    """Ответ на /ask: что ушло бы в чат, или почему промолчал бы."""
    if outcome.sent:
        lines = [_t(deps, "commands.ask_would_answer", text=outcome.text)]
        if outcome.reasons:
            lines.append(_t(deps, "commands.ask_shadow", reasons=", ".join(outcome.reasons)))
    elif outcome.reason == "llm:silent":
        lines = [_t(deps, "commands.ask_silent")]
    elif outcome.reason.startswith("llm:"):
        lines = [_t(deps, "commands.ask_llm_error", reason=outcome.reason)]
        if outcome.text:
            lines.append(_t(deps, "commands.ask_model_returned", text=outcome.text))
    else:
        lines = [
            _t(
                deps,
                "commands.ask_filtered",
                reasons=", ".join(outcome.reasons) or outcome.reason,
            )
        ]
        if outcome.text:
            lines.append(_t(deps, "commands.candidate", text=outcome.text))
    if outcome.fixes:
        lines.append(_t(deps, "commands.ask_fixes", fixes=", ".join(outcome.fixes)))
    lines.append(_t(deps, "commands.ask_not_sent"))
    return "\n".join(lines)


async def _cmd_ask(
    message: Message, deps: _CommandsDeps, user: User, raw_text: str, tokens: list[str]
) -> None:
    """/ask <текст> (CLAUDE.md, "предпросмотр ответа") — что ответил бы Фёдор, если бы
    владелец обратился к нему в чате. Ничего не отправляет и не пишет."""
    text = normalize_text(_command_arg_text(raw_text, tokens))
    if not text:
        await _reply(message, _t(deps, "commands.ask_usage"))
        return
    if deps.responder is None:
        await _reply(message, _t(deps, "commands.ask_off"))
        return
    # Имя — то, под которым владелец уже есть в контексте чата; не писал в чат —
    # имя из Telegram через тот же sanitize, что и для сообщений.
    known = await deps.db.display_names([user.id])
    name = known.get(user.id) or sanitize_display_name(
        user.full_name, user.id, set(), frag=deps.frag
    )
    logger.info("ask preview: %r", text)
    outcome = await deps.responder.preview(text, name, user.id)
    await _reply(message, _preview_text(deps, outcome))


def _plural_words(deps: _CommandsDeps, n: int) -> str:
    return _catalog(deps).plural("commands.words_count", n)


def _words_too_big(deps: _CommandsDeps) -> str:
    return _t(deps, "commands.words_too_big", kb=_WORDS_MAX_FILE_BYTES // 1024)


async def _cmd_words_upload(
    message: Message, deps: _CommandsDeps, store: _VocabStoreLike, now: int, admin_user_id: int
) -> None:
    """Документ с подписью /words — загрузить словарь (тот же name — замена)."""
    document = message.document
    assert document is not None  # гарантирует вызывающий
    if document.file_size is not None and document.file_size > _WORDS_MAX_FILE_BYTES:
        await _reply(message, _words_too_big(deps))
        return
    if deps.bot is None:
        await _reply(message, _t(deps, "commands.words_no_download"))
        return
    buffer = io.BytesIO()
    await deps.bot.download(document.file_id, destination=buffer)
    data = buffer.getvalue()
    if len(data) > _WORDS_MAX_FILE_BYTES:
        await _reply(message, _words_too_big(deps))
        return
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        await _reply(message, _t(deps, "commands.words_not_utf8"))
        return
    try:
        parsed = await store.put(raw, now)
    except ValueError as exc:
        await _reply(message, _t(deps, "commands.words_parse_error", error=exc))
        return
    await deps.db.audit_stop("words:put", admin_user_id, now, parsed.name)
    logger.info("words put: %r, %d words", parsed.name, len(parsed.words))
    enabled = next((row.enabled for row, _ in await store.dicts() if row.name == parsed.name), True)
    state = _t(deps, "commands.words_state_on" if enabled else "commands.words_state_off_hint")
    await _reply(
        message,
        _t(
            deps,
            "commands.words_loaded",
            name=parsed.name,
            words=_plural_words(deps, len(parsed.words)),
            state=state,
        ),
    )


def _words_usage(deps: _CommandsDeps) -> str:
    return _t(deps, "commands.words_usage")


def _resolve_dict_name(
    dicts: Sequence[tuple[VocabDictRow, VocabDict | None]], name: str
) -> VocabDictRow | None:
    """Имя словаря: точное совпадение, иначе единственное без учёта регистра."""
    exact = [row for row, _ in dicts if row.name == name]
    if exact:
        return exact[0]
    folded = [row for row, _ in dicts if row.name.casefold() == name.casefold()]
    return folded[0] if len(folded) == 1 else None


async def _cmd_words(
    message: Message,
    deps: _CommandsDeps,
    now: int,
    admin_user_id: int,
    args: list[str],
) -> None:
    """/words — словари словечек (CLAUDE.md, "словечки из словаря")."""
    store = deps.vocab_store
    if store is None:
        await _reply(message, _t(deps, "commands.words_unavailable"))
        return
    if message.document is not None:
        await _cmd_words_upload(message, deps, store, now, admin_user_id)
        return
    dicts = await store.dicts()
    tz = deps.config_store.get().persona.timezone
    if not args:
        if not dicts:
            await _reply(message, _t(deps, "commands.words_none") + "\n" + _words_usage(deps))
            return
        lines = []
        for row, parsed in dicts:
            size = (
                _plural_words(deps, len(parsed.words))
                if parsed is not None
                else _t(deps, "commands.words_unparsable")
            )
            lines.append(
                _t(
                    deps,
                    "commands.words_line",
                    name=row.name,
                    size=size,
                    state=_t(deps, "commands.words_on" if row.enabled else "commands.words_off"),
                    date=local_dt(row.updated_at, tz).strftime("%d.%m"),
                )
            )
        lines.append(_words_usage(deps))
        await _reply(message, "\n".join(lines))
        return
    sub_cmd = args[0].lower()
    name = " ".join(args[1:]).strip()
    if sub_cmd not in {"show", "on", "off", "rm"} or not name:
        await _reply(message, _words_usage(deps))
        return
    found = _resolve_dict_name(dicts, name)
    if found is None:
        await _reply(message, _t(deps, "commands.words_no_dict", name=name))
        return
    row = found
    if sub_cmd == "show":
        parsed = next(p for r, p in dicts if r.name == row.name)
        if parsed is None:
            await _reply(message, _t(deps, "commands.words_show_unparsable", name=row.name))
            return
        words_text = _plural_words(deps, len(parsed.words))
        if parsed.about:
            header = _t(
                deps,
                "commands.words_header_about",
                name=row.name,
                words=words_text,
                about=parsed.about,
            )
        else:
            header = _t(deps, "commands.words_header", name=row.name, words=words_text)
        lines = [header] + [f"{item.word}: {item.meaning}" for item in parsed.words]
        await _reply(message, "\n".join(lines))
        return
    if sub_cmd == "rm":
        await store.remove(row.name)
        await deps.db.audit_stop("words:rm", admin_user_id, now, row.name)
        logger.info("words rm: %r", row.name)
        await _reply(message, _t(deps, "commands.words_removed", name=row.name))
        return
    enabled = sub_cmd == "on"
    await store.set_enabled(row.name, enabled)
    await deps.db.audit_stop(f"words:{sub_cmd}", admin_user_id, now, row.name)
    logger.info("words %s: %r", sub_cmd, row.name)
    await _reply(
        message,
        _t(deps, "commands.words_enabled" if enabled else "commands.words_disabled", name=row.name),
    )


async def _cmd_changelog(message: Message, deps: _CommandsDeps, args: list[str]) -> None:
    """/changelog [owner|<версия>] — релиз-ноты (CLAUDE.md, "версии и changelog").

    Блок для чата — `chat_notes.md` пакета, а без него (переходное время) блок «Для чата» из
    CHANGELOG.md пакета; блок для владельца — из CHANGELOG.md пакета. Файлы читаются при старте:
    смена пакета это рестарт.
    """
    persona = deps.persona
    chat_notes = persona.chat_notes if persona is not None else ""
    legacy = persona.legacy_changelog if persona is not None else ""

    owner_view = bool(args) and args[0].lower() == "owner"
    releases = owner_releases(legacy) if owner_view else chat_releases(chat_notes, legacy)
    if not releases:
        await _reply(message, _t(deps, "commands.changelog_missing"))
        return

    found: Release | None
    if owner_view or not args:
        found = releases[0]
    else:
        raw = args[0].lower()
        wanted = raw[1:] if raw.startswith("v") else raw
        found = next((r for r in releases if r.version == wanted), None)
        if found is None:
            await _reply(message, _t(deps, "commands.changelog_no_version", arg=args[0]))
            return
    release = found

    body = release.for_owner if owner_view else release.for_chat
    await _reply(
        message,
        _t(
            deps,
            "commands.changelog_release",
            version=release.version,
            date=release.date,
            body=body,
        ),
    )


def build_commands_router(deps: _CommandsDeps) -> Router:
    """Собирает Router с одним хендлером-диспетчером команд.

    Должен быть включён в Dispatcher ПЕРВЫМ, перед sotto.bot.build_router —
    команды не должны попадать в гейт и в лог/БД основного хендлера. aiogram
    сам останавливает распространение сообщения после первого хендлера, чья
    функция вернулась без исключения, так что явного "cancel propagation" не
    требуется — важен только порядок регистрации роутеров у интегратора.
    """
    router = Router(name=__name__)

    @router.message((F.text & F.text.startswith("/")) | (F.caption & F.caption.startswith("/")))
    async def handle_command(message: Message) -> None:
        text = message.text or message.caption or ""
        tokens = text.split()
        if not tokens:
            return
        cmd_token = tokens[0][1:]
        if "@" in cmd_token:
            cmd_part, _, suffix = cmd_token.partition("@")
            # Суффикс "@другой_бот" в групповых чатах Telegram рассылает команду
            # всем ботам сразу — исполняем только адресованную нам (без учёта
            # регистра); иначе, например, /stop@weatherbot остановил бы и нас.
            if suffix.lower() != deps.bot_username.lower():
                return
            cmd = cmd_part.lower()
        else:
            cmd = cmd_token.lower()
        args = tokens[1:]

        user = message.from_user
        now = int(message.date.timestamp())

        is_chat = message.chat.id == deps.settings.allowed_chat_id
        is_owner_private = (
            message.chat.type == "private"
            and deps.settings.admin_user_id != 0
            and user is not None
            and user.id == deps.settings.admin_user_id
        )
        if not is_chat and not is_owner_private:
            return

        try:
            if is_chat:
                if cmd == "stop":
                    await _cmd_stop(message, deps, now, user)
                elif cmd == "mute":
                    await _cmd_mute(message, deps, now, user)
                elif cmd == "unmute":
                    await _cmd_unmute(message, deps, now, user)
                elif cmd == "ex" and args and args[0].lower() == "add" and _is_owner(user, deps):
                    await _cmd_ex_add(message, deps, now)
                # Неизвестная/чужая команда в чате — молчание, без лога и без ответа.
            else:
                assert user is not None  # гарантировано is_owner_private выше
                admin_user_id = deps.settings.admin_user_id
                if cmd == "panic":
                    await _cmd_panic(message, deps, now, user)
                elif cmd == "resume":
                    await _cmd_resume(message, deps, now, user)
                elif cmd == "status":
                    await _cmd_status(message, deps, now)
                elif cmd == "last":
                    await _cmd_last(message, deps, args)
                elif cmd == "why":
                    await _cmd_why(message, deps, now, args)
                elif cmd == "get":
                    await _cmd_get(message, deps, args)
                elif cmd == "set":
                    await _cmd_set(message, deps, now, admin_user_id, args)
                elif cmd == "unset":
                    await _cmd_unset(message, deps, now, admin_user_id, args)
                elif cmd == "prompt":
                    await _cmd_prompt(message, deps, args)
                elif cmd == "rollback":
                    await _cmd_rollback(message, deps, args)
                elif cmd == "ex" and args and args[0].lower() == "last":
                    await _cmd_ex_last(message, deps, args)
                elif cmd == "ex" and args and args[0].lower() == "rm":
                    await _cmd_ex_rm(message, deps, now, args)
                elif cmd == "life":
                    await _cmd_life(message, deps, now, admin_user_id, args, text, tokens)
                elif cmd == "say":
                    await _cmd_say(message, deps, now, admin_user_id, text, tokens)
                elif cmd == "ask":
                    await _cmd_ask(message, deps, user, text, tokens)
                elif cmd == "memory":
                    await _cmd_memory(message, deps, now, admin_user_id, args)
                elif cmd == "diary":
                    await _cmd_diary(message, deps, now, admin_user_id, args)
                elif cmd == "jokes":
                    await _cmd_jokes(message, deps, now, admin_user_id, args)
                elif cmd == "people":
                    await _cmd_people(message, deps, now, admin_user_id, args)
                elif cmd == "words":
                    await _cmd_words(message, deps, now, admin_user_id, args)
                elif cmd == "changelog":
                    await _cmd_changelog(message, deps, args)
                else:  # /help and any unknown command
                    await _reply(message, _t(deps, "help.text"))
        except Exception as exc:
            logger.exception("command /%s failed", cmd)
            if is_owner_private:
                detail = exc.render(_catalog(deps)) if isinstance(exc, OwnerError) else str(exc)
                await _reply(message, _t(deps, "errors.generic", message=detail))

    return router
