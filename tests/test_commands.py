"""Tests for sotto.commands: the command dispatcher, with no network and no real Bot.

Message/Chat/User are built directly, as in tests/test_bot.py. The handler is taken from the
router and called directly. ConfigStore/PromptStore/Database are faked with classes that
structurally match the Protocols of commands.py (_ConfigStoreLike, _PromptStoreLike, _DbLike):
they need no inheritance, only the right methods. Message.answer is intercepted by the `sent`
fixture so the real Bot API is never called. The persona is the English demo pack, so every
owner-facing answer comes from the English catalog (locales/en).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from zoneinfo import ZoneInfo

import pytest
from aiogram.types import Chat, Document, Message, MessageEntity, PhotoSize, User

from sotto.commands import build_commands_router
from sotto.config import KeyInfo
from sotto.config_models import Config
from sotto.db import (
    ChatMemoryRow,
    Database,
    JokeRow,
    LifeEventRow,
    PeopleThreadRow,
    SelfFactRow,
)
from sotto.few_shot import FewShot
from sotto.fragments import Fragments
from sotto.patterns import Patterns
from sotto.persona import Persona, load_persona
from sotto.responder import PreviewOutcome, SendOutcome
from sotto.settings import Settings
from sotto.timeutil import day_key, local_dt
from sotto.vocab import VocabStore
from sotto.weather import Place, Weather
from tests.demo import DEMO_FRAG, demo_config
from tests.packs import DEMO_PACK

OWN_CHAT_ID = -1001234567890
FOREIGN_CHAT_ID = -100999
ADMIN_ID = 555
BOT_USER_ID = 999
BOT_USERNAME = "gusbot"
NOW = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)


# --- fakes of Database/ConfigStore/PromptStore -----------------------------------


@dataclass
class FakeBotReplyRow:
    trigger: str
    trigger_tg_message_id: int | None
    text: str
    prompt_version: int
    few_shot_version: int
    delay_sec: int
    created_at: int


@dataclass
class FakeMessageRow:
    display_name: str | None
    text: str | None


@dataclass
class FakeVersionRow:
    version: int


@dataclass
class FakeDb:
    state: dict[str, str] = field(default_factory=dict)
    audit_stop_calls: list[tuple[str, int, int]] = field(default_factory=list)
    audit_values: dict[str, str | None] = field(default_factory=dict)
    add_mute_calls: list[tuple[int, str, int, int]] = field(default_factory=list)
    remove_mute_calls: list[int] = field(default_factory=list)
    bot_replies_by_id: dict[int, FakeBotReplyRow] = field(default_factory=dict)
    messages: dict[tuple[int, int], FakeMessageRow] = field(default_factory=dict)
    last_replies_value: list[FakeBotReplyRow] = field(default_factory=list)
    filter_log_summary_calls: list[int] = field(default_factory=list)
    filter_log_summary_value: list[tuple[str, int]] = field(default_factory=list)
    pending_value: list[object] = field(default_factory=list)
    night_value: list[object] = field(default_factory=list)
    prompt_versions_value: list[FakeVersionRow] = field(
        default_factory=lambda: [FakeVersionRow(1), FakeVersionRow(2), FakeVersionRow(3)]
    )
    life_events_store: dict[int, LifeEventRow] = field(default_factory=dict)
    _next_life_event_id: int = 1
    chat_memory_store: dict[int, ChatMemoryRow] = field(default_factory=dict)
    self_facts_store: dict[int, SelfFactRow] = field(default_factory=dict)
    bot_replies_store: list[FakeBotReplyRow] = field(default_factory=list)
    jokes_store: dict[int, JokeRow] = field(default_factory=dict)
    threads_store: dict[int, PeopleThreadRow] = field(default_factory=dict)
    # Presence cap: /status computes it from the tables; here it is just two numbers.
    messages_today_value: int = 0
    bot_replies_today_value: int = 0

    display_names_value: dict[int, str] = field(default_factory=dict)

    async def display_names(self, user_ids: list[int]) -> dict[int, str]:
        return {uid: name for uid, name in self.display_names_value.items() if uid in user_ids}

    async def count_messages_today(self, chat_id: int, day_start: int) -> int:
        return self.messages_today_value

    async def count_bot_replies_today(self, day_start: int) -> int:
        return self.bot_replies_today_value

    async def get_state(self, key: str) -> str | None:
        return self.state.get(key)

    async def set_state(self, key: str, value: str) -> None:
        self.state[key] = value

    async def delete_state(self, key: str) -> None:
        self.state.pop(key, None)

    async def audit_stop(
        self, key: str, changed_by: int, now: int, new_value: str | None = None
    ) -> None:
        self.audit_stop_calls.append((key, changed_by, now))
        self.audit_values[key] = new_value

    async def add_mute(self, user_id: int, display_name: str, muted_by: int, now: int) -> None:
        self.add_mute_calls.append((user_id, display_name, muted_by, now))

    async def remove_mute(self, user_id: int) -> bool:
        self.remove_mute_calls.append(user_id)
        return True

    async def last_bot_replies(self, n: int) -> list[FakeBotReplyRow]:
        return self.last_replies_value[:n]

    async def message_by_tg_id(self, chat_id: int, tg_message_id: int) -> FakeMessageRow | None:
        return self.messages.get((chat_id, tg_message_id))

    async def bot_reply_by_tg_id(self, tg_message_id: int) -> FakeBotReplyRow | None:
        return self.bot_replies_by_id.get(tg_message_id)

    async def filter_log_summary(self, since: int) -> list[tuple[str, int]]:
        self.filter_log_summary_calls.append(since)
        return self.filter_log_summary_value

    async def load_pending(self) -> list[object]:
        return self.pending_value

    async def night_unanswered(self) -> list[object]:
        return self.night_value

    async def prompt_versions(self) -> list[FakeVersionRow]:
        return self.prompt_versions_value

    async def insert_life_event(self, *, text: str, created_at: int) -> int:
        event_id = self._next_life_event_id
        self._next_life_event_id += 1
        self.life_events_store[event_id] = LifeEventRow(
            id=event_id,
            text=text,
            created_at=created_at,
            announced_at=None,
            announced_tg_message_id=None,
        )
        return event_id

    async def life_events(self) -> list[LifeEventRow]:
        return sorted(self.life_events_store.values(), key=lambda e: (e.created_at, e.id))

    async def life_event(self, event_id: int) -> LifeEventRow | None:
        return self.life_events_store.get(event_id)

    async def delete_life_event(self, event_id: int) -> bool:
        return self.life_events_store.pop(event_id, None) is not None

    async def chat_memories(self, limit: int) -> list[ChatMemoryRow]:
        rows = sorted(self.chat_memory_store.values(), key=lambda r: (r.period_end, r.id))
        return rows[-limit:] if limit > 0 else []

    async def chat_memory(self, memory_id: int) -> ChatMemoryRow | None:
        return self.chat_memory_store.get(memory_id)

    async def chat_memory_count(self) -> int:
        return len(self.chat_memory_store)

    async def delete_chat_memory(self, memory_id: int) -> bool:
        return self.chat_memory_store.pop(memory_id, None) is not None

    async def last_chat_memory_end(self) -> int | None:
        if not self.chat_memory_store:
            return None
        return max(row.period_end for row in self.chat_memory_store.values())

    async def bot_replies_between(self, start: int, end: int) -> list[FakeBotReplyRow]:
        return [row for row in self.bot_replies_store if start <= row.created_at < end]

    async def self_facts_since(self, since: int) -> list[SelfFactRow]:
        rows = [row for row in self.self_facts_store.values() if row.created_at >= since]
        return sorted(rows, key=lambda r: (r.created_at, r.id))

    async def delete_self_fact(self, fact_id: int) -> bool:
        return self.self_facts_store.pop(fact_id, None) is not None

    async def jokes(self, limit: int) -> list[JokeRow]:
        rows = sorted(self.jokes_store.values(), key=lambda r: (r.created_at, r.id), reverse=True)
        return rows[:limit]

    async def delete_joke(self, joke_id: int) -> bool:
        return self.jokes_store.pop(joke_id, None) is not None

    async def people_threads_all(self) -> list[PeopleThreadRow]:
        return sorted(self.threads_store.values(), key=lambda r: (r.created_at, r.id))

    async def delete_people_thread(self, thread_id: int) -> bool:
        return self.threads_store.pop(thread_id, None) is not None


@dataclass
class FakeConfigStore:
    cfg: Config
    set_calls: list[tuple[str, str, int, int]] = field(default_factory=list)
    set_result: tuple[str | None, str] = ("3", "5")
    set_error: Exception | None = None
    unset_calls: list[tuple[str, int, int]] = field(default_factory=list)
    unset_result: str | None = "3"
    flat_value: list[tuple[str, str, bool]] = field(
        default_factory=lambda: [
            ("behaviour.daily_cap", "3", False),
            ("behaviour.ambient_probability", "0.15*", True),
            ("llm.main_model", "openrouter/x", False),
        ]
    )
    describe_value: dict[str, KeyInfo | None] = field(default_factory=dict)

    def get(self) -> Config:
        return self.cfg

    def describe(self, key: str) -> KeyInfo | None:
        return self.describe_value.get(key)

    def patterns(self) -> Patterns:
        return Patterns(self.cfg.filters, name_triggers=[], bot_username="")

    async def set(
        self, key: str, raw_value: str, changed_by: int, now: int
    ) -> tuple[str | None, str]:
        self.set_calls.append((key, raw_value, changed_by, now))
        if self.set_error is not None:
            raise self.set_error
        return self.set_result

    async def unset(self, key: str, changed_by: int, now: int) -> str | None:
        self.unset_calls.append((key, changed_by, now))
        return self.unset_result

    def flat(self) -> list[tuple[str, str, bool]]:
        return self.flat_value


@dataclass
class FakePromptStore:
    prompt_version_value: int = 3
    few_shot_version_value: int = 2
    system_prompt_value: str = "the persona system prompt"
    rollback_result: bool = True
    rollback_calls: list[int] = field(default_factory=list)
    add_example_calls: list[tuple[str, str, str, int]] = field(default_factory=list)
    remove_example_calls: list[tuple[int, int]] = field(default_factory=list)
    remove_example_result: int = 7
    examples_value: list[FewShot] = field(default_factory=list)

    def system_prompt(self) -> str:
        return self.system_prompt_value

    def prompt_version(self) -> int:
        return self.prompt_version_value

    def few_shot_version(self) -> int:
        return self.few_shot_version_value

    async def rollback_prompt(self, version: int) -> bool:
        self.rollback_calls.append(version)
        return self.rollback_result

    async def add_example(self, name: str, user: str, text: str, now: int) -> int:
        self.add_example_calls.append((name, user, text, now))
        return 1

    async def remove_example(self, index: int, now: int) -> int:
        self.remove_example_calls.append((index, now))
        return self.remove_example_result

    def examples(self) -> list[FewShot]:
        return self.examples_value


@dataclass
class FakeResponder:
    """Structurally matches commands._ResponderLike."""

    announce_life_result: SendOutcome = field(
        default_factory=lambda: SendOutcome(sent=True, text="Sold the cart", reason="send:life")
    )
    say_result: SendOutcome = field(
        default_factory=lambda: SendOutcome(sent=True, text="", reason="send:say")
    )
    announce_life_calls: list[LifeEventRow] = field(default_factory=list)
    say_calls: list[str] = field(default_factory=list)
    preview_result: PreviewOutcome = field(
        default_factory=lambda: PreviewOutcome(sent=True, text="Happens.", reason="send:mention")
    )
    preview_calls: list[tuple[str, str]] = field(default_factory=list)

    async def preview(self, text: str, display_name: str, user_id: int = 0) -> PreviewOutcome:
        assert user_id == ADMIN_ID
        self.preview_calls.append((text, display_name))
        return self.preview_result

    async def announce_life(self, event: LifeEventRow) -> SendOutcome:
        self.announce_life_calls.append(event)
        return self.announce_life_result

    async def say(self, text: str) -> SendOutcome:
        self.say_calls.append(text)
        return self.say_result


@dataclass
class FakeMemorizer:
    """Structurally matches commands._ChatMemorizerLike."""

    run_due_result: list[ChatMemoryRow] = field(default_factory=list)
    run_due_calls: list[int] = field(default_factory=list)

    async def run_due(self, *, now: int) -> list[ChatMemoryRow]:
        self.run_due_calls.append(now)
        return self.run_due_result


class FakeWeather:
    """A WeatherClient fake for /status: a snapshot without the network."""

    def __init__(self, snapshot: Weather | None = None, home_name: str = "") -> None:
        self.snapshot = snapshot
        self._home_name = home_name
        self.calls = 0

    @property
    def home_name(self) -> str:
        return self._home_name

    async def get(self, place: Place | None = None) -> Weather | None:
        self.calls += 1
        return self.snapshot


@dataclass
class FakeDeps:
    """Structurally matches commands._CommandsDeps."""

    settings: Settings
    config_store: FakeConfigStore
    prompt_store: FakePromptStore
    db: FakeDb
    responder: FakeResponder | None
    bot_user_id: int
    bot_username: str
    sticker_catalog_enabled: int = 0
    memorizer: FakeMemorizer | None = None
    weather: FakeWeather | None = None
    vocab_store: VocabStore | None = None
    bot: FakeDownloader | None = None
    persona: Persona | None = None
    frag: Fragments = DEMO_FRAG


class FakeDownloader:
    """A fake of aiogram.Bot.download for dictionary uploads: returns the given bytes."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.calls: list[str] = []

    async def download(self, file: str, destination: BinaryIO | None = None) -> BinaryIO | None:
        self.calls.append(file)
        assert destination is not None
        destination.write(self.data)
        return destination


# --- helper constructors -------------------------------------------------


def _chat(chat_id: int = OWN_CHAT_ID) -> Chat:
    return Chat(id=chat_id, type="supergroup", title="Friends chat")


def _private_chat(chat_id: int) -> Chat:
    return Chat(id=chat_id, type="private")


def _user(user_id: int, first_name: str = "Gene", last_name: str | None = None) -> User:
    return User(id=user_id, is_bot=False, first_name=first_name, last_name=last_name)


def _message(
    *,
    chat: Chat,
    from_user: User | None,
    message_id: int = 10,
    text: str | None = None,
    caption: str | None = None,
    photo: list[PhotoSize] | None = None,
    reply_to_message: Message | None = None,
    entities: list[MessageEntity] | None = None,
    date: datetime = NOW,
    document: Document | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        date=date,
        chat=chat,
        from_user=from_user,
        text=text,
        caption=caption,
        photo=photo,
        reply_to_message=reply_to_message,
        entities=entities,
        document=document,
    )


def _make_settings(
    monkeypatch: pytest.MonkeyPatch, *, allowed_chat_id: int, admin_user_id: int
) -> Settings:
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("ALLOWED_CHAT_ID", str(allowed_chat_id))
    monkeypatch.setenv("ADMIN_USER_ID", str(admin_user_id))
    return Settings()


@pytest.fixture
def config() -> Config:
    return demo_config()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Intercepts Message.answer so the real Bot API is never called."""
    captured: list[str] = []

    async def fake_answer(self: Message, text: str, **kwargs: object) -> None:
        captured.append(text)

    monkeypatch.setattr(Message, "answer", fake_answer)
    return captured


def _deps(
    *,
    settings: Settings,
    config: Config,
    db: FakeDb | None = None,
    responder: FakeResponder | None = None,
    memorizer: FakeMemorizer | None = None,
    weather: FakeWeather | None = None,
    changelog: str = "",
    chat_notes: str = "",
) -> tuple[FakeDeps, FakeDb, FakeConfigStore, FakePromptStore]:
    fake_db = db if db is not None else FakeDb()
    config_store = FakeConfigStore(cfg=config)
    prompt_store = FakePromptStore()
    deps = FakeDeps(
        settings=settings,
        config_store=config_store,
        prompt_store=prompt_store,
        db=fake_db,
        responder=responder,
        bot_user_id=BOT_USER_ID,
        bot_username=BOT_USERNAME,
        memorizer=memorizer,
        weather=weather,
        persona=replace(load_persona(DEMO_PACK), legacy_changelog=changelog, chat_notes=chat_notes),
    )
    return deps, fake_db, config_store, prompt_store


def _handler(deps: FakeDeps):  # type: ignore[no-untyped-def]
    router = build_commands_router(deps)  # type: ignore[arg-type]
    return router.message.handlers[0].callback


# --- /stop --------------------------------------------------------------------


async def test_stop_by_member_sets_stop_until_and_audits_silently(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(7), text="/stop")
    await handler(message)

    now = int(NOW.timestamp())
    assert db.state["stop_until"] == str(now + 86400)
    assert db.audit_stop_calls == [("stop", 7, now)]
    assert sent == []


async def test_stop_in_foreign_chat_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_chat(FOREIGN_CHAT_ID), from_user=_user(7), text="/stop")
    await handler(message)

    assert db.state == {}
    assert db.audit_stop_calls == []
    assert sent == []


# --- /panic, /resume ------------------------------------------------------------


async def test_panic_by_owner_sets_panic_and_replies(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/panic")
    await handler(message)

    assert db.state["panic"] == "1"
    assert sent == ["Panic. The bot stays silent until /resume."]
    now = int(NOW.timestamp())
    assert db.audit_stop_calls == [("panic", ADMIN_ID, now)]


async def test_panic_by_non_owner_in_private_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    other_id = 42
    message = _message(chat=_private_chat(other_id), from_user=_user(other_id), text="/panic")
    await handler(message)

    assert db.state == {}
    assert sent == []


async def test_panic_in_chat_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(ADMIN_ID), text="/panic")
    await handler(message)

    assert db.state == {}
    assert sent == []


async def test_resume_clears_panic_and_stop_until(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.state["panic"] = "1"
    db.state["stop_until"] = "123"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/resume")
    await handler(message)

    assert "panic" not in db.state
    assert "stop_until" not in db.state
    assert sent == ["Resuming. Cleared: panic, stop."]
    now = int(NOW.timestamp())
    assert db.audit_stop_calls == [("resume", ADMIN_ID, now)]


async def test_resume_with_nothing_set_replies_without_cleared_list(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """Nothing was set, so we do not tell the owner that something was cleared."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/resume")
    await handler(message)

    assert sent == ["Resuming."]
    now = int(NOW.timestamp())
    assert db.audit_stop_calls == [("resume", ADMIN_ID, now)]


async def test_resume_clears_llm_circuit_and_resets_streak(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.state["llm_circuit_until"] = str(int(NOW.timestamp()) + 1800)
    db.state["llm_error_streak"] = "5"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/resume")
    await handler(message)

    assert "llm_circuit_until" not in db.state
    assert db.state["llm_error_streak"] == "0"
    assert sent == ["Resuming. Cleared: LLM circuit breaker (5 consecutive errors)."]


async def test_resume_clears_everything_at_once(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.state["panic"] = "1"
    db.state["stop_until"] = "123"
    db.state["llm_circuit_until"] = str(int(NOW.timestamp()) + 1800)
    db.state["llm_error_streak"] = "5"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/resume")
    await handler(message)

    assert "panic" not in db.state
    assert "stop_until" not in db.state
    assert "llm_circuit_until" not in db.state
    assert db.state["llm_error_streak"] == "0"
    assert sent == ["Resuming. Cleared: panic, stop, LLM circuit breaker (5 consecutive errors)."]


async def test_resume_with_expired_circuit_key_is_still_reported_as_cleared(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """An llm_circuit_until in the past: the breaker no longer acts (llm.py compares with now),
    but the key is still in state, so it really was set: /resume clears it and mentions it in
    the answer just like an open one."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.state["llm_circuit_until"] = str(int(NOW.timestamp()) - 10)
    db.state["llm_error_streak"] = "5"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/resume")
    await handler(message)

    assert "llm_circuit_until" not in db.state
    assert db.state["llm_error_streak"] == "0"
    assert sent == ["Resuming. Cleared: LLM circuit breaker (5 consecutive errors)."]


# --- /mute, /unmute --------------------------------------------------------------


async def test_mute_without_reply_mutes_self(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """Without a reply it mutes the sender, available to any member."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(7, "Gene"), text="/mute")
    await handler(message)

    now = int(NOW.timestamp())
    assert db.add_mute_calls == [(7, "Gene", 7, now)]
    assert db.audit_stop_calls == [("mute:7", 7, now)]
    assert sent == []


async def test_mute_by_reply_from_member_on_someone_else_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """Only the owner may mute by replying to SOMEONE ELSE's message: an ordinary member
    must not get a ready-made harassment tool."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    target = _message(chat=_chat(), from_user=_user(8, "Tony", "007 🔥"), text="original")
    message = _message(
        chat=_chat(), from_user=_user(7), text="/mute", reply_to_message=target, message_id=11
    )
    await handler(message)

    assert db.add_mute_calls == []
    assert db.audit_stop_calls == []
    assert sent == []


async def test_mute_by_reply_from_owner_on_someone_else_adds_mute(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """The owner mutes another member by reply: the only one allowed to."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    target = _message(chat=_chat(), from_user=_user(8, "Tony", "007 🔥"), text="original")
    message = _message(
        chat=_chat(),
        from_user=_user(ADMIN_ID, "Owner"),
        text="/mute",
        reply_to_message=target,
        message_id=11,
    )
    await handler(message)

    now = int(NOW.timestamp())
    assert db.add_mute_calls == [(8, "Tony", ADMIN_ID, now)]
    assert db.audit_stop_calls == [("mute:8", ADMIN_ID, now)]
    assert sent == []


async def test_mute_target_owner_ignored(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    target = _message(chat=_chat(), from_user=_user(ADMIN_ID, "Owner"), text="original")
    message = _message(
        chat=_chat(),
        from_user=_user(ADMIN_ID, "Owner"),
        text="/mute",
        reply_to_message=target,
    )
    await handler(message)

    assert db.add_mute_calls == []
    assert sent == []


async def test_mute_target_bot_ignored(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    target = _message(chat=_chat(), from_user=_user(BOT_USER_ID, "Gus Marlow"), text="original")
    message = _message(
        chat=_chat(),
        from_user=_user(ADMIN_ID, "Owner"),
        text="/mute",
        reply_to_message=target,
    )
    await handler(message)

    assert db.add_mute_calls == []
    assert sent == []


async def test_unmute_without_reply_unmutes_self(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """Without a reply it unmutes the sender, available to any member."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(7, "Gene"), text="/unmute")
    await handler(message)

    now = int(NOW.timestamp())
    assert db.remove_mute_calls == [7]
    assert db.audit_stop_calls == [("unmute:7", 7, now)]
    assert sent == []


async def test_unmute_from_member_ignored_from_owner_removes(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    target = _message(chat=_chat(), from_user=_user(8, "Tony"), text="original")

    member_msg = _message(
        chat=_chat(), from_user=_user(7), text="/unmute", reply_to_message=target, message_id=20
    )
    await handler(member_msg)
    assert db.remove_mute_calls == []

    owner_msg = _message(
        chat=_chat(),
        from_user=_user(ADMIN_ID),
        text="/unmute",
        reply_to_message=target,
        message_id=21,
    )
    await handler(owner_msg)
    assert db.remove_mute_calls == [8]
    assert db.audit_stop_calls == [("unmute:8", ADMIN_ID, int(NOW.timestamp()))]
    assert sent == []


# --- /ex add ----------------------------------------------------------------------


async def test_ex_add_by_owner_reply_to_bot_adds_example(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, prompt_store = _deps(settings=settings, config=config)

    db.bot_replies_by_id[100] = FakeBotReplyRow(
        trigger="mention",
        trigger_tg_message_id=50,
        text="Same to you.",
        prompt_version=3,
        few_shot_version=2,
        delay_sec=12,
        created_at=int(NOW.timestamp()),
    )
    db.messages[(OWN_CHAT_ID, 50)] = FakeMessageRow(
        display_name="Gene", text="how are you, old man?"
    )

    handler = _handler(deps)
    bot_reply_msg = _message(
        chat=_chat(), from_user=_user(BOT_USER_ID), text="Same to you.", message_id=100
    )
    message = _message(
        chat=_chat(),
        from_user=_user(ADMIN_ID),
        text="/ex add",
        reply_to_message=bot_reply_msg,
        message_id=101,
    )
    await handler(message)

    assert prompt_store.add_example_calls == [
        ("Gene", "how are you, old man?", "Same to you.", int(NOW.timestamp()))
    ]
    assert sent == []


async def test_ex_add_without_trigger_message_uses_the_pack_fallback_names(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, prompt_store = _deps(settings=settings, config=config)
    db.bot_replies_by_id[100] = FakeBotReplyRow(
        trigger="ambient",
        trigger_tg_message_id=None,
        text="Reply text.",
        prompt_version=3,
        few_shot_version=2,
        delay_sec=0,
        created_at=int(NOW.timestamp()),
    )

    handler = _handler(deps)
    bot_reply_msg = _message(
        chat=_chat(), from_user=_user(BOT_USER_ID), text="Reply text.", message_id=100
    )
    message = _message(
        chat=_chat(),
        from_user=_user(ADMIN_ID),
        text="/ex add",
        reply_to_message=bot_reply_msg,
        message_id=101,
    )
    await handler(message)

    # The pair is few-shot data the model sees, so the wording comes from the pack fragments.
    assert prompt_store.add_example_calls == [
        (
            deps.frag.text("placeholders.ex_add_name"),
            deps.frag.text("placeholders.ex_add_user"),
            "Reply text.",
            int(NOW.timestamp()),
        )
    ]
    assert sent == []


async def test_ex_add_by_member_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, prompt_store = _deps(settings=settings, config=config)
    db.bot_replies_by_id[100] = FakeBotReplyRow(
        trigger="mention",
        trigger_tg_message_id=50,
        text="reply",
        prompt_version=1,
        few_shot_version=1,
        delay_sec=1,
        created_at=int(NOW.timestamp()),
    )
    handler = _handler(deps)

    bot_reply_msg = _message(
        chat=_chat(), from_user=_user(BOT_USER_ID), text="reply", message_id=100
    )
    message = _message(
        chat=_chat(), from_user=_user(7), text="/ex add", reply_to_message=bot_reply_msg
    )
    await handler(message)

    assert prompt_store.add_example_calls == []
    assert sent == []


# --- /set, /unset -----------------------------------------------------------------


async def test_set_calls_store_and_replies_with_old_and_new(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, config_store, _ = _deps(settings=settings, config=config)
    config_store.set_result = ("3", "5")
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID),
        from_user=_user(ADMIN_ID),
        text="/set behaviour.daily_cap 5",
    )
    await handler(message)

    assert config_store.set_calls == [("behaviour.daily_cap", "5", ADMIN_ID, int(NOW.timestamp()))]
    assert sent == ["behaviour.daily_cap: 3 → 5"]


async def test_set_value_error_replies_with_error_text(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, config_store, _ = _deps(settings=settings, config=config)
    config_store.set_error = ValueError("ambient_probability must be <= 1.0")
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID),
        from_user=_user(ADMIN_ID),
        text="/set behaviour.ambient_probability 5",
    )
    await handler(message)

    assert sent == ["Error: ambient_probability must be <= 1.0"]


async def test_set_without_args_replies_with_usage(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, config_store, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/set")
    await handler(message)

    assert config_store.set_calls == []
    assert sent == ["Usage: /set <key> <value>. Keys: /get, description: /get <key>"]


async def test_unset_replies_with_default(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, config_store, _ = _deps(settings=settings, config=config)
    config_store.unset_result = "3"
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/unset behaviour.daily_cap"
    )
    await handler(message)

    assert config_store.unset_calls == [("behaviour.daily_cap", ADMIN_ID, int(NOW.timestamp()))]
    assert sent == ["behaviour.daily_cap: reset to 3"]


# --- /get ---------------------------------------------------------------------


async def test_get_all_and_with_prefix(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/get"))
    assert len(sent) == 1
    assert "behaviour.daily_cap: 3" in sent[0]
    assert "llm.main_model: openrouter/x" in sent[0]

    await handler(
        _message(
            chat=_private_chat(ADMIN_ID),
            from_user=_user(ADMIN_ID),
            text="/get behaviour.",
            message_id=2,
        )
    )
    assert "behaviour.daily_cap: 3" in sent[1]
    assert "behaviour.ambient_probability" in sent[1]
    assert "llm.main_model" not in sent[1]

    hint = "* - overridden with /set. Key description: /get <key>"
    assert sent[0].endswith(hint)
    assert sent[1].endswith(hint)


async def test_get_exact_key_shows_card(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, config_store, _ = _deps(settings=settings, config=config)
    config_store.describe_value["behaviour.daily_cap"] = KeyInfo(
        key="behaviour.daily_cap",
        value="3",
        default="3",
        overridden=False,
        type_name="int",
        bounds="0..50",
        description="how many times a day the bot may chime in unprompted",
        settable=True,
    )
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/get behaviour.daily_cap"
    )
    await handler(message)

    assert sent == [
        "behaviour.daily_cap: 3\n"
        "type: int, 0..50\n"
        "how many times a day the bot may chime in unprompted"
    ]


async def test_get_exact_key_card_overridden_and_not_settable(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, config_store, _ = _deps(settings=settings, config=config)
    config_store.describe_value["persona.name"] = KeyInfo(
        key="persona.name",
        value="Gus",
        default="Gus the Second",
        overridden=True,
        type_name="str",
        bounds="",
        description="",
        settable=False,
    )
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/get persona.name"
    )
    await handler(message)

    assert sent == [
        "persona.name: Gus\n"
        "type: str\n"
        "overridden, yaml value: Gus the Second\n"
        "can only be changed in config.yaml"
    ]


# --- /last ------------------------------------------------------------------------


async def test_last_format(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    created_at = int(datetime(2026, 1, 10, 10, 30, tzinfo=UTC).timestamp())
    db.last_replies_value = [
        FakeBotReplyRow(
            trigger="mention",
            trigger_tg_message_id=50,
            text="Hello",
            prompt_version=3,
            few_shot_version=2,
            delay_sec=45,
            created_at=created_at,
        )
    ]
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/last 3")
    await handler(message)

    expected_time = local_dt(created_at, config.persona.timezone).strftime("%H:%M %d.%m")
    assert sent == [f"{expected_time} | mention | p3/f2 | +45s | Hello"]


# --- /why -------------------------------------------------------------------------


async def test_why_computes_since_from_hours(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.filter_log_summary_value = [("gate:not_live", 14), ("regex:length", 2)]
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/why 2")
    await handler(message)

    now = int(NOW.timestamp())
    assert db.filter_log_summary_calls == [now - 2 * 3600]
    assert sent == ["gate:not_live 14\nregex:length 2"]


# --- /prompt ------------------------------------------------------------------


async def test_prompt_preview_and_full(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, prompt_store = _deps(settings=settings, config=config)
    prompt_store.system_prompt_value = "A" * 2000
    handler = _handler(deps)

    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/prompt"))
    assert "version 3 (active), 3 in total" in sent[0]
    assert sent[0].endswith("…")
    assert len(sent[0]) < len(prompt_store.system_prompt_value)

    await handler(
        _message(
            chat=_private_chat(ADMIN_ID),
            from_user=_user(ADMIN_ID),
            text="/prompt full",
            message_id=2,
        )
    )
    assert prompt_store.system_prompt_value in sent[1]


# --- /rollback ----------------------------------------------------------------


async def test_rollback(monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, prompt_store = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/rollback 1")
    await handler(message)

    assert prompt_store.rollback_calls == [1]
    assert sent == ["prompt: version 1"]


# --- /ex last, /ex rm --------------------------------------------------------------


async def test_ex_last(monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, prompt_store = _deps(settings=settings, config=config)
    prompt_store.examples_value = [
        FewShot(name="Gene", user="how are you?", speak=True, text="Fine."),
        FewShot(name="Tony", user="where is the pub?", speak=True, text="Down by the bridge."),
    ]
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/ex last 2")
    await handler(message)

    assert sent == [
        'Gene: how are you? → {"speak": true, "text": "Fine."}\n'
        'Tony: where is the pub? → {"speak": true, "text": "Down by the bridge."}'
    ]


async def test_ex_rm(monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, prompt_store = _deps(settings=settings, config=config)
    prompt_store.remove_example_result = 9
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/ex rm")
    await handler(message)

    assert prompt_store.remove_example_calls == [(1, int(NOW.timestamp()))]
    assert sent == ["few-shot: version 9 (example 1 from the end removed)"]


# --- /life --------------------------------------------------------------------


async def test_life_add_empty_text_replies_usage(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life")
    await handler(message)

    assert sent == ["Usage: /life <text> | list | rm N | post N"]
    assert db.life_events_store == {}
    assert db.audit_stop_calls == []


async def test_life_add_without_responder_only_records(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config, responder=None)
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID),
        from_user=_user(ADMIN_ID),
        text="/life sold the old van, got another one",
    )
    await handler(message)

    now = int(NOW.timestamp())
    assert list(db.life_events_store.values()) == [
        LifeEventRow(
            id=1,
            text="sold the old van, got another one",
            created_at=now,
            announced_at=None,
            announced_tg_message_id=None,
        )
    ]
    assert db.audit_stop_calls == [("life:add", ADMIN_ID, now)]
    assert db.audit_values["life:add"] == "sold the old van, got another one"
    assert sent == ["Saved #1. The LLM is not configured, nothing was sent to the chat."]


async def test_life_add_with_responder_sent(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder(
        announce_life_result=SendOutcome(
            sent=True, text="Sold the cart, got a Hyundai.", reason="send:life"
        )
    )
    deps, db, _, _ = _deps(settings=settings, config=config, responder=responder)
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID),
        from_user=_user(ADMIN_ID),
        text="/life sold the cart, got a Hyundai",
    )
    await handler(message)

    now = int(NOW.timestamp())
    assert len(responder.announce_life_calls) == 1
    assert responder.announce_life_calls[0].id == 1
    assert responder.announce_life_calls[0].text == "sold the cart, got a Hyundai"
    assert db.audit_stop_calls == [("life:add", ADMIN_ID, now)]
    assert sent == ["Saved #1. Sent: Sold the cart, got a Hyundai."]


async def test_life_add_blocked_by_panic(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder(
        announce_life_result=SendOutcome(sent=False, text="", reason="blocked:panic")
    )
    deps, _, _, _ = _deps(settings=settings, config=config, responder=responder)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life news")
    await handler(message)

    assert sent == ["Saved #1. The bot is silent (panic/stop) - /resume."]


async def test_life_add_not_sent_shows_reason_and_candidate(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder(
        announce_life_result=SendOutcome(sent=False, text="Sold the van.", reason="regex:sentences")
    )
    deps, _, _, _ = _deps(settings=settings, config=config, responder=responder)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life news")
    await handler(message)

    assert sent == ["Saved #1. Not sent: regex:sentences\nCandidate: Sold the van."]


async def test_life_add_not_sent_without_candidate_text(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder(
        announce_life_result=SendOutcome(sent=False, text="", reason="llm:silent")
    )
    deps, _, _, _ = _deps(settings=settings, config=config, responder=responder)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life news")
    await handler(message)

    assert sent == ["Saved #1. Not sent: llm:silent"]


async def test_life_list_empty(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life list")
    await handler(message)

    assert sent == ["No events."]


async def test_life_list_formats_rows_oldest_first(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.life_events_store[1] = LifeEventRow(
        id=1,
        text="sold the cart",
        created_at=int(datetime(2026, 1, 5, 9, 0, tzinfo=UTC).timestamp()),
        announced_at=int(datetime(2026, 1, 5, 9, 1, tzinfo=UTC).timestamp()),
        announced_tg_message_id=42,
    )
    db.life_events_store[2] = LifeEventRow(
        id=2,
        text="got a Hyundai",
        created_at=int(datetime(2026, 1, 9, 9, 0, tzinfo=UTC).timestamp()),
        announced_at=None,
        announced_tg_message_id=None,
    )
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life list")
    await handler(message)

    assert sent == ["#1 05.01.2026 ✓ sold the cart\n#2 09.01.2026 - got a Hyundai"]


async def test_life_rm_deletes_existing_event(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.life_events_store[3] = LifeEventRow(
        id=3, text="sold the van", created_at=1, announced_at=None, announced_tg_message_id=None
    )
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life rm 3")
    await handler(message)

    now = int(NOW.timestamp())
    assert 3 not in db.life_events_store
    assert db.audit_stop_calls == [("life:rm", ADMIN_ID, now)]
    assert sent == ["Event #3 deleted."]


async def test_life_rm_missing_event(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life rm 9")
    await handler(message)

    assert sent == ["No event #9."]
    assert db.audit_stop_calls == []


async def test_life_rm_without_number_replies_usage(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life rm")
    await handler(message)

    assert sent == ["Usage: /life <text> | list | rm N | post N"]


async def test_life_post_resends_existing_event_without_zapisal_prefix(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder(
        announce_life_result=SendOutcome(sent=True, text="Sold the van.", reason="send:life")
    )
    deps, db, _, _ = _deps(settings=settings, config=config, responder=responder)
    db.life_events_store[4] = LifeEventRow(
        id=4, text="sold the van", created_at=1, announced_at=None, announced_tg_message_id=None
    )
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life post 4")
    await handler(message)

    now = int(NOW.timestamp())
    assert len(responder.announce_life_calls) == 1
    assert responder.announce_life_calls[0].id == 4
    assert db.audit_stop_calls == [("life:post", ADMIN_ID, now)]
    assert sent == ["Sent: Sold the van."]


async def test_life_post_missing_event(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life post 9")
    await handler(message)

    assert sent == ["No event #9."]


async def test_life_post_without_responder(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config, responder=None)
    db.life_events_store[5] = LifeEventRow(
        id=5, text="sold the van", created_at=1, announced_at=None, announced_tg_message_id=None
    )
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/life post 5")
    await handler(message)

    assert sent == ["The LLM is not configured, nothing was sent to the chat."]


async def test_life_command_in_chat_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(ADMIN_ID), text="/life news")
    await handler(message)

    assert db.life_events_store == {}
    assert sent == []


# --- /say ---------------------------------------------------------------------


async def test_say_empty_text_replies_usage(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/say")
    await handler(message)

    assert sent == ["Usage: /say <text>"]
    assert db.audit_stop_calls == []


async def test_say_without_responder(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config, responder=None)
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/say Hello everyone"
    )
    await handler(message)

    now = int(NOW.timestamp())
    assert db.audit_stop_calls == [("say", ADMIN_ID, now)]
    assert sent == ["The LLM part is off, /say is unavailable."]


async def test_say_sends_text_verbatim(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder(say_result=SendOutcome(sent=True, text="", reason="send:say"))
    deps, db, _, _ = _deps(settings=settings, config=config, responder=responder)
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID),
        from_user=_user(ADMIN_ID),
        text="/say Hello   everyone,   how are you?",
    )
    await handler(message)

    now = int(NOW.timestamp())
    assert responder.say_calls == ["Hello   everyone,   how are you?"]
    assert db.audit_stop_calls == [("say", ADMIN_ID, now)]
    assert sent == ["Sent."]


async def test_say_blocked_by_stop(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder(say_result=SendOutcome(sent=False, text="", reason="blocked:stop"))
    deps, _, _, _ = _deps(settings=settings, config=config, responder=responder)
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/say Hello everyone"
    )
    await handler(message)

    assert sent == ["The bot is silent (panic/stop) - /resume."]


async def test_say_command_in_chat_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(ADMIN_ID), text="/say Hello everyone")
    await handler(message)

    assert db.audit_stop_calls == []
    assert sent == []


# --- /status ------------------------------------------------------------------


async def test_status_contains_versions_and_models(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    llm = config.llm.model_copy(update={"main_model": "openrouter/main", "judge_model": "j/x"})
    cfg = config.model_copy(update={"llm": llm})
    deps, _, _, prompt_store = _deps(settings=settings, config=cfg)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert len(sent) == 1
    text = sent[0]
    assert f"v{prompt_store.prompt_version_value}" in text
    assert f"v{prompt_store.few_shot_version_value}" in text
    assert "openrouter/main" in text
    assert "j/x" in text


async def test_status_shows_circuit_closed_and_zero_streak_by_default(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert len(sent) == 1
    text = sent[0]
    assert "LLM circuit breaker: closed, consecutive errors: 0" in text


async def test_status_shows_circuit_open_until_local_time_and_streak(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    now = int(NOW.timestamp())
    circuit_until = now + 1800
    db.state["llm_circuit_until"] = str(circuit_until)
    db.state["llm_error_streak"] = "5"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert len(sent) == 1
    text = sent[0]
    expected_time = local_dt(circuit_until, config.persona.timezone).strftime("%H:%M")
    assert f"LLM circuit breaker: open until {expected_time}, consecutive errors: 5" in text


async def test_status_shows_circuit_closed_when_key_expired(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """An llm_circuit_until in the past: the breaker released by itself on the timer (the same
    check as in llm.py: `> now`); /status must not claim it is still open even if the key
    has not been cleaned up yet."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.state["llm_circuit_until"] = str(int(NOW.timestamp()) - 10)
    db.state["llm_error_streak"] = "5"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert len(sent) == 1
    text = sent[0]
    assert "LLM circuit breaker: closed, consecutive errors: 5" in text


async def test_status_shows_last_provider_refusal_for_a_day(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """Cheap models reset the consecutive-error streak while the main model kept getting 402;
    /status shows the last provider refusal separately, for a day after it."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    now = int(NOW.timestamp())
    db.state["llm_last_refusal"] = f"{now - 3600}|402"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    expected = local_dt(now - 3600, config.persona.timezone).strftime("%H:%M %d.%m")
    assert f"last provider refusal: 402 at {expected}" in sent[0]

    db.state["llm_last_refusal"] = f"{now - 86400}|402"
    await handler(message)
    assert "last provider refusal: none in 24 h" in sent[1]


async def test_status_shows_stop_only_while_active(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """A stop_until in the past: the gate no longer applies the stop (it compares with now)
    while nobody deletes the key; /status must not show an expired stop (a real case: "the
    bot shows a stop although it has expired"). An active one is shown."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)
    now = int(NOW.timestamp())

    db.state["stop_until"] = str(now - 10)
    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status"))
    assert "Stopped until: none" in sent[0]

    db.state["stop_until"] = str(now + 3600)
    await handler(
        _message(
            chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status", message_id=2
        )
    )
    assert "Stopped until: none" not in sent[1]
    assert "Stopped until: " in sent[1]


async def test_status_shows_life_events_total_and_unsent(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.life_events_store[1] = LifeEventRow(
        id=1, text="sold the van", created_at=1, announced_at=1, announced_tg_message_id=42
    )
    db.life_events_store[2] = LifeEventRow(
        id=2, text="got another one", created_at=2, announced_at=None, announced_tg_message_id=None
    )
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert len(sent) == 1
    assert "life events: 2 (1)" in sent[0]


async def test_status_shows_no_life_events_by_default(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "life events: 0 (0)" in sent[0]


async def test_status_shows_hot_window_none_by_default(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "hot window: none" in sent[0]


async def test_status_shows_hot_window_open_until_local_time_and_count(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    now = int(NOW.timestamp())
    hot_until = now + 900
    db.state["hot_until"] = str(hot_until)
    db.state["hot_ambient_count"] = "2"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    expected_time = local_dt(hot_until, config.persona.timezone).strftime("%H:%M")
    cap = config.behaviour.hot_window.ambient_cap
    assert f"hot window: until {expected_time} (2/{cap})" in sent[0]


async def test_status_shows_hot_window_none_when_expired(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """A hot_until in the past: /status must not claim the window is still open."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    now = int(NOW.timestamp())
    db.state["hot_until"] = str(now - 10)
    db.state["hot_ambient_count"] = "1"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "hot window: none" in sent[0]


async def test_status_shows_followup_calls_zero_by_default(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    cap = config.behaviour.followup.daily_cap
    assert f"followup calls: 0/{cap}" in sent[0]


async def test_status_shows_followup_calls_today_count(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    now = int(NOW.timestamp())
    db.state[day_key("followup_calls", now, config.persona.timezone)] = "7"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    cap = config.behaviour.followup.daily_cap
    assert f"followup calls: 7/{cap}" in sent[0]


async def test_status_shows_reaction_semantic_calls(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """Reactions are counted with two numbers: how many were placed and how many model calls
    they cost (their own cap, semantic_daily_cap)."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    now = int(NOW.timestamp())
    tz = config.persona.timezone
    db.state[day_key("reaction_count", now, tz)] = "2"
    db.state[day_key("reaction_calls", now, tz)] = "5"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    reactions_cfg = config.behaviour.reactions
    assert f"reactions=2/{reactions_cfg.daily_cap}" in sent[0]
    assert f"semantic calls 5/{reactions_cfg.semantic_daily_cap}" in sent[0]


async def test_status_shows_zero_reaction_semantic_calls_by_default(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    cap = config.behaviour.reactions.semantic_daily_cap
    assert f"semantic calls 0/{cap}" in sent[0]


async def test_status_shows_vision_zero_by_default(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    cap = config.behaviour.vision.daily_cap
    assert f"vision: 0/{cap}" in sent[0]


async def test_status_shows_vision_today_count(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    now = int(NOW.timestamp())
    db.state[day_key("vision_count", now, config.persona.timezone)] = "4"
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    cap = config.behaviour.vision.daily_cap
    assert f"vision: 4/{cap}" in sent[0]


async def test_status_shows_checkin_none_by_default(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "checkin: none" in sent[0]


async def test_status_shows_presence_allowance_and_counts(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """Presence cap: 20 human messages with max_share 0.15 and free_replies 2 give an
    allowance of 5."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.messages_today_value = 20
    db.bot_replies_today_value = 3
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "presence: 3/5 (people 20)" in sent[0]


async def test_status_shows_presence_free_replies_in_empty_chat(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "presence: 0/2 (people 0)" in sent[0]


async def test_status_shows_checkin_due_local_time(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    now = int(NOW.timestamp())
    due = now + 3600
    db.state["checkin_due"] = str(due)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    expected_time = local_dt(due, config.persona.timezone).strftime("%H:%M")
    assert f"checkin: due {expected_time}" in sent[0]


async def test_status_shows_no_weather_without_client(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """No client at all (or no coordinates set): "weather: none"."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "weather: none" in sent[0]


async def test_status_shows_no_weather_when_snapshot_is_none(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    weather = FakeWeather(snapshot=None, home_name="Brindlemoor")
    deps, _, _, _ = _deps(settings=settings, config=config, weather=weather)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "weather: none" in sent[0]
    assert weather.calls == 1


async def test_status_shows_weather_and_fetch_time(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    fetched_at = int(NOW.timestamp()) - 600
    snapshot = Weather(
        temp_now=9.4,
        code_now=3,
        today_min=4.2,
        today_max=11.6,
        today_code=3,
        tomorrow_min=-2.6,
        tomorrow_max=3.4,
        tomorrow_code=61,
        fetched_at=fetched_at,
    )
    weather = FakeWeather(snapshot=snapshot, home_name="Brindlemoor")
    deps, _, _, _ = _deps(settings=settings, config=config, weather=weather)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    expected_time = local_dt(fetched_at, config.persona.timezone).strftime("%H:%M")
    assert f"weather: +9, overcast, updated {expected_time}" in sent[0]


async def test_status_weather_without_home_name(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    snapshot = Weather(
        temp_now=-0.4,
        code_now=4242,  # unknown code -> no description
        today_min=-3.0,
        today_max=1.0,
        today_code=3,
        tomorrow_min=-5.0,
        tomorrow_max=0.0,
        tomorrow_code=71,
        fetched_at=int(NOW.timestamp()),
    )
    deps, _, _, _ = _deps(settings=settings, config=config, weather=FakeWeather(snapshot=snapshot))
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "weather: 0, updated" in sent[0]


# --- help ------------------------------------------------------------------


async def test_unknown_command_replies_help(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/frobnicate")
    await handler(message)

    assert len(sent) == 1
    assert "/status" in sent[0]
    assert "/panic" in sent[0]
    assert "/stop" in sent[0]
    assert "/mute" in sent[0]
    assert "/life" in sent[0]
    assert "/say" in sent[0]
    assert "/help" in sent[0]


async def test_help_command_replies_with_same_help_text(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    unknown_msg = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/frobnicate"
    )
    await handler(unknown_msg)

    help_msg = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/help", message_id=2
    )
    await handler(help_msg)

    assert len(sent) == 2
    assert sent[0] == sent[1]
    assert "/stop" in sent[1]
    assert "/mute" in sent[1]


# --- @bot_username suffix -----------------------------------------------------


async def test_command_with_bot_suffix_works(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(7), text=f"/stop@{BOT_USERNAME}")
    await handler(message)

    assert db.state.get("stop_until") == str(int(NOW.timestamp()) + 86400)


async def test_command_with_wrong_bot_suffix_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """An @other_bot suffix in a group broadcasts the command to every bot in the chat: we
    only execute the one addressed to us."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(7), text="/stop@weatherbot")
    await handler(message)

    assert db.state == {}
    assert db.audit_stop_calls == []
    assert sent == []


# --- command in a photo caption ---------------------------------------------------


async def test_panic_command_as_photo_caption_from_owner(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    photo = [PhotoSize(file_id="f1", file_unique_id="u1", width=100, height=100)]
    message = _message(
        chat=_private_chat(ADMIN_ID),
        from_user=_user(ADMIN_ID),
        caption="/panic",
        photo=photo,
    )
    await handler(message)

    assert db.state["panic"] == "1"
    assert sent == ["Panic. The bot stays silent until /resume."]


# --- truncation of a long answer ----------------------------------------------------


async def test_long_reply_is_truncated_to_3500(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, prompt_store = _deps(settings=settings, config=config)
    prompt_store.system_prompt_value = "Z" * 5000
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/prompt full")
    await handler(message)

    assert len(sent) == 1
    assert len(sent[0]) == 3500
    assert sent[0].endswith("…")


# --- /mute via a text_mention entity --------------------------------------------


async def test_mute_via_text_mention_entity_by_owner(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """A text_mention is also "the target is someone else", so it is owner-only."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    mentioned = _user(9, "Olly")
    entity = MessageEntity(type="text_mention", offset=6, length=3, user=mentioned)
    message = _message(
        chat=_chat(),
        from_user=_user(ADMIN_ID, "Owner"),
        text="/mute Olly",
        entities=[entity],
    )
    await handler(message)

    assert db.add_mute_calls == [(9, "Olly", ADMIN_ID, int(NOW.timestamp()))]


async def test_mute_via_text_mention_entity_by_member_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    mentioned = _user(9, "Olly")
    entity = MessageEntity(type="text_mention", offset=6, length=3, user=mentioned)
    message = _message(chat=_chat(), from_user=_user(7), text="/mute Olly", entities=[entity])
    await handler(message)

    assert db.add_mute_calls == []


# --- /memory --------------------------------------------------------------------


def _memory_row(memory_id: int, text: str) -> ChatMemoryRow:
    """The period 08.09-14.09.2026 in the persona timezone."""
    start = int(datetime(2026, 9, 8, tzinfo=ZoneInfo("Europe/London")).timestamp())
    end = int(datetime(2026, 9, 15, tzinfo=ZoneInfo("Europe/London")).timestamp())
    return ChatMemoryRow(
        id=memory_id,
        period_start=start + (memory_id - 1) * 7 * 86400,
        period_end=end + (memory_id - 1) * 7 * 86400,
        text=text,
        created_at=end,
    )


async def test_memory_list_empty(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/memory")
    await handler(message)

    assert sent == ["No memories yet."]


async def test_memory_list_shows_period_and_text_oldest_first(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.chat_memory_store[1] = _memory_row(1, "Mike was boasting about his scooter")
    db.chat_memory_store[2] = _memory_row(2, "They were planning a fishing trip")
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/memory list")
    await handler(message)

    assert len(sent) == 1
    text = sent[0]
    assert text.startswith("#1 08.09–14.09.2026\nMike was boasting about his scooter")
    assert "#2 15.09–21.09.2026\nThey were planning a fishing trip" in text


async def test_memory_rm_deletes_and_audits(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.chat_memory_store[1] = _memory_row(1, "Mike was boasting about his scooter")
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/memory rm 1")
    await handler(message)

    assert sent == ["Summary #1 deleted."]
    assert db.chat_memory_store == {}
    assert db.audit_stop_calls == [("memory:rm", ADMIN_ID, int(NOW.timestamp()))]
    assert db.audit_values["memory:rm"] == "Mike was boasting about his scooter"


async def test_memory_rm_unknown_id(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/memory rm 7")
    await handler(message)

    assert sent == ["No summary #7."]


async def test_memory_rm_without_number_replies_usage(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/memory rm")
    await handler(message)

    assert sent == ["Usage: /memory [list | rm N | run]"]


async def test_memory_run_reports_created_rows(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    memorizer = FakeMemorizer(
        run_due_result=[_memory_row(1, "Mike was boasting about his scooter")]
    )
    deps, db, _, _ = _deps(settings=settings, config=config, memorizer=memorizer)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/memory run")
    await handler(message)

    assert memorizer.run_due_calls == [int(NOW.timestamp())]
    assert len(sent) == 1
    assert sent[0].startswith("Summaries added: 1")
    assert "#1 08.09–14.09.2026\nMike was boasting about his scooter" in sent[0]
    assert db.audit_stop_calls == [("memory:run", ADMIN_ID, int(NOW.timestamp()))]


async def test_memory_run_nothing_to_summarize(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    memorizer = FakeMemorizer()
    deps, _, _, _ = _deps(settings=settings, config=config, memorizer=memorizer)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/memory run")
    await handler(message)

    assert sent == ["Nothing to summarize."]


async def test_memory_run_without_memorizer(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/memory run")
    await handler(message)

    assert sent == ["The LLM part is off, summaries are unavailable."]
    assert db.audit_stop_calls == []


async def test_memory_unknown_subcommand_replies_usage(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/memory forget it all"
    )
    await handler(message)

    assert sent == ["Usage: /memory [list | rm N | run]"]


async def test_memory_command_in_chat_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    """An owner command works in private only; the chat never sees the memory."""
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.chat_memory_store[1] = _memory_row(1, "Mike was boasting about his scooter")
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(ADMIN_ID), text="/memory list")
    await handler(message)

    assert sent == []


async def test_help_mentions_memory_command(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/help")
    await handler(message)

    assert "/memory" in sent[0]


async def test_status_shows_no_chat_memory_by_default(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "chat memory: none" in sent[0]


async def test_status_shows_chat_memory_total_and_last_period_end(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.chat_memory_store[1] = _memory_row(1, "Mike was boasting about his scooter")
    db.chat_memory_store[2] = _memory_row(2, "They were planning a fishing trip")
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert "chat memory: 2, last up to 22.09.2026" in sent[0]


# --- /changelog -------------------------------------------------------------


# The legacy CHANGELOG.md format: Russian block headers.
_CHAT = "### For the chat"
_OWNER = "### For the owner"
_CHANGELOG_SAMPLE = f"""# Changelog

## [Unreleased]

{_CHAT}

{_OWNER}

## [0.5.0] — 2026-09-16

{_CHAT}

- Gus now talks noticeably less often.

{_OWNER}

- Presence cap `behaviour.presence` (#12).

## [0.4.0] — 2026-09-15

{_CHAT}

- Gus can see photos.

{_OWNER}

- `/life`, `/say` (#10).

[Unreleased]: https://example.com/compare/v0.5.0...HEAD
[0.5.0]: https://example.com/releases/tag/v0.5.0
[0.4.0]: https://example.com/compare/v0.4.0...v0.5.0
"""


async def test_changelog_bare_returns_latest_for_chat(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config, changelog=_CHANGELOG_SAMPLE)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/changelog")
    await handler(message)

    assert sent == ["v0.5.0 (2026-09-16)\n- Gus now talks noticeably less often."]


async def test_changelog_owner_returns_latest_for_owner(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config, changelog=_CHANGELOG_SAMPLE)
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/changelog owner"
    )
    await handler(message)

    assert sent == ["v0.5.0 (2026-09-16)\n- Presence cap `behaviour.presence` (#12)."]


@pytest.mark.parametrize("version_arg", ["0.4.0", "v0.4.0"])
async def test_changelog_specific_version_accepts_v_prefix(
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
    sent: list[str],
    version_arg: str,
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config, changelog=_CHANGELOG_SAMPLE)
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text=f"/changelog {version_arg}"
    )
    await handler(message)

    assert sent == ["v0.4.0 (2026-09-15)\n- Gus can see photos."]


async def test_changelog_unknown_version_replies_not_found(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config, changelog=_CHANGELOG_SAMPLE)
    handler = _handler(deps)

    message = _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/changelog 9.9.9"
    )
    await handler(message)

    assert sent == ["No version 9.9.9."]


async def test_changelog_missing_file_replies_not_found(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config, changelog="")
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/changelog")
    await handler(message)

    assert sent == ["CHANGELOG.md not found or empty."]


async def test_changelog_ignored_in_chat(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config, changelog=_CHANGELOG_SAMPLE)
    handler = _handler(deps)

    message = _message(chat=_chat(), from_user=_user(ADMIN_ID), text="/changelog")
    await handler(message)

    assert sent == []


async def test_status_starts_with_version(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status")
    await handler(message)

    assert sent[0].startswith("sotto v")


# --- /diary ------------------------------------------------------------------


def _fact(fact_id: int, text: str, created_at: datetime) -> SelfFactRow:
    return SelfFactRow(
        id=fact_id,
        text=text,
        bot_reply_tg_message_id=None,
        created_at=int(created_at.timestamp()),
    )


async def test_diary_list_empty(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/diary")
    await handler(message)

    assert sent == ["No facts."]


async def test_diary_list_shows_week_in_local_time(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.self_facts_store[1] = _fact(1, "went to the sauna", datetime(2026, 1, 10, 9, 46, tzinfo=UTC))
    db.self_facts_store[2] = _fact(2, "an old fact", datetime(2025, 12, 1, 9, 0, tzinfo=UTC))
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/diary")
    await handler(message)

    assert sent == ["#1 10.01 09:46 went to the sauna"]


async def test_diary_rm_deletes_and_audits(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.self_facts_store[1] = _fact(1, "went to the sauna", NOW)
    handler = _handler(deps)

    message = _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/diary rm 1")
    await handler(message)

    assert sent == ["Fact #1 deleted."]
    assert db.self_facts_store == {}
    assert db.audit_stop_calls == [("diary:rm", ADMIN_ID, int(NOW.timestamp()))]
    assert db.audit_values["diary:rm"] == "went to the sauna"


async def test_diary_rm_unknown_and_bad_args(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    for text in ("/diary rm 7", "/diary rm", "/diary something"):
        await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text=text))

    usage = "Usage: /diary | /diary rm N"
    assert sent == ["No fact #7.", usage, usage]


async def test_diary_command_in_chat_does_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.self_facts_store[1] = _fact(1, "went to the sauna", NOW)
    handler = _handler(deps)

    await handler(_message(chat=_chat(), from_user=_user(ADMIN_ID), text="/diary"))

    assert sent == []


async def test_help_mentions_diary_and_fits_limit(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/help"))

    assert "/diary" in sent[0]
    assert len(sent[0]) <= 3500


async def test_status_shows_diary_line(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.self_facts_store[1] = _fact(1, "went to the sauna", NOW)
    db.state[day_key("diary_calls", int(NOW.timestamp()), "Europe/London")] = "4"
    handler = _handler(deps)

    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status"))

    cap = config.behaviour.diary.daily_cap
    assert f"diary: 1, calls 4/{cap}" in sent[0]


async def test_status_shows_self_share_line(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    now = int(NOW.timestamp())

    def row(trigger: str, text: str, created_at: int) -> FakeBotReplyRow:
        return FakeBotReplyRow(
            trigger=trigger,
            trigger_tg_message_id=None,
            text=text,
            prompt_version=1,
            few_shot_version=1,
            delay_sec=0,
            created_at=created_at,
        )

    db.bot_replies_store = [
        row("ambient", "We argued just like that in the lab", now - 3600),
        row("ambient", "A third time is already a tradition", now - 3500),
        row("name", "My grandson will not eat garlic", now - 3400),
        row("morning", "Good morning all", now - 3300),
        row("say", "I am off to the sauna", now - 3200),
        row("ambient", "[sticker #3] Well, look at you", now - 3100),
        row("ambient", "I keep those in a jar", now - 4 * 86400),  # outside the window
    ]
    handler = _handler(deps)

    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status"))

    assert "about himself over 3 days: ambient 1/2, addressed 1/1, other 0/1" in sent[0]


# --- /jokes and /people -----------------------------------------------------


def _joke(joke_id: int, text: str, created_at: datetime, used: datetime | None = None) -> JokeRow:
    return JokeRow(
        id=joke_id,
        text=text,
        created_at=int(created_at.timestamp()),
        last_used_at=int(used.timestamp()) if used is not None else None,
        uses=1 if used is not None else 0,
    )


def _thread(
    thread_id: int, name: str, text: str, created_at: datetime, asked: datetime | None = None
) -> PeopleThreadRow:
    return PeopleThreadRow(
        id=thread_id,
        user_id=100 + thread_id,
        display_name=name,
        text=text,
        created_at=int(created_at.timestamp()),
        asked_at=int(asked.timestamp()) if asked is not None else None,
        closed=asked is not None,
    )


async def _owner_say(handler: Any, text: str) -> None:
    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text=text))


async def test_jokes_list_empty_and_newest_first(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    await _owner_say(handler, "/jokes")
    assert sent == ["No jokes yet."]

    db.jokes_store[1] = _joke(1, "screws again", datetime(2026, 1, 1, tzinfo=UTC))
    db.jokes_store[2] = _joke(
        2,
        "like Vince's",
        datetime(2026, 1, 5, tzinfo=UTC),
        used=datetime(2026, 1, 9, 9, tzinfo=UTC),
    )
    sent.clear()
    await _owner_say(handler, "/jokes")

    assert sent == ["#2 like Vince's (used 09.01)\n#1 screws again (used -)"]


async def test_jokes_rm_deletes_audits_and_bad_args(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.jokes_store[1] = _joke(1, "screws again", NOW)
    handler = _handler(deps)

    for text in ("/jokes rm 7", "/jokes rm", "/jokes something", "/jokes rm 1"):
        await _owner_say(handler, text)

    usage = "Usage: /jokes | /jokes rm N"
    assert sent == ["No joke #7.", usage, usage, "Joke #1 deleted."]
    assert db.jokes_store == {}
    assert db.audit_stop_calls == [("jokes:rm", ADMIN_ID, int(NOW.timestamp()))]
    assert db.audit_values["jokes:rm"] == "screws again"


async def test_people_list_empty_and_statuses(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    await _owner_say(handler, "/people")
    assert sent == ["No stories yet."]

    db.threads_store[1] = _thread(
        1,
        "Mike",
        "awaits audition news",
        datetime(2026, 1, 2, tzinfo=UTC),
        asked=datetime(2026, 1, 8, 12, tzinfo=UTC),
    )
    db.threads_store[2] = _thread(2, "Julia", "wants a boat", datetime(2026, 1, 3, tzinfo=UTC))
    sent.clear()
    await _owner_say(handler, "/people")

    assert sent == ["#1 Mike: awaits audition news (asked 08.01)\n#2 Julia: wants a boat (open)"]


async def test_people_rm_deletes_audits_and_bad_args(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.threads_store[1] = _thread(1, "Julia", "wants a boat", NOW)
    handler = _handler(deps)

    for text in ("/people rm 9", "/people rm", "/people x", "/people rm 1"):
        await _owner_say(handler, text)

    usage = "Usage: /people | /people rm N"
    assert sent == ["No story #9.", usage, usage, "Story #1 deleted."]
    assert db.threads_store == {}
    assert db.audit_stop_calls == [("people:rm", ADMIN_ID, int(NOW.timestamp()))]
    assert db.audit_values["people:rm"] == "wants a boat"


async def test_jokes_and_people_in_chat_do_nothing(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    db.jokes_store[1] = _joke(1, "screws again", NOW)
    handler = _handler(deps)

    await handler(_message(chat=_chat(), from_user=_user(ADMIN_ID), text="/jokes"))
    await handler(_message(chat=_chat(), from_user=_user(ADMIN_ID), text="/people"))

    assert sent == []


async def test_help_mentions_jokes_people_and_fits_limit(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, _, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    await _owner_say(handler, "/help")

    assert "/jokes" in sent[0]
    assert "/people" in sent[0]
    assert len(sent[0]) <= 3500


async def test_status_shows_jokes_people_callback_line(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    handler = _handler(deps)

    await _owner_say(handler, "/status")
    assert "jokes: 0, people: 0, callback: none" in sent[0]

    db.jokes_store[1] = _joke(1, "screws again", NOW)
    db.threads_store[1] = _thread(1, "Julia", "wants a boat", NOW)
    db.threads_store[2] = _thread(2, "Mike", "renovation", NOW, asked=NOW)
    db.state["callback_last_at"] = str(int(datetime(2026, 1, 8, 12, tzinfo=UTC).timestamp()))
    sent.clear()
    await _owner_say(handler, "/status")

    assert "jokes: 1, people: 1, callback: last 08.01" in sent[0]


# --- /words: slang dictionaries ----------------------------------------------

VOCAB_YAML = """
name: yard
about: yard slang
words:
  - {word: sloshy, meaning: Slushy weather}
  - {word: zoop, meaning: An invitation to sit down}
"""


@pytest.fixture
async def vocab_store(tmp_path: Path) -> VocabStore:
    database = Database(tmp_path / "bot.db")
    await database.connect()
    store = VocabStore(database)
    await store.load()
    try:
        yield store
    finally:
        await database.close()


def _doc_message(*, size: int | None = None, caption: str = "/words") -> Message:
    return _message(
        chat=_private_chat(ADMIN_ID),
        from_user=_user(ADMIN_ID),
        caption=caption,
        document=Document(
            file_id="doc-1", file_unique_id="u-1", file_name="words.yaml", file_size=size
        ),
    )


def _words_deps(
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
    store: VocabStore | None,
    data: bytes = b"",
) -> tuple[FakeDeps, FakeDb]:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    deps, db, _, _ = _deps(settings=settings, config=config)
    deps.vocab_store = store
    deps.bot = FakeDownloader(data)
    return deps, db


async def test_words_upload_document_loads_dict(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str], vocab_store: VocabStore
) -> None:
    data = "\ufeff".encode() + VOCAB_YAML.encode()
    deps, db = _words_deps(monkeypatch, config, vocab_store, data)
    handler = _handler(deps)

    await handler(_doc_message(size=200))

    assert sent == ['Dictionary "yard": 2 words, enabled.']
    assert [e.word for e in vocab_store.get().find("zoop")] == ["zoop"]
    assert db.audit_values["words:put"] == "yard"


@pytest.mark.parametrize(
    ("data", "size", "expected"),
    [
        (b"name: [", 10, "Could not parse the file: cannot read the YAML"),
        (b"name: x\nwords: []", 10, "Could not parse the file: words"),
        (b"\xff\xfe\x00", 3, "The file is not UTF-8"),
        (VOCAB_YAML.encode(), 10**7, "The file is larger than 512 KB"),
    ],
)
async def test_words_upload_errors(
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
    sent: list[str],
    vocab_store: VocabStore,
    data: bytes,
    size: int,
    expected: str,
) -> None:
    deps, db = _words_deps(monkeypatch, config, vocab_store, data)
    handler = _handler(deps)

    await handler(_doc_message(size=size))

    assert len(sent) == 1
    assert sent[0].startswith(expected)
    assert not vocab_store.get()
    assert "words:put" not in db.audit_values


async def test_words_list_show_toggle_remove(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str], vocab_store: VocabStore
) -> None:
    deps, db = _words_deps(monkeypatch, config, vocab_store)
    handler = _handler(deps)

    def private(text: str) -> Message:
        return _message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text=text)

    await handler(private("/words"))
    assert sent[-1].startswith("No dictionaries.")

    await vocab_store.put(VOCAB_YAML, now=int(NOW.timestamp()))
    await handler(private("/words"))
    assert sent[-1].startswith("yard: 2 words, on, 10.01")

    await handler(private("/words show Yard"))
    assert sent[-1].splitlines()[:3] == [
        "yard: 2 words (yard slang)",
        "sloshy: Slushy weather",
        "zoop: An invitation to sit down",
    ]

    await handler(private("/words off yard"))
    assert sent[-1] == 'Dictionary "yard" disabled.'
    assert not vocab_store.get()
    await handler(private("/words on yard"))
    assert sent[-1] == 'Dictionary "yard" enabled.'
    assert vocab_store.get()

    await handler(private("/words show nothing"))
    assert sent[-1] == 'No dictionary "nothing".'
    await handler(private("/words frob yard"))
    assert sent[-1].startswith("Usage: /words")

    await handler(private("/words rm yard"))
    assert sent[-1] == 'Dictionary "yard" deleted.'
    assert await vocab_store.dicts() == []
    assert db.audit_values["words:rm"] == "yard"
    assert db.audit_values["words:off"] == "yard"
    assert db.audit_values["words:on"] == "yard"


async def test_words_without_store_says_unavailable(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    deps, _ = _words_deps(monkeypatch, config, None)
    handler = _handler(deps)
    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/words"))
    assert sent == ["Word dictionaries are unavailable."]


async def test_words_upload_from_non_owner_is_ignored(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str], vocab_store: VocabStore
) -> None:
    deps, _ = _words_deps(monkeypatch, config, vocab_store, VOCAB_YAML.encode())
    handler = _handler(deps)
    message = _message(
        chat=_private_chat(7),
        from_user=_user(7),
        caption="/words",
        document=Document(file_id="d", file_unique_id="u", file_size=10),
    )
    await handler(message)
    assert sent == []
    assert not vocab_store.get()


async def test_status_shows_vocab_line(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str], vocab_store: VocabStore
) -> None:
    deps, db = _words_deps(monkeypatch, config, vocab_store)
    handler = _handler(deps)
    now = int(NOW.timestamp())
    await vocab_store.put(VOCAB_YAML, now=now)
    db.state[day_key("vocab_count", now, config.persona.timezone)] = "1"
    db.state["vocab_last"] = f"{now}|sloshy"

    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status"))

    assert "vocab: 1/1 dictionaries (2 words), today 1/2, last sloshy 10.01" in sent[0]


async def test_status_vocab_line_without_store(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    deps, _ = _words_deps(monkeypatch, config, None)
    handler = _handler(deps)
    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/status"))
    assert "vocab: none" in sent[0]


async def test_help_mentions_words(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    deps, _ = _words_deps(monkeypatch, config, None)
    handler = _handler(deps)
    await handler(_message(chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID), text="/help"))
    assert "/words" in sent[0]


# --- /ask: preview of an answer ----------------------------------------------


def _ask(text: str) -> Message:
    return _message(
        chat=_private_chat(ADMIN_ID), from_user=_user(ADMIN_ID, first_name="Wesley"), text=text
    )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            PreviewOutcome(sent=True, text="Happens.", reason="send:mention"),
            ["Would answer: Happens.", "(nothing was sent to the chat)"],
        ),
        (
            PreviewOutcome(
                sent=True,
                text="Sure.",
                reason="send:mention",
                reasons=("regex:venue",),
                fixes=("fix:dash",),
            ),
            [
                "Would answer: Sure.",
                "The filter flagged it but let it through (shadow): regex:venue",
                "Fixes: fix:dash",
                "(nothing was sent to the chat)",
            ],
        ),
        (
            PreviewOutcome(sent=False, text="", reason="llm:silent"),
            [
                "Would stay silent (the model chose not to answer).",
                "(nothing was sent to the chat)",
            ],
        ),
        (
            PreviewOutcome(sent=False, text="", reason="llm:timeout"),
            ["Would not answer: llm:timeout.", "(nothing was sent to the chat)"],
        ),
        (
            PreviewOutcome(
                sent=False,
                text="Zoop again.",
                reason="dedup:vocab",
                reasons=("dedup:vocab", "regex:venue"),
            ),
            [
                "Would not answer, the filter would cut it: dedup:vocab, regex:venue.",
                "Candidate: Zoop again.",
                "(nothing was sent to the chat)",
            ],
        ),
    ],
)
async def test_ask_renders_preview(
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
    sent: list[str],
    outcome: PreviewOutcome,
    expected: list[str],
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder(preview_result=outcome)
    deps, db, _, _ = _deps(settings=settings, config=config, responder=responder)
    db.display_names_value = {ADMIN_ID: "Wes"}
    handler = _handler(deps)

    await handler(_ask("/ask   Gus,  what is the weather?"))

    assert responder.preview_calls == [("Gus, what is the weather?", "Wes")]
    assert sent == ["\n".join(expected)]
    assert db.audit_stop_calls == []


async def test_ask_falls_back_to_telegram_name(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder()
    deps, _, _, _ = _deps(settings=settings, config=config, responder=responder)
    handler = _handler(deps)

    await handler(_ask("/ask hello"))

    assert responder.preview_calls == [("hello", "Wesley")]


async def test_ask_usage_and_no_responder(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder()
    deps, _, _, _ = _deps(settings=settings, config=config, responder=responder)
    await _handler(deps)(_ask("/ask"))
    assert sent == ["Usage: /ask <text>"]
    assert responder.preview_calls == []

    deps_none, _, _, _ = _deps(settings=settings, config=config, responder=None)
    await _handler(deps_none)(_ask("/ask hello"))
    assert sent[-1] == "The LLM part is off, /ask is unavailable."


async def test_ask_in_group_chat_is_ignored(
    monkeypatch: pytest.MonkeyPatch, config: Config, sent: list[str]
) -> None:
    settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
    responder = FakeResponder()
    deps, _, _, _ = _deps(settings=settings, config=config, responder=responder)
    await _handler(deps)(_message(chat=_chat(), from_user=_user(ADMIN_ID), text="/ask hello"))
    assert sent == []
    assert responder.preview_calls == []
