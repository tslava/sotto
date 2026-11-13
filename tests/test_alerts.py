from __future__ import annotations

import asyncio

import pytest
from aiogram.exceptions import TelegramForbiddenError, TelegramNetworkError
from aiogram.methods import SendMessage

from sotto.alerts import OwnerAlerter, is_refusal, render_alert
from sotto.config_models import Config
from sotto.persona import Persona

ADMIN_ID = 42
NOW = 1_793_000_000


class FakeBot:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.sent: list[tuple[int, str]] = []
        self.error = error

    async def send_message(self, chat_id: int, text: str) -> object:
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        self.sent.append((chat_id, text))
        return object()


class HangingBot:
    """A Telegram that never answers: send_message hangs until it is cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def send_message(self, chat_id: int, text: str) -> object:
        self.started.set()
        await asyncio.Event().wait()
        return object()


class FakeStore:
    def __init__(self) -> None:
        self.state: dict[str, str] = {}

    async def get_state(self, key: str) -> str | None:
        return self.state.get(key)

    async def set_state(self, key: str, value: str) -> None:
        self.state[key] = value


class Clock:
    def __init__(self, now: int = NOW) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _alerter(
    bot: FakeBot | HangingBot,
    store: FakeStore,
    *,
    cfg: Config | None = None,
    admin_user_id: int = ADMIN_ID,
    clock: Clock | None = None,
) -> OwnerAlerter:
    config = cfg or Config()
    return OwnerAlerter(
        bot=bot,
        db=store,
        admin_user_id=admin_user_id,
        cfg_getter=lambda: config,
        clock=clock or Clock(),
    )


async def _refuse(alerter: OwnerAlerter, status: int = 402, message: str = "") -> None:
    await alerter.provider_refused(status=status, model="m", message=message)
    # Sending is a background task: let it run.
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, True),
        (401, True),
        (402, True),
        (403, True),
        (404, True),
        (408, False),
        (429, False),
        (500, False),
        (502, False),
        (200, False),
    ],
)
def test_is_refusal(status: int, expected: bool) -> None:
    assert is_refusal(status) is expected


def test_render_alert_has_hint_model_and_message(demo_pack: Persona) -> None:
    text = render_alert(
        status=402,
        model="anthropic/claude-opus-5",
        message="x" * 500,
        interval_hours=24,
        catalog=demo_pack.catalog,
    )
    assert "402" in text
    assert "openrouter.ai/settings/credits" in text
    assert "anthropic/claude-opus-5" in text
    assert "x" * 200 in text and "x" * 201 not in text
    assert "24 h" in text


def test_render_alert_without_message_and_unknown_status(demo_pack: Persona) -> None:
    text = render_alert(
        status=404, model="m", message="", interval_hours=6, catalog=demo_pack.catalog
    )
    assert "404, request rejected" in text
    assert "Response:" not in text


def test_render_alert_defaults_to_the_english_catalog() -> None:
    text = render_alert(status=401, model="m", message="bad key", interval_hours=12)
    assert text.splitlines() == [
        "OpenRouter refused the request: 401, the OpenRouter key was not accepted, "
        "check OPENROUTER_API_KEY.",
        "Model: m.",
        "Response: bad key",
        "Until this is fixed, the bot stays silent wherever this model is needed. "
        "The next message like this will come in 12 h at the earliest.",
    ]


# Stays Russian: pins that a pack locale overrides the engine catalog (legacy ru locale).
async def test_alert_text_comes_from_the_persona_locale(legacy_pack: Persona) -> None:
    bot, store = FakeBot(), FakeStore()
    alerter = OwnerAlerter(
        bot=bot,
        db=store,
        admin_user_id=ADMIN_ID,
        cfg_getter=Config,
        clock=lambda: NOW,
        persona=legacy_pack,
    )
    await alerter.provider_refused(status=402, model="m", message="no credits")
    await asyncio.sleep(0.05)
    assert "кончились кредиты" in bot.sent[0][1]


async def test_refusal_sends_once_per_interval_per_status() -> None:
    bot, store, clock = FakeBot(), FakeStore(), Clock()
    alerter = _alerter(bot, store, clock=clock)

    await _refuse(alerter, message="no credits")
    clock.now = NOW + 3600
    await _refuse(alerter, message="no credits")
    assert len(bot.sent) == 1
    assert bot.sent[0][0] == ADMIN_ID
    assert "no credits" in bot.sent[0][1]
    assert store.state["alert_next:llm_402"] == str(NOW + 24 * 3600)

    # Another code is its own alert; the same one after the interval fires again.
    await _refuse(alerter, status=401)
    clock.now = NOW + 24 * 3600
    await _refuse(alerter)
    assert len(bot.sent) == 3


async def test_interval_uses_own_clock_not_caller_time() -> None:
    # The hook does not get the caller's now (for a photo description that is the date of an
    # old message): the pause between alerts is counted only by the alerter's own clock.
    bot, store, clock = FakeBot(), FakeStore(), Clock()
    alerter = _alerter(bot, store, clock=clock)
    await _refuse(alerter)
    clock.now = NOW + 60
    await _refuse(alerter)
    assert len(bot.sent) == 1


async def test_transient_status_is_not_alerted() -> None:
    bot, store = FakeBot(), FakeStore()
    alerter = _alerter(bot, store)
    for status in (408, 429, 500, 503):
        await _refuse(alerter, status=status)
    assert bot.sent == []
    assert store.state == {}


async def test_disabled_or_no_admin_sends_nothing() -> None:
    bot, store = FakeBot(), FakeStore()
    cfg = Config()
    cfg.llm.alert_owner = False
    await _refuse(_alerter(bot, store, cfg=cfg))
    await _refuse(_alerter(bot, store, admin_user_id=0))
    assert bot.sent == []


async def test_parallel_refusals_send_one_message() -> None:
    bot, store = FakeBot(), FakeStore()
    alerter = _alerter(bot, store)
    await asyncio.gather(
        *(alerter.provider_refused(status=402, model="m", message="") for _ in range(5))
    )
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(bot.sent) == 1


async def test_hanging_telegram_does_not_block_caller() -> None:
    bot, store = HangingBot(), FakeStore()
    alerter = _alerter(bot, store)
    # The model call waits only for the slot to be taken, not for the send.
    await asyncio.wait_for(alerter.provider_refused(status=402, model="m", message=""), 1)
    await asyncio.wait_for(bot.started.wait(), 1)
    # The slot is taken before sending: a second refusal neither waits nor sends a repeat.
    await asyncio.wait_for(alerter.provider_refused(status=402, model="m", message=""), 1)
    assert store.state["alert_next:llm_402"] == str(NOW + 24 * 3600)

    # A shutdown in the middle of sending: a retry in an hour, not in a day.
    await alerter.shutdown()
    assert store.state["alert_next:llm_402"] == str(NOW + 3600)


async def test_send_failure_retries_in_an_hour(caplog: pytest.LogCaptureFixture) -> None:
    error = TelegramNetworkError(method=SendMessage(chat_id=ADMIN_ID, text="x"), message="timeout")
    bot, store, clock = FakeBot(error=error), FakeStore(), Clock()
    alerter = _alerter(bot, store, clock=clock)

    await _refuse(alerter)
    assert store.state["alert_next:llm_402"] == str(NOW + 3600)
    assert "owner alert not sent" in caplog.text

    bot.error = None
    clock.now = NOW + 1800
    await _refuse(alerter)
    assert bot.sent == []
    clock.now = NOW + 3600
    await _refuse(alerter)
    assert len(bot.sent) == 1


async def test_forbidden_also_retries_later() -> None:
    error = TelegramForbiddenError(
        method=SendMessage(chat_id=ADMIN_ID, text="x"), message="Forbidden"
    )
    bot, store = FakeBot(error=error), FakeStore()
    await _refuse(_alerter(bot, store))
    assert store.state["alert_next:llm_402"] == str(NOW + 3600)
