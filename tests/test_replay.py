"""Tests for sotto.replay.

ReplayState is a pure replay-state class, tested without gate.py/patterns.py.
run_replay() pulls in gate.py and patterns.py: a full run switches itself on as soon as
they exist on disk.
"""

import argparse
import asyncio
import importlib.util
import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from sotto import replay as replay_module
from sotto.db import Database
from sotto.fragments import load_valid_fragments
from sotto.gate_types import GateMessage, StateChange, Trigger
from sotto.replay import ReplayState, run_replay
from tests.packs import DEMO_PACK

DEMO_FRAG = load_valid_fragments(DEMO_PACK / "fragments.yaml")

TZ = "Europe/London"


@pytest.fixture(autouse=True)
def _demo_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests run against the demo pack (conftest defaults PERSONA_DIR to the legacy one)."""
    monkeypatch.setenv("PERSONA_DIR", str(DEMO_PACK))


_GATE_AND_PATTERNS_AVAILABLE = (
    importlib.util.find_spec("sotto.gate") is not None
    and importlib.util.find_spec("sotto.patterns") is not None
)


def _gate_msg(user_id: int, created_at: int, tg_message_id: int = 1) -> GateMessage:
    return GateMessage(
        chat_id=0,
        tg_message_id=tg_message_id,
        user_id=user_id,
        is_bot=False,
        text="hello",
        reply_to_bot=False,
        created_at=created_at,
    )


# --- ReplayState.recent (live_talk window) ------------------------------------


def test_recent_window_prunes_old_messages() -> None:
    state = ReplayState(TZ, window_min=10)
    base = 1_000_000

    state.push_recent(user_id=1, created_at=base)
    state.push_recent(user_id=2, created_at=base + 60)
    # +11 minutes after the first message -> it is already outside the 10-minute window.
    state.push_recent(user_id=3, created_at=base + 11 * 60)

    window = state.recent_window(base + 11 * 60)
    assert [activity.user_id for activity in window] == [2, 3]


def test_recent_window_includes_current_message() -> None:
    state = ReplayState(TZ, window_min=10)
    state.push_recent(user_id=1, created_at=1_000_000)
    window = state.recent_window(1_000_000)
    assert len(window) == 1
    assert window[0].user_id == 1


def test_gate_state_reflects_recent_and_counters() -> None:
    state = ReplayState(TZ, window_min=10)
    now = 1_757_000_000  # arbitrary but fixed moment
    state.push_recent(user_id=1, created_at=now)

    msg = _gate_msg(user_id=1, created_at=now)
    gate_state = state.gate_state(msg, now)

    assert len(gate_state.recent) == 1
    assert gate_state.mention_count_today == 0
    assert gate_state.ambient_count_today == 0
    assert gate_state.last_mention_reply_at is None
    assert gate_state.last_mention_reply_at_user is None


# --- ReplayState.apply_pass (counters and last_*_at) --------------------------


@pytest.mark.parametrize("trigger", [Trigger.MENTION, Trigger.REPLY, Trigger.NAME])
def test_apply_pass_increments_mention_counters(trigger: Trigger) -> None:
    state = ReplayState(TZ, window_min=10)
    now = 1_757_000_000
    msg = _gate_msg(user_id=7, created_at=now)

    state.apply_pass(trigger, msg, now)

    gate_state = state.gate_state(msg, now)
    assert gate_state.mention_count_today == 1
    assert gate_state.ambient_count_today == 0
    assert gate_state.last_mention_reply_at == now
    assert gate_state.last_mention_reply_at_user == now


def test_apply_pass_ambient_increments_ambient_counter_only() -> None:
    state = ReplayState(TZ, window_min=10)
    now = 1_757_000_000
    msg = _gate_msg(user_id=7, created_at=now)

    state.apply_pass(Trigger.AMBIENT, msg, now)

    gate_state = state.gate_state(msg, now)
    assert gate_state.ambient_count_today == 1
    assert gate_state.mention_count_today == 0
    assert gate_state.last_mention_reply_at is None
    assert gate_state.last_mention_reply_at_user is None


def test_apply_pass_counters_are_per_day() -> None:
    state = ReplayState(TZ, window_min=10)
    # 2026-09-10 23:50 and 2026-09-11 00:10 Europe/London -> different days.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(TZ)
    day1 = int(datetime.fromisoformat("2026-09-10T23:50:00").replace(tzinfo=zone).timestamp())
    day2 = int(datetime.fromisoformat("2026-09-11T00:10:00").replace(tzinfo=zone).timestamp())

    msg1 = _gate_msg(user_id=1, created_at=day1)
    msg2 = _gate_msg(user_id=1, created_at=day2)

    state.apply_pass(Trigger.AMBIENT, msg1, day1)
    state.apply_pass(Trigger.AMBIENT, msg2, day2)

    assert state.gate_state(msg1, day1).ambient_count_today == 1
    assert state.gate_state(msg2, day2).ambient_count_today == 1


# --- ReplayState.apply_state_changes -----------------------------------------


def test_apply_state_changes_sets_topic_cooldown() -> None:
    state = ReplayState(TZ, window_min=10)
    state.apply_state_changes((StateChange("topic_cooldown_until", "123456"),))
    assert state.topic_cooldown_until == 123456


def test_apply_state_changes_clears_topic_cooldown() -> None:
    state = ReplayState(TZ, window_min=10)
    state.apply_state_changes((StateChange("topic_cooldown_until", "123456"),))
    state.apply_state_changes((StateChange("topic_cooldown_until", None),))
    assert state.topic_cooldown_until is None


# --- run_replay: a full run, needs gate.py and patterns.py ---------------


def _write_synthetic_export(path: Path, tz: str) -> None:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    start = datetime(2026, 9, 10, 18, 0, 0, tzinfo=ZoneInfo(tz))
    authors = [("Gene", "user111"), ("Julia", "user222")]
    messages = []
    for i in range(6):
        name, from_id = authors[i % 2]
        when = start + timedelta(minutes=i)
        messages.append(
            {
                "id": i + 1,
                "type": "message",
                "date": when.strftime("%Y-%m-%dT%H:%M:%S"),
                "from": name,
                "from_id": from_id,
                "text": f"message number {i + 1}",
            }
        )
    payload = {"name": "test", "type": "private_group", "id": 1, "messages": messages}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.mark.skipif(
    not _GATE_AND_PATTERNS_AVAILABLE,
    reason="sotto.gate and/or sotto.patterns are not written yet",
)
def test_run_replay_produces_day_line_and_summary(tmp_path: Path) -> None:
    export_path = tmp_path / "result.json"
    _write_synthetic_export(export_path, TZ)

    config_path = DEMO_PACK / "config.yaml"

    args = argparse.Namespace(
        source=export_path,
        config=config_path,
        seed=1,
        bot_username="example_persona_bot",
        bot_user_id=0,
        verbose=False,
        generate=False,
        judge=False,
        max_calls=20,
    )

    report = run_replay(args)

    assert "Итого:" in report  # PR-8: hardcoded Russian label in src/sotto/replay.py
    assert "2026-09-10" in report


def _gate_args(export_path: Path, **overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "source": export_path,
        "config": None,
        "persona": None,
        "seed": 1,
        "bot_username": "example_persona_bot",
        "bot_user_id": 0,
        "verbose": False,
        "generate": False,
        "judge": False,
        "max_calls": 20,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.skipif(
    not _GATE_AND_PATTERNS_AVAILABLE,
    reason="sotto.gate and/or sotto.patterns are not written yet",
)
def test_run_replay_persona_argument_matches_explicit_config(tmp_path: Path) -> None:
    export_path = tmp_path / "result.json"
    _write_synthetic_export(export_path, TZ)

    via_persona = run_replay(_gate_args(export_path, persona=DEMO_PACK))
    via_config = run_replay(_gate_args(export_path, config=DEMO_PACK / "config.yaml"))
    via_env = run_replay(_gate_args(export_path))  # PERSONA_DIR from the autouse fixture

    assert via_persona == via_config == via_env


def test_run_replay_fails_closed_without_persona_or_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sotto.persona import PackError

    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("PERSONA_DIR")
    export_path = tmp_path / "result.json"
    _write_synthetic_export(export_path, TZ)

    with pytest.raises(PackError, match="PERSONA_DIR not set"):
        run_replay(_gate_args(export_path))


def test_replay_parser_has_persona_and_no_default_config() -> None:
    from sotto.replay import _parse_args

    args = _parse_args(["export.json", "--persona", "/srv/pack"])

    assert args.persona == Path("/srv/pack")
    assert args.config is None


# --- run_replay --generate: real generation + filter on top of PASS -----------


def _write_name_trigger_export(path: Path, tz: str) -> None:
    """One PASS by the name trigger ("gus"): guaranteed, without live-talk or the dice."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    when = datetime(2026, 9, 10, 18, 0, 0, tzinfo=ZoneInfo(tz))
    payload = {
        "name": "test",
        "type": "private_group",
        "id": 1,
        "messages": [
            {
                "id": 1,
                "type": "message",
                "date": when.strftime("%Y-%m-%dT%H:%M:%S"),
                "from": "Gene",
                "from_id": "user111",
                "text": "gus hello how are you",
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_minimal_generate_config(path: Path) -> None:
    path.write_text(
        f'llm:\n  main_model: "test/model"\npersona:\n  name_triggers: [gus]\n  timezone: {TZ}\n',
        encoding="utf-8",
    )


@pytest.mark.skipif(
    not _GATE_AND_PATTERNS_AVAILABLE,
    reason="sotto.gate and/or sotto.patterns are not written yet",
)
def test_run_replay_generate_prints_reply_and_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    export_path = tmp_path / "result.json"
    _write_name_trigger_export(export_path, TZ)
    config_path = tmp_path / "config.yaml"
    _write_minimal_generate_config(config_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        body = {
            "choices": [{"message": {"content": '{"speak": true, "text": "Happens, pal."}'}}],
            "usage": {"cost": 0.0007, "prompt_tokens": 42, "completion_tokens": 7},
        }
        return httpx.Response(200, json=body)

    mock_transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=mock_transport)

    monkeypatch.setattr(replay_module.httpx, "AsyncClient", fake_async_client)

    args = argparse.Namespace(
        source=export_path,
        config=config_path,
        seed=1,
        bot_username="example_persona_bot",
        bot_user_id=0,
        verbose=False,
        generate=True,
        judge=False,
        max_calls=20,
    )

    report = run_replay(args)

    assert "Генерация:" in report  # PR-8: hardcoded Russian label in src/sotto/replay.py
    assert "gus hello how are you" in report
    assert "Happens, pal." in report
    assert "| pass" in report
    assert (
        "вызовов=1, отправлено бы=1" in report
    )  # PR-8: hardcoded Russian label in src/sotto/replay.py
    assert "потрачено $=0.0007" in report  # PR-8: hardcoded Russian label in src/sotto/replay.py


def test_run_replay_generate_without_api_key_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cwd without .env: guarantees the repository's real .env (with its own key) is not
    # mixed in, so Settings() honestly sees openrouter_api_key=None.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOT_TOKEN", "123:test-token")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    export_path = tmp_path / "result.json"
    _write_name_trigger_export(export_path, TZ)
    config_path = tmp_path / "config.yaml"
    _write_minimal_generate_config(config_path)

    args = argparse.Namespace(
        source=export_path,
        config=config_path,
        seed=1,
        bot_username="example_persona_bot",
        bot_user_id=0,
        verbose=False,
        generate=True,
        judge=False,
        max_calls=20,
    )

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        run_replay(args)


# --- run_replay --generate --max-calls: a cap on real network calls --------


def _write_repeated_name_trigger_export(path: Path, tz: str, count: int) -> None:
    """``count`` guaranteed PASSes through the "gus" name trigger, spread over time
    (the test itself zeroes mention_cooldown_sec/mention_chat_cooldown_sec so that the
    address cooldown does not block them)."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    start = datetime(2026, 9, 10, 18, 0, 0, tzinfo=ZoneInfo(tz))
    authors = [("Gene", "user111"), ("Julia", "user222")]
    messages = []
    for i in range(count):
        name, from_id = authors[i % 2]
        when = start + timedelta(minutes=2 * i)
        messages.append(
            {
                "id": i + 1,
                "type": "message",
                "date": when.strftime("%Y-%m-%dT%H:%M:%S"),
                "from": name,
                "from_id": from_id,
                "text": f"gus hello how are you number {i + 1}",
            }
        )
    payload = {"name": "test", "type": "private_group", "id": 1, "messages": messages}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_max_calls_config(path: Path) -> None:
    path.write_text(
        "llm:\n"
        '  main_model: "test/model"\n'
        '  judge_model: "test/judge-model"\n'
        "behaviour:\n"
        "  mention_cooldown_sec: 0\n"
        "  mention_chat_cooldown_sec: 0\n",
        encoding="utf-8",
    )


def test_run_replay_generate_max_calls_counts_real_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--max-calls counts real network calls (main + judge), not the number of PASSes
    with generation.

    The export gives 6 guaranteed PASSes. filters.shadow=true by default (the config is
    minimal, filters are not overridden) -> the judge is called on every generation that
    passed the check before the start. PASS 1: calls_so_far=0 < 3 -> generation, the
    counter in the store becomes 2 (main + judge). PASS 2: calls_so_far=2 is still < 3 ->
    one more generation, the counter becomes 4. From PASS 3 to PASS 6: calls_so_far=4 >= 3
    -> no generation starts. Result: 4 real network calls, one judge call above the
    declared cap of 3, an accepted overshoot (see replay.py).
    """
    monkeypatch.setenv("BOT_TOKEN", "123:test-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")

    export_path = tmp_path / "result.json"
    _write_repeated_name_trigger_export(export_path, TZ, count=6)
    config_path = tmp_path / "config.yaml"
    _write_max_calls_config(config_path)

    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = {
            "choices": [{"message": {"content": '{"speak": true, "text": "Happens, pal."}'}}],
            "usage": {"cost": 0.0001, "prompt_tokens": 10, "completion_tokens": 5},
        }
        return httpx.Response(200, json=body)

    mock_transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=mock_transport)

    monkeypatch.setattr(replay_module.httpx, "AsyncClient", fake_async_client)

    args = argparse.Namespace(
        source=export_path,
        config=config_path,
        seed=1,
        bot_username="example_persona_bot",
        bot_user_id=0,
        verbose=False,
        generate=True,
        judge=True,
        max_calls=3,
    )

    report = run_replay(args)

    assert call_count <= 4
    assert "вызовов=" in report  # PR-8: hardcoded Russian label in src/sotto/replay.py
    reported_calls = int(
        report.split("вызовов=")[1].split(",")[0]
    )  # PR-8: hardcoded Russian label in src/sotto/replay.py
    assert reported_calls == call_count
    assert reported_calls <= 4


# --- run_replay: the source is the bot's own database (bot.db), not only an export -----------


async def _seed_bot_db(db_path: Path, tz: str) -> None:
    """8 messages from two people + 1 from the bot, on two different local days.

    The owner cannot always export history from Telegram Desktop (saving content is
    forbidden in the group), so the replay reads the bot's own database instead.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(tz)
    day1 = datetime(2026, 9, 8, 18, 0, 0, tzinfo=zone)
    day2 = datetime(2026, 9, 10, 18, 0, 0, tzinfo=zone)
    authors = [(111, "Gene"), (222, "Julia")]

    db = Database(db_path)
    await db.connect()
    try:
        for i in range(4):
            user_id, name = authors[i % 2]
            await db.insert_message(
                tg_message_id=i + 1,
                chat_id=555,
                user_id=user_id,
                display_name=name,
                text=f"day one, message {i + 1}",
                reply_to_tg_message_id=None,
                is_bot=False,
                created_at=int((day1 + timedelta(minutes=i)).timestamp()),
            )
        for i in range(4):
            user_id, name = authors[i % 2]
            await db.insert_message(
                tg_message_id=i + 5,
                chat_id=555,
                user_id=user_id,
                display_name=name,
                text=f"day two, message {i + 1}",
                reply_to_tg_message_id=None,
                is_bot=False,
                created_at=int((day2 + timedelta(minutes=i)).timestamp()),
            )
        await db.insert_message(
            tg_message_id=9,
            chat_id=555,
            user_id=999,
            display_name="Gus Marlow",
            text="Same to you",
            reply_to_tg_message_id=None,
            is_bot=True,
            created_at=int((day2 + timedelta(minutes=5)).timestamp()),
        )
    finally:
        await db.close()


def _seed_bot_db_sync(db_path: Path, tz: str) -> None:
    asyncio.run(_seed_bot_db(db_path, tz))


def _db_args(source: Path, **overrides: object) -> argparse.Namespace:
    config_path = DEMO_PACK / "config.yaml"
    base: dict[str, object] = {
        "source": source,
        "config": config_path,
        "seed": 1,
        "bot_username": "example_persona_bot",
        "bot_user_id": 0,
        "from_db": False,
        "chat_id": None,
        "since": None,
        "until": None,
        "verbose": False,
        "generate": False,
        "judge": False,
        "max_calls": 20,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.skipif(
    not _GATE_AND_PATTERNS_AVAILABLE,
    reason="sotto.gate and/or sotto.patterns are not written yet",
)
def test_run_replay_from_bot_db_produces_day_line_and_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "bot.db"
    _seed_bot_db_sync(db_path, TZ)

    report = run_replay(_db_args(db_path))

    assert "Итого:" in report  # PR-8: hardcoded Russian label in src/sotto/replay.py
    assert "2026-09-08" in report
    assert "2026-09-10" in report


@pytest.mark.skipif(
    not _GATE_AND_PATTERNS_AVAILABLE,
    reason="sotto.gate and/or sotto.patterns are not written yet",
)
def test_run_replay_from_bot_db_since_cuts_off_early_messages(tmp_path: Path) -> None:
    db_path = tmp_path / "bot.db"
    _seed_bot_db_sync(db_path, TZ)

    report = run_replay(_db_args(db_path, since=date(2026, 9, 9)))

    assert "2026-09-10" in report
    assert "2026-09-08" not in report


@pytest.mark.skipif(
    not _GATE_AND_PATTERNS_AVAILABLE,
    reason="sotto.gate and/or sotto.patterns are not written yet",
)
def test_run_replay_from_bot_db_does_not_modify_live_file(tmp_path: Path) -> None:
    """The replay reads the bot's database read-only (sqlite3 URI mode=ro): the live file
    (and its WAL) must not change, neither size nor mtime."""
    db_path = tmp_path / "bot.db"
    _seed_bot_db_sync(db_path, TZ)

    stat_before = db_path.stat()

    run_replay(_db_args(db_path))

    stat_after = db_path.stat()
    assert stat_before.st_mtime_ns == stat_after.st_mtime_ns
    assert stat_before.st_size == stat_after.st_size


def test_situation_for_ambient_matches_responder() -> None:
    from sotto.gate_types import Trigger

    ambient = DEMO_FRAG.text("situations.ambient")
    assert replay_module.situation_for(Trigger.AMBIENT, frag=DEMO_FRAG) == ambient
    assert replay_module.situation_for(Trigger.NAME, frag=DEMO_FRAG) == ""
