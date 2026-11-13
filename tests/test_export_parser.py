"""Tests for sotto.export_parser: a small result.json is assembled by hand."""

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sotto.export_parser import ExportMessage, parse_export
from sotto.fragments import load_valid_fragments
from tests.packs import DEMO_PACK

DEMO_FRAG = load_valid_fragments(DEMO_PACK / "fragments.yaml")

TZ = "Europe/London"


def _dt(iso: str) -> int:
    """ISO without tz -> unix seconds in TZ, for comparing with created_at in tests."""
    return int(datetime.fromisoformat(iso).replace(tzinfo=ZoneInfo(TZ)).timestamp())


def _write_export(tmp_path: Path, messages: list[dict]) -> Path:
    payload = {
        "name": "Brindlemoor friends",
        "type": "private_group",
        "id": 1234567890,
        "messages": messages,
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# Raw export records, deliberately shuffled (id/date do not grow in list order):
# checks that parse_export really sorts and does not just keep the JSON order.
_RAW_MESSAGES = [
    {
        "id": 8,
        "type": "message",
        "date": "2026-09-10T12:07:00",
        # from_id is deliberately missing -> user_id must become 0
        "from": "Guest",
        "text": "No sender id",
    },
    {
        "id": 2,
        "type": "message",
        "date": "2026-09-10T12:01:00",
        "from": "Julia",
        "from_id": "user222",
        "text": [
            "Hello, ",
            {"type": "mention", "text": "@example_persona_bot"},
            "!",
        ],
    },
    {
        "id": 1,
        "type": "message",
        "date": "2026-09-10T12:00:00",
        "from": "Gene",
        "from_id": "user111",
        "text": "Hello everyone",
    },
    {
        "id": 4,
        "type": "message",
        "date": "2026-09-10T12:03:00",
        "from": "Gene",
        "from_id": "user111",
        "photo": "photos/photo_4.jpg",
        "text": "",
    },
    {
        "id": 9,
        "type": "message",
        # neither "date" nor "date_unixtime" -> broken record, skipped with a WARNING
        "from": "Gene",
        "from_id": "user111",
        "text": "Broken message without a date",
    },
    {
        "id": 3,
        "type": "service",
        "date": "2026-09-10T12:02:00",
        "action": "pin_message",
        "text": "",
    },
    {
        "id": 6,
        "type": "message",
        "date": "2026-09-10T12:05:00",
        "from": "Gene",
        "from_id": "user111",
        "media_type": "voice_message",
        "text": "",
    },
    {
        "id": 5,
        "type": "message",
        "date": "2026-09-10T12:04:00",
        "from": "Julia",
        "from_id": "user222",
        "media_type": "sticker",
        "text": "",
    },
    {
        "id": 7,
        "type": "message",
        "date": "2026-09-10T12:06:00",
        "from": "Julia",
        "from_id": "user222",
        "reply_to_message_id": 1,
        "text": "Reply to the first",
    },
    {
        "id": 10,
        "type": "message",
        # date deliberately differs from date_unixtime: unixtime wins
        "date": "2026-09-10T23:59:00",
        "date_unixtime": "1700000000",
        "from": "Gene",
        "from_id": "user111",
        "text": "With unixtime",
    },
]


def test_parse_export_full(tmp_path: Path) -> None:
    path = _write_export(tmp_path, _RAW_MESSAGES)
    result = parse_export(path, TZ, frag=DEMO_FRAG)

    # service (id=3) and the broken one without a date (id=9) are dropped -> 8 of 10 records.
    # id=10 uses date_unixtime=1700000000 (year 2023), so it comes before the rest
    # by created_at, even though the record's "date" points at 2026.
    assert [m.tg_message_id for m in result] == [10, 1, 2, 4, 5, 6, 7, 8]

    by_id: dict[int, ExportMessage] = {m.tg_message_id: m for m in result}

    # A plain message.
    msg1 = by_id[1]
    assert msg1.text == "Hello everyone"
    assert msg1.display_name == "Gene"
    assert msg1.user_id == 111
    assert msg1.reply_to_tg_message_id is None
    assert msg1.created_at == _dt("2026-09-10T12:00:00")

    # list-text with a mention entity glued in.
    msg2 = by_id[2]
    assert msg2.text == "Hello, @example_persona_bot!"
    assert msg2.user_id == 222

    # Photo without a caption -> placeholder.
    msg4 = by_id[4]
    assert msg4.text == "[photo]"

    # Sticker -> placeholder.
    msg5 = by_id[5]
    assert msg5.text == "[sticker]"

    # Voice message -> placeholder.
    msg6 = by_id[6]
    assert msg6.text == "[voice message]"

    # A reply keeps reply_to_message_id.
    msg7 = by_id[7]
    assert msg7.text == "Reply to the first"
    assert msg7.reply_to_tg_message_id == 1

    # No from_id -> user_id 0.
    msg8 = by_id[8]
    assert msg8.user_id == 0
    assert msg8.display_name == "Guest"

    # date_unixtime takes priority over date.
    msg10 = by_id[10]
    assert msg10.created_at == 1700000000
    assert msg10.created_at != _dt("2026-09-10T23:59:00")

    # Sorted by (created_at, id) ascending despite the shuffled JSON.
    assert [m.created_at for m in result] == sorted(m.created_at for m in result)


def test_service_messages_are_skipped(tmp_path: Path) -> None:
    path = _write_export(tmp_path, _RAW_MESSAGES)
    result = parse_export(path, TZ, frag=DEMO_FRAG)
    assert 3 not in [m.tg_message_id for m in result]


def test_broken_record_without_date_is_skipped_with_warning(tmp_path: Path, caplog) -> None:
    path = _write_export(tmp_path, _RAW_MESSAGES)
    with caplog.at_level(logging.WARNING, logger="sotto.export_parser"):
        result = parse_export(path, TZ, frag=DEMO_FRAG)

    assert 9 not in [m.tg_message_id for m in result]
    assert any("date" in record.message for record in caplog.records)


def test_broken_record_without_id_is_skipped_with_warning(tmp_path: Path, caplog) -> None:
    messages = [
        {
            "type": "message",
            "date": "2026-09-10T12:00:00",
            "from": "Gene",
            "from_id": "user111",
            "text": "No id",
        },
        {
            "id": 1,
            "type": "message",
            "date": "2026-09-10T12:00:00",
            "from": "Gene",
            "from_id": "user111",
            "text": "Normal",
        },
    ]
    path = _write_export(tmp_path, messages)
    with caplog.at_level(logging.WARNING, logger="sotto.export_parser"):
        result = parse_export(path, TZ, frag=DEMO_FRAG)

    assert [m.tg_message_id for m in result] == [1]
    assert any("id" in record.message for record in caplog.records)


def test_media_and_empty_text_without_placeholder_is_dropped(tmp_path: Path) -> None:
    """No text and no recognised media -> the message is silently skipped."""
    messages = [
        {
            "id": 1,
            "type": "message",
            "date": "2026-09-10T12:00:00",
            "from": "Gene",
            "from_id": "user111",
            "text": "",
        }
    ]
    path = _write_export(tmp_path, messages)
    result = parse_export(path, TZ, frag=DEMO_FRAG)
    assert result == []
