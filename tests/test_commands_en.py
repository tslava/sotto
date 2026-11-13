"""Owner commands with the English catalog (the production owner UI after the cutover).

The Russian texts are pinned by tests/test_commands.py against the legacy pack; here the same
handlers run with ``locales/en``: exact lines for the commands that carry logic, plus a sweep that
no answer to a data-free command contains Cyrillic.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from aiogram.types import Message

from sotto.config_models import Config
from sotto.i18n import engine_catalog
from sotto.responder import PreviewOutcome, SendOutcome
from tests.test_commands import (
    ADMIN_ID,
    NOW,
    OWN_CHAT_ID,
    FakeResponder,
    _deps,
    _handler,
    _make_settings,
    _message,
    _private_chat,
    _user,
)

_CYRILLIC = re.compile("[А-Яа-яЁё]")


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Intercepts Message.answer, like the fixture of tests/test_commands.py."""
    captured: list[str] = []

    async def fake_answer(self: Message, text: str, **kwargs: object) -> None:
        captured.append(text)

    monkeypatch.setattr(Message, "answer", fake_answer)
    return captured


def _en(deps):  # type: ignore[no-untyped-def]
    deps.persona = replace(deps.persona, catalog=engine_catalog("en"))
    return deps


async def _run(handler, text: str, message_id: int = 10) -> None:  # type: ignore[no-untyped-def]
    await handler(
        _message(
            chat=_private_chat(ADMIN_ID),
            from_user=_user(ADMIN_ID),
            text=text,
            message_id=message_id,
        )
    )


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch, config: Config):  # type: ignore[no-untyped-def]
    def build(**kwargs):  # type: ignore[no-untyped-def]
        settings = _make_settings(monkeypatch, allowed_chat_id=OWN_CHAT_ID, admin_user_id=ADMIN_ID)
        deps, db, store, prompts = _deps(settings=settings, config=config, **kwargs)
        return _en(deps), db, store, prompts, _handler(deps)

    return build


@pytest.mark.parametrize(
    "command",
    [
        "/help",
        "/frobnicate",
        "/last",
        "/why",
        "/get no.such.prefix",
        "/set",
        "/unset",
        "/unset behaviour.daily_cap",
        "/rollback",
        "/rollback abc",
        "/rollback 99",
        "/ex last",
        "/life",
        "/life list",
        "/life rm 5",
        "/life post 5",
        "/life rm",
        "/say",
        "/say hello",
        "/ask",
        "/ask hello",
        "/memory",
        "/memory rm 1",
        "/memory run",
        "/memory zzz",
        "/diary",
        "/diary rm 3",
        "/diary zzz",
        "/jokes",
        "/jokes rm 9",
        "/people",
        "/people rm 9",
        "/words",
        "/words show nothing",
        "/panic",
        "/resume",
    ],
)
async def test_data_free_commands_answer_in_english(setup, sent: list[str], command: str) -> None:  # type: ignore[no-untyped-def]
    _, _, _, _, handler = setup()
    await _run(handler, command)
    assert sent, command
    assert not _CYRILLIC.search(sent[0]), (command, sent[0])


async def test_status_in_english_has_no_russian_labels(setup, sent: list[str]) -> None:  # type: ignore[no-untyped-def]
    _, db, _, _, handler = setup()
    db.state["panic"] = "1"
    await _run(handler, "/status")
    lines = sent[0].splitlines()
    assert lines[1] == "Panic: yes"
    assert lines[2] == "Stopped until: none"
    assert lines[3].startswith("LLM circuit breaker: closed, consecutive errors: 0, last provider")
    assert lines[3].endswith("refusal: none in 24 h")
    assert any(line.startswith("Today: ambient=0, mention=0") for line in lines)
    assert "hot window: none" in lines
    assert "weather: none" in lines


async def test_resume_lists_what_was_cleared_in_english(setup, sent: list[str]) -> None:  # type: ignore[no-untyped-def]
    _, db, _, _, handler = setup()
    db.state["panic"] = "1"
    db.state["llm_circuit_until"] = str(int(NOW.timestamp()) + 1800)
    db.state["llm_error_streak"] = "5"
    await _run(handler, "/resume")
    assert sent == ["Resuming. Cleared: panic, LLM circuit breaker (5 consecutive errors)."]


async def test_life_outcomes_in_english(setup, sent: list[str]) -> None:  # type: ignore[no-untyped-def]
    responder = FakeResponder(
        announce_life_result=SendOutcome(sent=True, text="Sold the cart", reason="send:life")
    )
    _, _, _, _, handler = setup(responder=responder)
    await _run(handler, "/life Sold the cart")
    responder.announce_life_result = SendOutcome(
        sent=False, text="Candidate text", reason="regex:length"
    )
    await _run(handler, "/life post 1", message_id=11)
    responder.announce_life_result = SendOutcome(sent=False, text="", reason="blocked:panic")
    await _run(handler, "/life post 1", message_id=12)
    await _run(handler, "/life list", message_id=13)
    assert sent[0] == "Saved #1. Sent: Sold the cart"
    assert sent[1] == "Not sent: regex:length\nCandidate: Candidate text"
    assert sent[2] == "The bot is silent (panic/stop) - /resume."
    assert re.fullmatch(r"#1 \d\d\.\d\d\.\d{4} - Sold the cart", sent[3])


async def test_ask_preview_variants_in_english(setup, sent: list[str]) -> None:  # type: ignore[no-untyped-def]
    responder = FakeResponder()
    _, _, _, _, handler = setup(responder=responder)
    variants = [
        PreviewOutcome(sent=True, text="Fine.", reason="send:mention", reasons=("style:x",)),
        PreviewOutcome(sent=False, text="", reason="llm:silent"),
        PreviewOutcome(sent=False, text="raw", reason="llm:http"),
        PreviewOutcome(
            sent=False,
            text="cut me",
            reason="regex:length",
            reasons=("regex:length",),
            fixes=("fix:dash",),
        ),
    ]
    for index, outcome in enumerate(variants):
        responder.preview_result = outcome
        await _run(handler, "/ask hello", message_id=20 + index)
    assert sent[0].splitlines() == [
        "Would answer: Fine.",
        "The filter flagged it but let it through (shadow): style:x",
        "(nothing was sent to the chat)",
    ]
    assert sent[1].splitlines() == [
        "Would stay silent (the model chose not to answer).",
        "(nothing was sent to the chat)",
    ]
    assert sent[2].splitlines()[:2] == ["Would not answer: llm:http.", "The model returned: raw"]
    assert sent[3].splitlines() == [
        "Would not answer, the filter would cut it: regex:length.",
        "Candidate: cut me",
        "Fixes: fix:dash",
        "(nothing was sent to the chat)",
    ]


async def test_owner_errors_render_through_the_english_catalog(setup, sent: list[str]) -> None:  # type: ignore[no-untyped-def]
    _, _, store, _, handler = setup()

    async def refuse(key: str, value: str, changed_by: int, now: int):  # type: ignore[no-untyped-def]
        from sotto.errors import OwnerError

        raise OwnerError("set_persona")

    store.set = refuse  # type: ignore[method-assign]
    await _run(handler, "/set persona.name x")
    assert sent == ["Error: persona can only be changed in config.yaml"]


async def test_changelog_in_english(setup, sent: list[str]) -> None:  # type: ignore[no-untyped-def]
    deps, _, _, _, handler = setup()
    deps.persona = replace(
        deps.persona,
        chat_notes=(
            "## [1.0.0] - 2026-01-01\n\nFresh news\n\n## [0.9.0] - 2025-12-01\n\nOlder news\n"
        ),
        legacy_changelog=(
            "## [1.0.0] - 2026-01-01\n\n### Для чата\n\nignored\n\n"
            "### Для владельца\n\nOwner text\n"
        ),
    )
    await _run(handler, "/changelog")
    await _run(handler, "/changelog v0.9.0", message_id=11)
    await _run(handler, "/changelog owner", message_id=12)
    await _run(handler, "/changelog 9.9.9", message_id=13)
    assert sent == [
        "v1.0.0 (2026-01-01)\nFresh news",
        "v0.9.0 (2025-12-01)\nOlder news",
        "v1.0.0 (2026-01-01)\nOwner text",
        "No version 9.9.9.",
    ]
    deps.persona = replace(deps.persona, chat_notes="", legacy_changelog="")
    await _run(handler, "/changelog", message_id=14)
    await _run(handler, "/changelog owner", message_id=15)
    assert sent[4:] == ["CHANGELOG.md not found or empty."] * 2


async def test_words_in_english(setup, sent: list[str]) -> None:  # type: ignore[no-untyped-def]
    from sotto.db import VocabDictRow
    from sotto.vocab import VocabDict, VocabWord

    class Store:
        def __init__(self) -> None:
            word = VocabWord(word="lorem", meaning="ipsum")
            self.rows = [
                (
                    VocabDictRow(
                        name="demo", body="", enabled=True, updated_at=int(NOW.timestamp())
                    ),
                    VocabDict(
                        name="demo",
                        about="a test",
                        words=[word, word.model_copy(update={"word": "dolor"})],
                    ),
                ),
                (
                    VocabDictRow(
                        name="odd", body="", enabled=False, updated_at=int(NOW.timestamp())
                    ),
                    None,
                ),
            ]

        def get(self):  # type: ignore[no-untyped-def]
            raise AssertionError

        async def dicts(self):  # type: ignore[no-untyped-def]
            return self.rows

        async def remove(self, name: str) -> bool:
            return True

        async def set_enabled(self, name: str, enabled: bool) -> bool:
            return True

    deps, _, _, _, handler = setup()
    deps.vocab_store = Store()
    for index, command in enumerate(
        [
            "/words",
            "/words show demo",
            "/words show odd",
            "/words off demo",
            "/words on demo",
            "/words rm demo",
        ]
    ):
        await _run(handler, command, message_id=30 + index)
    first = sent[0].splitlines()
    assert first[0] == "demo: 2 words, on, 10.01"
    assert first[1] == "odd: cannot be parsed, off, 10.01"
    assert first[2].startswith("Usage: /words | /words show|on|off|rm <name>.")
    assert sent[1].splitlines()[0] == "demo: 2 words (a test)"
    assert sent[2] == 'Dictionary "odd" cannot be parsed, upload it again.'
    assert sent[3:] == [
        'Dictionary "demo" disabled.',
        'Dictionary "demo" enabled.',
        'Dictionary "demo" deleted.',
    ]
