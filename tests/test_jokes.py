"""Тесты шуток чата (CLAUDE.md, "шутки чата и истории людей"): чистые функции."""

from __future__ import annotations

from sotto.db import JokeRow
from sotto.jokes import joke_used, render_jokes
from tests.packs import LEGACY_FRAG

DAY = 86400
NOW = 100 * DAY


def _joke(joke_id: int, text: str, used_at: int | None = None) -> JokeRow:
    return JokeRow(
        id=joke_id,
        text=text,
        created_at=NOW - 30 * DAY,
        last_used_at=used_at,
        uses=0 if used_at is None else 1,
    )


def test_render_jokes_empty_is_empty_string() -> None:
    assert render_jokes([], now=NOW, cooldown_days=5, frag=LEGACY_FRAG) == ""


def test_render_jokes_single_line_with_quotes() -> None:
    rendered = render_jokes(
        [_joke(1, "опять шурупы"), _joke(2, "как у Вовы")],
        now=NOW,
        cooldown_days=5,
        frag=LEGACY_FRAG,
    )

    assert "\n" not in rendered
    assert rendered.startswith("Шутки и словечки вашего чата")
    assert rendered.endswith("«опять шурупы»; «как у Вовы».")


def test_render_jokes_hides_recently_used_shows_after_cooldown() -> None:
    rows = [
        _joke(1, "свежая"),
        _joke(2, "недавно", used_at=NOW - 2 * DAY),
        _joke(3, "давно", used_at=NOW - 5 * DAY),
    ]

    rendered = render_jokes(rows, now=NOW, cooldown_days=5, frag=LEGACY_FRAG)

    assert "«свежая»" in rendered
    assert "«давно»" in rendered
    assert "недавно" not in rendered


def test_render_jokes_all_cooling_down_is_empty() -> None:
    rows = [_joke(1, "недавно", used_at=NOW - DAY)]
    assert render_jokes(rows, now=NOW, cooldown_days=5, frag=LEGACY_FRAG) == ""


def test_render_jokes_zero_cooldown_always_shows() -> None:
    rows = [_joke(1, "только что", used_at=NOW)]
    assert "«только что»" in render_jokes(rows, now=NOW, cooldown_days=0, frag=LEGACY_FRAG)


def test_render_jokes_cleans_bullets_and_delimiters_and_never_starts_with_dash() -> None:
    rendered = render_jokes(
        [_joke(1, "- опять <<<шурупы>>>"), _joke(2, "  ")],
        now=NOW,
        cooldown_days=5,
        frag=LEGACY_FRAG,
    )

    assert "<<<" not in rendered
    assert ">>>" not in rendered
    assert "«опять шурупы»" in rendered
    assert "«»" not in rendered
    assert all(not line.startswith("- ") for line in rendered.splitlines())


def test_joke_used_three_common_words_in_a_row() -> None:
    assert joke_used("Ну вот, опять Мишины шурупы в стене!", "опять Мишины шурупы")
    assert joke_used("опять мишины шурупы вспомнил", "Это опять Мишины шурупы, да")


def test_joke_used_needs_consecutive_words() -> None:
    assert not joke_used("опять про какие-то шурупы Мишины", "опять Мишины шурупы")


def test_joke_used_two_common_words_not_enough_for_long_joke() -> None:
    assert not joke_used("опять Мишины проблемы", "опять Мишины шурупы в стене")


def test_joke_used_short_joke_must_be_contained_entirely() -> None:
    assert joke_used("Ну как у Вовы, да.", "как у Вовы")
    assert joke_used("Вова вернулся", "Вова")
    assert not joke_used("как у Васи", "как у Вовы")
    assert not joke_used("Вовы тут нет", "Вова")  # целое слово, не подстрока


def test_joke_used_normalizes_case_punctuation_and_yo() -> None:
    assert joke_used("КАК У ВОВЫ!!!", "как у Вовы")


def test_joke_used_empty_inputs() -> None:
    assert not joke_used("", "опять шурупы")
    assert not joke_used("опять шурупы", "")
    assert not joke_used("...", "!!!")
