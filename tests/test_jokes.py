"""Tests for chat jokes: pure functions."""

from __future__ import annotations

from sotto.db import JokeRow
from sotto.jokes import joke_used, render_jokes
from tests.demo import DEMO_FRAG, EN

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
    assert render_jokes([], now=NOW, cooldown_days=5, frag=DEMO_FRAG) == ""


def test_render_jokes_single_line_with_quotes() -> None:
    rendered = render_jokes(
        [_joke(1, "those screws again"), _joke(2, "like Vic's")],
        now=NOW,
        cooldown_days=5,
        frag=DEMO_FRAG,
    )

    assert "\n" not in rendered
    assert rendered.startswith(DEMO_FRAG.text("render.jokes_header"))
    assert rendered.endswith("\u201cthose screws again\u201d; \u201clike Vic's\u201d.")


def test_render_jokes_hides_recently_used_shows_after_cooldown() -> None:
    rows = [
        _joke(1, "fresh"),
        _joke(2, "recent", used_at=NOW - 2 * DAY),
        _joke(3, "ancient", used_at=NOW - 5 * DAY),
    ]

    rendered = render_jokes(rows, now=NOW, cooldown_days=5, frag=DEMO_FRAG)

    assert "\u201cfresh\u201d" in rendered
    assert "\u201cancient\u201d" in rendered
    assert "recent" not in rendered


def test_render_jokes_all_cooling_down_is_empty() -> None:
    rows = [_joke(1, "recent", used_at=NOW - DAY)]
    assert render_jokes(rows, now=NOW, cooldown_days=5, frag=DEMO_FRAG) == ""


def test_render_jokes_zero_cooldown_always_shows() -> None:
    rows = [_joke(1, "just now", used_at=NOW)]
    assert "\u201cjust now\u201d" in render_jokes(rows, now=NOW, cooldown_days=0, frag=DEMO_FRAG)


def test_render_jokes_cleans_bullets_and_delimiters_and_never_starts_with_dash() -> None:
    rendered = render_jokes(
        [_joke(1, "- those <<<screws>>> again"), _joke(2, "  ")],
        now=NOW,
        cooldown_days=5,
        frag=DEMO_FRAG,
    )

    assert "<<<" not in rendered
    assert ">>>" not in rendered
    assert "\u201cthose screws again\u201d" in rendered
    assert "\u201c\u201d" not in rendered
    assert all(not line.startswith("- ") for line in rendered.splitlines())


def test_joke_used_three_common_words_in_a_row() -> None:
    assert joke_used("Well, Mike's screws in the wall again!", "Mike's screws in", EN)
    assert joke_used("mike's screws in the wall, I recalled", "That is Mike's screws in, yes", EN)


def test_joke_used_needs_consecutive_words() -> None:
    assert not joke_used("again some screws of Mike's", "again Mike's screws", EN)


def test_joke_used_two_common_words_not_enough_for_long_joke() -> None:
    assert not joke_used("again Mike problems", "again Mike screws in the wall", EN)


def test_joke_used_short_joke_must_be_contained_entirely() -> None:
    assert joke_used("Well, like Vic's, yes.", "like Vic's", EN)
    assert joke_used("Vic came back", "Vic", EN)
    assert not joke_used("like Sam's", "like Vic's", EN)
    assert not joke_used("Vickers is not here", "Vic", EN)  # whole word, not a substring


def test_joke_used_normalizes_case_and_punctuation() -> None:
    assert joke_used("LIKE VIC'S!!!", "like Vic's", EN)


def test_joke_used_empty_inputs() -> None:
    assert not joke_used("", "those screws again", EN)
    assert not joke_used("those screws again", "", EN)
    assert not joke_used("...", "!!!", EN)


def test_joke_used_folds_with_the_language() -> None:
    assert joke_used("Again Mike's screws in the wall", "again Mike's screws", EN)
    assert not joke_used("screws Mike's again", "again Mike's screws", EN)
