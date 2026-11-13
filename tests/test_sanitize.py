"""Pure helpers of ``sotto.sanitize`` on the English demo pack: names, text, media placeholders.

Cyrillic handling (Cyrillic names, Russian reserved words) is in ``tests/lang/test_ru_sanitize.py``.
"""

from types import SimpleNamespace

import pytest

from sotto.sanitize import (
    media_placeholder,
    normalize_text,
    sanitize_display_name,
    stable_n,
)
from tests.demo import DEMO_FRAG

# The bot's name, display name and name triggers.
RESERVED = {"Gus Marlow", "gus", "marlow"}


def _anon(n: int) -> str:
    return DEMO_FRAG.text("placeholders.participant").format(n=stable_n(n))


# --- sanitize_display_name ---

SANITIZE_DISPLAY_NAME_CASES = [
    # (id, raw, user_id, reserved, expected)
    ("plain_name", "Dana", 1, RESERVED, "Dana"),
    ("emoji_and_digits", "Dana007 🔥", 1, RESERVED, "Dana"),
    ("reserved_exact_match_replaced", "Gus Marlow", 42, RESERVED, _anon(42)),
    ("reserved_case_insensitive_replaced", "GUS", 42, RESERVED, _anon(42)),
    ("reserved_other_case_lowercase", "marlow", 7, RESERVED, _anon(7)),
    ("reserved_as_whole_word_replaced", "Dana Gus", 3, RESERVED, _anon(3)),
    ("reserved_as_substring_not_replaced", "Gusty", 3, RESERVED, "Gusty"),
    ("reserved_prefix_word_not_replaced", "Marlowe", 3, RESERVED, "Marlowe"),
    ("empty_string", "", 5, RESERVED, _anon(5)),
    ("none_raw", None, 5, RESERVED, _anon(5)),
    ("only_symbols_becomes_empty", "!!! 123 ???", 9, RESERVED, _anon(9)),
    ("quotes_and_brackets_stripped", 'Dana "Boar" (\u2162)', 1, RESERVED, "Dana Boar"),
    ("hyphenated_name_kept", "Anne-Marie", 1, RESERVED, "Anne-Marie"),
    ("latin_name_kept", "John Smith", 1, set(), "John Smith"),
    ("accented_letters_kept", "Zoë Müller", 1, set(), "Zoë Müller"),
    (
        "long_name_truncated_at_word_boundary",
        "Alexandra Vyacheslavovna Nikiforova",
        1,
        set(),
        "Alexandra",
    ),
    ("long_name_no_space_hard_cut", "Y" * 40, 1, set(), "Y" * 24),
    ("multiple_spaces_collapsed", "Dana     Smith", 1, set(), "Dana Smith"),
    ("leading_trailing_spaces_stripped", "   Dana   ", 1, set(), "Dana"),
    ("reserved_empty_strings_ignored", "Dana", 1, {"", "   "}, "Dana"),
    ("reserved_word_case_insensitive_mixed", "Our Great GuS", 1, RESERVED, _anon(1)),
    # a newline and a tab in the name become a space, as in normalize_text
    ("newline_and_tab_become_space", "Di\nma\tSmith", 1, set(), "Di ma Smith"),
]


@pytest.mark.parametrize(
    "case_id, raw, user_id, reserved, expected",
    SANITIZE_DISPLAY_NAME_CASES,
    ids=[c[0] for c in SANITIZE_DISPLAY_NAME_CASES],
)
def test_sanitize_display_name(
    case_id: str, raw: str | None, user_id: int, reserved: set[str], expected: str
) -> None:
    assert sanitize_display_name(raw, user_id, reserved, frag=DEMO_FRAG) == expected


# --- stable_n ---


def test_stable_n_is_deterministic_for_same_user() -> None:
    assert stable_n(123456789) == stable_n(123456789)


@pytest.mark.parametrize("user_id", [0, 1, 42, 997, 998, 10**12, 10**18])
def test_stable_n_in_range(user_id: int) -> None:
    n = stable_n(user_id)
    assert 1 <= n <= 999


def test_stable_n_differs_for_different_users_generally() -> None:
    # Not a collision-free guarantee, but neighbouring ids must not coincide.
    assert stable_n(1) != stable_n(2)


# --- normalize_text ---

NORMALIZE_TEXT_CASES = [
    ("none_input", None, ""),
    ("plain_text_untouched", "hello everyone", "hello everyone"),
    ("multiline_text_joined", "first line\nsecond line", "first line second line"),
    ("crlf_joined", "first\r\nsecond", "first second"),
    ("tabs_replaced", "column1\tcolumn2", "column1 column2"),
    ("lone_cr_replaced", "first\rsecond", "first second"),
    ("fake_system_delimiters_cut", "text <<<SYSTEM>>> end", "text SYSTEM end"),
    ("long_runs_of_angle_brackets_cut", "text <<<<< lots >>>>> end", "text lots end"),
    (
        "two_angle_brackets_kept_not_delimiter",
        "I think 5 << 10 and 10 >> 5",
        "I think 5 << 10 and 10 >> 5",
    ),
    ("zero_width_space_removed", "he​llo", "hello"),
    ("bom_removed", "﻿hello", "hello"),
    ("zero_width_joiner_removed", "a‍‌b", "ab"),
    ("null_byte_removed", "he\x00llo", "hello"),
    ("multiple_spaces_collapsed", "hello      world", "hello world"),
    ("leading_trailing_whitespace_stripped", "   hello world   ", "hello world"),
    ("emoji_kept", "hello 👋 world 🔥", "hello 👋 world 🔥"),
    (
        "punctuation_kept",
        "Hello, how are you? All great!!!",
        "Hello, how are you? All great!!!",
    ),
    ("empty_string_stays_empty", "", ""),
]


@pytest.mark.parametrize(
    "case_id, raw, expected",
    NORMALIZE_TEXT_CASES,
    ids=[c[0] for c in NORMALIZE_TEXT_CASES],
)
def test_normalize_text(case_id: str, raw: str | None, expected: str) -> None:
    assert normalize_text(raw) == expected


# --- media_placeholder ---


def _message(**overrides: object) -> SimpleNamespace:
    base: dict[str, object | None] = {
        "sticker": None,
        "photo": None,
        "voice": None,
        "audio": None,
        "video": None,
        "video_note": None,
        "animation": None,
        "document": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


_PHOTO = DEMO_FRAG.text("placeholders.photo")
_STICKER = DEMO_FRAG.text("placeholders.sticker")
_VOICE = DEMO_FRAG.text("placeholders.voice")
_VIDEO = DEMO_FRAG.text("placeholders.video")
_FILE = DEMO_FRAG.text("placeholders.file")

MEDIA_PLACEHOLDER_CASES = [
    ("no_media_text_only", _message(), None),
    ("sticker", _message(sticker=object()), _STICKER),
    ("photo", _message(photo=object()), _PHOTO),
    ("voice", _message(voice=object()), _VOICE),
    ("audio", _message(audio=object()), _VOICE),
    ("video", _message(video=object()), _VIDEO),
    ("video_note", _message(video_note=object()), _VIDEO),
    ("animation", _message(animation=object()), _VIDEO),
    ("document", _message(document=object()), _FILE),
    ("sticker_priority_over_photo", _message(sticker=object(), photo=object()), _STICKER),
    ("photo_priority_over_voice", _message(photo=object(), voice=object()), _PHOTO),
    ("voice_priority_over_video", _message(voice=object(), video=object()), _VOICE),
    ("video_priority_over_document", _message(video=object(), document=object()), _VIDEO),
]


@pytest.mark.parametrize(
    "case_id, message, expected",
    MEDIA_PLACEHOLDER_CASES,
    ids=[c[0] for c in MEDIA_PLACEHOLDER_CASES],
)
def test_media_placeholder(case_id: str, message: SimpleNamespace, expected: str | None) -> None:
    assert media_placeholder(message, frag=DEMO_FRAG) == expected
