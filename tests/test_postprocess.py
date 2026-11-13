"""Tests for sotto.postprocess: normalize_dashes, strip_emoji, keep_first_emoji, soften.

Pure functions, no I/O, like the module itself.
"""

from __future__ import annotations

from sotto.config_models import FiltersConfig
from sotto.postprocess import (
    Fixed,
    keep_first_emoji,
    normalize_dashes,
    soften,
    strip_emoji,
)
from tests.demo import DEMO_FRAG

# --------------------------------------------------------------------------- #
# normalize_dashes
# --------------------------------------------------------------------------- #


def test_normalize_dashes_em_dash() -> None:
    assert normalize_dashes("word — word") == "word - word"


def test_normalize_dashes_en_dash() -> None:
    assert normalize_dashes("word – word") == "word - word"


def test_normalize_dashes_figure_dash() -> None:
    assert normalize_dashes("word ‒ word") == "word - word"


def test_normalize_dashes_horizontal_bar() -> None:
    assert normalize_dashes("word ― word") == "word - word"


def test_normalize_dashes_hyphen_in_word_untouched() -> None:
    assert normalize_dashes("well-known someone-else") == "well-known someone-else"


def test_normalize_dashes_no_dash_unchanged() -> None:
    assert normalize_dashes("It happens to the best of us.") == "It happens to the best of us."


def test_normalize_dashes_multiple() -> None:
    assert normalize_dashes("one — two – three") == "one - two - three"


# --------------------------------------------------------------------------- #
# strip_emoji
# --------------------------------------------------------------------------- #


def test_strip_emoji_at_end() -> None:
    assert strip_emoji("It happens 🙂") == "It happens"


def test_strip_emoji_at_start() -> None:
    assert strip_emoji("🙂 It happens") == "It happens"


def test_strip_emoji_in_middle() -> None:
    assert strip_emoji("It happens 🙂 hello") == "It happens hello"


def test_strip_emoji_before_punctuation_no_hanging_space() -> None:
    assert strip_emoji("It happens 🙂, how are you") == "It happens, how are you"


def test_strip_emoji_after_dot() -> None:
    assert strip_emoji("It happens. 💩") == "It happens."


def test_strip_emoji_with_variation_selector() -> None:
    assert strip_emoji("It happens 👍️") == "It happens"


def test_strip_emoji_with_skin_tone_modifier() -> None:
    assert strip_emoji("It happens 👍🏻") == "It happens"


def test_strip_emoji_multiple() -> None:
    assert strip_emoji("It happens 🙂🙂") == "It happens"


def test_strip_emoji_no_emoji_unchanged() -> None:
    assert strip_emoji("It happens to the best of us.") == "It happens to the best of us."


def test_strip_emoji_no_double_space_result() -> None:
    result = strip_emoji("It happens 🙂 hello")
    assert "  " not in result


# --------------------------------------------------------------------------- #
# keep_first_emoji
# --------------------------------------------------------------------------- #


def test_keep_first_emoji_two() -> None:
    assert keep_first_emoji("It happens 🙂🙂") == "It happens 🙂"


def test_keep_first_emoji_far_apart() -> None:
    assert keep_first_emoji("It happens 🙂 and here 💩 too") == "It happens 🙂 and here too"


def test_keep_first_emoji_single_unchanged() -> None:
    assert keep_first_emoji("It happens 🙂") == "It happens 🙂"


def test_keep_first_emoji_none_unchanged() -> None:
    assert keep_first_emoji("It happens.") == "It happens."


# --------------------------------------------------------------------------- #
# soften
# --------------------------------------------------------------------------- #

_CFG = FiltersConfig()  # default allowed_emoji, emoji_max_per_reply=1, emoji_recent_window=4


def test_soften_dash_only() -> None:
    fixed = soften(
        "It used to be easier — now it is not.", recent_replies=[], cfg=_CFG, frag=DEMO_FRAG
    )
    assert fixed == Fixed(text="It used to be easier - now it is not.", fixes=("fix:dash",))


def test_soften_no_fixes_needed() -> None:
    fixed = soften("It happens to the best of us.", recent_replies=[], cfg=_CFG, frag=DEMO_FRAG)
    assert fixed == Fixed(text="It happens to the best of us.", fixes=())


def test_soften_emoji_freq_strips_when_recent_has_emoji() -> None:
    recent = ["Everyone home.", "It happens.", "Fine.", "Pointless 😂"]
    fixed = soften("It happens 🙂", recent_replies=recent, cfg=_CFG, frag=DEMO_FRAG)
    assert fixed.fixes == ("fix:emoji_freq",)
    assert fixed.text == "It happens"


def test_soften_emoji_freq_outside_window_pass() -> None:
    recent = [
        "Pointless 😂",
        "Everyone went home.",
        "Fine, I am in the shed.",
        "Bought peat for the seedlings.",
        "Good night, all.",
    ]
    fixed = soften("It happens 🙂", recent_replies=recent, cfg=_CFG, frag=DEMO_FRAG)
    assert fixed.fixes == ()
    assert fixed.text == "It happens 🙂"


def test_soften_emoji_count_keeps_first() -> None:
    fixed = soften("It happens 🙂🙂", recent_replies=[], cfg=_CFG, frag=DEMO_FRAG)
    assert fixed.fixes == ("fix:emoji_count",)
    assert fixed.text == "It happens 🙂"


def test_soften_order_dash_then_emoji_freq() -> None:
    recent = ["Pointless 😂"]
    fixed = soften("It used to be — now 🙂", recent_replies=recent, cfg=_CFG, frag=DEMO_FRAG)
    assert fixed.fixes == ("fix:dash", "fix:emoji_freq")
    assert fixed.text == "It used to be - now"


def test_soften_emoji_freq_and_count_only_freq_applies() -> None:
    # emoji_freq is checked first: when it fires, emoji_count is never reached,
    # even if the candidate has several allowed emoji.
    recent = ["Pointless 😂"]
    fixed = soften("It happens 🙂🙂", recent_replies=recent, cfg=_CFG, frag=DEMO_FRAG)
    assert fixed.fixes == ("fix:emoji_freq",)
    assert fixed.text == "It happens"


def test_soften_sticker_placeholder_not_counted_as_emoji() -> None:
    recent = ["[sticker #3] Here we go"]
    fixed = soften("It happens 🙂", recent_replies=recent, cfg=_CFG, frag=DEMO_FRAG)
    assert fixed.fixes == ()
    assert fixed.text == "It happens 🙂"


def test_soften_window_respects_emoji_recent_window() -> None:
    cfg = FiltersConfig(emoji_recent_window=1)
    # the emoji was 2 replies ago, window = 1 -> not counted.
    recent = ["Pointless 😂", "Fine."]
    fixed = soften("It happens 🙂", recent_replies=recent, cfg=cfg, frag=DEMO_FRAG)
    assert fixed.fixes == ()
    assert fixed.text == "It happens 🙂"


def test_soften_no_emoji_no_dash_unchanged() -> None:
    fixed = soften(
        "The keys were in the shed.", recent_replies=["Pointless 😂"], cfg=_CFG, frag=DEMO_FRAG
    )
    assert fixed == Fixed(text="The keys were in the shed.", fixes=())
