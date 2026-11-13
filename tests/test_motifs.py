"""motifs.py on the demo pack: character props, the story quota and the "about himself" metric.

The Russian-specific parts (declension of the labels, Cyrillic markers) are in
``tests/lang/test_ru_motifs.py``.
"""

from __future__ import annotations

import re

import pytest

from sotto.motifs import (
    SelfShare,
    motifs_in,
    render_avoid,
    render_self_share,
    self_share,
    speaks_about_self,
    story_count,
    used_motifs,
)
from sotto.patterns import Patterns
from tests.demo import DEMO_FRAG, EN, demo_config


def _patterns() -> Patterns:
    return Patterns(demo_config().filters, ["gus"], "", EN)


def _motifs() -> dict[str, list[re.Pattern[str]]]:
    return _patterns().motifs


def _story_markers() -> list[re.Pattern[str]]:
    return _patterns().story_markers


# --- motifs_in ---


def test_motifs_in_finds_labels_in_config_order() -> None:
    text = "Sat in the shed, then my sister called about the clocks."

    assert motifs_in(text, _motifs()) == ["clocks", "shed", "sister"]


def test_motifs_in_empty_text_is_empty() -> None:
    assert motifs_in("", _motifs()) == []


def test_motifs_in_nothing_matched_is_empty() -> None:
    assert motifs_in("A quiet weekend, all told.", _motifs()) == []


def test_motifs_in_is_case_insensitive() -> None:
    assert motifs_in("THE TUBA is dented.", _motifs()) == ["tuba"]


def test_motifs_in_word_boundary_for_clocks() -> None:
    """«clock» and «clocks» are a motif, «clockwork oranges» only by its own regexes."""
    assert motifs_in("The clock stopped.", _motifs()) == ["clocks"]
    assert motifs_in("Two clocks on the wall.", _motifs()) == ["clocks"]
    assert "clocks" not in motifs_in("He was a bit of a clockwatcher.", _motifs())


def test_motifs_in_nineties_year_form() -> None:
    assert motifs_in("That was in 1994.", _motifs()) == ["nineties"]


# --- used_motifs ---


def test_used_motifs_only_looks_at_window_tail() -> None:
    replies = ["My sister said so.", "The tuba leaked.", "We went quiet.", "The keys turned up."]

    assert used_motifs(replies, 2, _motifs(), frag=DEMO_FRAG) == []
    assert used_motifs(replies, 4, _motifs(), frag=DEMO_FRAG) == ["tuba", "sister"]


def test_used_motifs_zero_window_is_empty() -> None:
    assert used_motifs(["My sister said so."], 0, _motifs(), frag=DEMO_FRAG) == []


def test_used_motifs_deduplicates_labels() -> None:
    replies = ["My sister said so.", "Maureen did not like it."]

    assert used_motifs(replies, 10, _motifs(), frag=DEMO_FRAG) == ["sister"]


def test_used_motifs_skips_sticker_records() -> None:
    """A sticker record is a picture, not a reply of the persona: no motifs in it."""
    replies = ["[sticker #3] My sister said so", "We went quiet."]

    assert used_motifs(replies, 10, _motifs(), frag=DEMO_FRAG) == []


# --- story_count ---


def test_story_count_counts_replies_not_matches() -> None:
    replies = [
        "I remember, back in the nineties it was the same.",
        "We went quiet.",
        "One time I got lucky.",
    ]

    assert story_count(replies, 10, _story_markers(), frag=DEMO_FRAG) == 2


def test_story_count_respects_window() -> None:
    replies = ["I remember that.", "We went quiet.", "The keys turned up."]

    assert story_count(replies, 2, _story_markers(), frag=DEMO_FRAG) == 0
    assert story_count(replies, 3, _story_markers(), frag=DEMO_FRAG) == 1


def test_story_count_skips_sticker_records() -> None:
    assert story_count(["[sticker #1] I remember that"], 10, _story_markers(), frag=DEMO_FRAG) == 0


def test_story_count_zero_window_is_zero() -> None:
    assert story_count(["I remember that."], 0, _story_markers(), frag=DEMO_FRAG) == 0


# --- render_avoid ---


def test_render_avoid_empty_when_nothing_to_say() -> None:
    assert render_avoid([], no_story=False, frag=DEMO_FRAG) == ""


def test_render_avoid_names_known_labels() -> None:
    text = render_avoid(["clocks", "tuba", "bus"], no_story=False, frag=DEMO_FRAG)

    assert "already mentioned: the clocks, the tuba, the bus" in text
    assert "another detail, or no anecdote at all" in text


def test_render_avoid_unknown_label_goes_as_is() -> None:
    assert "fishing" in render_avoid(["fishing"], no_story=False, frag=DEMO_FRAG)


def test_render_avoid_no_story_only() -> None:
    text = render_avoid([], no_story=True, frag=DEMO_FRAG)

    assert text.startswith("Do not tell an anecdote now")
    assert "already mentioned" not in text


def test_render_avoid_both_parts() -> None:
    text = render_avoid(["bus"], no_story=True, frag=DEMO_FRAG)

    assert "already mentioned: the bus" in text
    assert "Do not tell an anecdote now" in text


def test_render_avoid_has_no_typographic_dash() -> None:
    """The model copies the typography of the prompt: no long dashes in it."""
    text = render_avoid(["clocks"], no_story=True, frag=DEMO_FRAG)

    assert "—" not in text
    assert "–" not in text


# --- the "about himself" metric ---


def _self_markers() -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in demo_config().filters.self_markers]


@pytest.mark.parametrize(
    "text",
    [
        "We argued about the same thing at the depot, over the bonus",
        "I kept the foxes out of the shed all winter",
        "That bolt now lives its own life, I have a few in a drawer",
        "My sister does not eat onions, but she will empty a jam jar in an evening",
        "Mine do not take part in this discussion",
        "I'm not fussed, I take one cup",
        "Our milk came from the farm",
        "I spent an hour looking for my keys",
        "Maureen rang yesterday",
    ],
)
def test_speaks_about_self_catches_real_self_replies(text: str) -> None:
    assert speaks_about_self(text, _self_markers(), frag=DEMO_FRAG)


@pytest.mark.parametrize(
    "text",
    [
        "Second time is a routine. Third time, bring cake.",
        "Four fifty, but the foam is free.",
        "[sticker #3] I love you",
        "",
    ],
)
def test_speaks_about_self_ignores_other_replies(text: str) -> None:
    assert not speaks_about_self(text, _self_markers(), frag=DEMO_FRAG)


def test_self_share_groups_and_skips() -> None:
    share = self_share(
        [
            ("ambient", "We argued at the depot"),
            ("ambient", "Plus nine is still bracing"),
            ("mention", "I did not check"),
            ("followup", "It happens"),
            ("places", "We sat there"),
            ("checkin", "mine too"),
            ("say", "I am off to the shed"),
            ("life", "I sold the bus"),
            ("ambient", "[sticker #1] I"),
        ],
        _self_markers(),
        frag=DEMO_FRAG,
    )
    assert share == SelfShare(ambient=(1, 2), address=(2, 3), other=(1, 1))


def test_render_self_share() -> None:
    share = SelfShare(ambient=(6, 6), address=(2, 2), other=(0, 1))
    assert render_self_share(share, 3, frag=DEMO_FRAG) == (
        "about himself over 3 days: ambient 6/6, addressed 2/2, other 0/1"
    )
