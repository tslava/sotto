"""Vocabulary words (vocab.py) on the English demo pack: parsing, offers, rendering, the store.

Russian morphology and Cyrillic folding are in ``tests/lang/test_ru_vocab*.py``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pytest

from sotto.config_models import VocabConfig
from sotto.db import Database
from sotto.errors import OwnerError
from sotto.i18n import engine_catalog
from sotto.vocab import (
    Vocab,
    VocabStore,
    explain_requested,
    normalize,
    offer_allowed,
    pick_offer,
    render_vocab,
    vocab_over_quota,
    vocab_requested,
)
from tests.demo import DEMO_FRAG, EN, demo_config, en_vocab, parse_en_vocab

HILL = """
name: hillfolk
about: words they use on the hill
words:
  - word: Snickerwick
    meaning: a short rain shower that stops before you find your umbrella
    example: Just a snickerwick, give it five minutes.
  - word: bodge
    meaning: to mend something badly but well enough
  - word: hobnail
    meaning: to walk about on pointless errands
  - word: gossamer
    meaning: a thin, fine cobweb
  - word: dreep
    meaning: a person who talks a lot
  - word: half a mo
    meaning: just a moment
  - word: midge
    meaning: a tiny biting fly, a pest
  - word: scrump
    meaning: to steal apples
  - word: tuppenny
    meaning: cheap and cheerful
    forms: [tuppenny, tuppennies]
"""


# Validation errors are OwnerErrors; the owner text comes from the English catalog.
_ERR_SHAPE = "fields name and words"
_ERR_MEANING = "word #1 → meaning"
_ERR_NO_LETTERS = "has no letters"
_ERR_REPEATED = "is repeated"


def _vocab() -> Vocab:
    return en_vocab([parse_en_vocab(HILL)])


def _words(vocab: Vocab, text: str) -> list[str]:
    return [entry.word for entry in vocab.find(text)]


# --- parsing the file ---


def test_parse_vocab_keeps_text_and_folds_for_matching() -> None:
    parsed = parse_en_vocab(HILL)
    assert parsed.name == "hillfolk"
    assert len(parsed.words) == 9
    assert normalize("Snickerwick  STRESS", EN) == "snickerwick stress"


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        ("name: [", "YAML"),
        ("- just a list", _ERR_SHAPE),
        ("name: x\nwords: []", "words"),
        ("words:\n  - word: a\n    meaning: b", "name"),
        ("name: x\nwords:\n  - word: snickerwick", _ERR_MEANING),
        ("name: x\nwords:\n  - word: '!!!'\n    meaning: b", _ERR_NO_LETTERS),
        (
            "name: x\nwords:\n  - {word: Snickerwick, meaning: a}\n  - {word: snickerwick, meaning: b}",  # noqa: E501
            _ERR_REPEATED,
        ),
        ("name: x\nwords:\n  - {word: a, meaning: b, meanning: c}", "meanning"),
    ],
)
def test_parse_vocab_errors_are_readable(raw: str, needle: str) -> None:
    with pytest.raises(ValueError) as info:
        parse_en_vocab(raw)
    exc = info.value
    text = exc.render(engine_catalog()) if isinstance(exc, OwnerError) else str(exc)
    assert needle in text


# --- finding words in a text ---


def test_find_orders_by_first_appearance_without_duplicates() -> None:
    assert _words(_vocab(), "Bodge it, snickerwick too, bodge again, then hobnail") == [
        "bodge",
        "Snickerwick",
        "hobnail",
    ]


def test_find_matches_plain_english_endings() -> None:
    assert _words(_vocab(), "he bodged it and she was hobnailing about") == ["bodge", "hobnail"]


def test_find_does_not_match_a_longer_word() -> None:
    assert _words(_vocab(), "the midgets and a bodger") == []


def test_find_multiword_entry_needs_the_exact_sequence() -> None:
    assert _words(_vocab(), "give me half a mo") == ["half a mo"]
    assert _words(_vocab(), "half of a mo") == []


def test_find_with_explicit_forms_matches_only_those() -> None:
    assert _words(_vocab(), "two tuppennies") == ["tuppenny"]
    assert _words(_vocab(), "tuppenny things") == ["tuppenny"]
    assert _words(_vocab(), "tuppennyish") == []


def test_duplicate_word_in_second_dict_is_ignored() -> None:
    other = parse_en_vocab(
        "name: others\nwords:\n  - {word: bodge, meaning: somebody else's meaning}"
    )
    vocab = en_vocab([parse_en_vocab(HILL), other])
    assert vocab.find("bodge")[0].meaning == "to mend something badly but well enough"
    assert len(vocab.entries) == 9


def test_empty_vocab_is_falsy_and_finds_nothing() -> None:
    vocab = en_vocab()
    assert not vocab
    assert vocab.find("snickerwick") == []


# --- when to offer ---


class _Rng(random.Random):
    """Records whether the dice were rolled."""

    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value
        self.rolled = 0

    def random(self) -> float:
        self.rolled += 1
        return self.value


def test_offer_allowed_rolls_dice_last() -> None:
    cfg = VocabConfig(probability=0.15, daily_cap=2, recent_window=3)
    vocab = _vocab()
    rng = _Rng(0.1)
    assert offer_allowed(
        cfg=cfg, vocab=vocab, recent_replies=[], count_today=0, rng=rng, frag=DEMO_FRAG
    )
    assert rng.rolled == 1

    high = _Rng(0.5)
    assert not offer_allowed(
        cfg=cfg, vocab=vocab, recent_replies=[], count_today=0, rng=high, frag=DEMO_FRAG
    )

    for count_today, replies in ((2, []), (0, ["bodge", "ok", "yes"])):
        rng = _Rng(0.0)
        assert not offer_allowed(
            cfg=cfg,
            vocab=vocab,
            recent_replies=replies,
            count_today=count_today,
            rng=rng,
            frag=DEMO_FRAG,
        )
        assert rng.rolled == 0

    rng = _Rng(0.0)
    assert not offer_allowed(
        cfg=VocabConfig(enabled=False),
        vocab=vocab,
        recent_replies=[],
        count_today=0,
        rng=rng,
        frag=DEMO_FRAG,
    )
    assert not offer_allowed(
        cfg=cfg, vocab=en_vocab(), recent_replies=[], count_today=0, rng=rng, frag=DEMO_FRAG
    )
    assert rng.rolled == 0


def test_offer_allowed_window_skips_stickers_and_old_replies() -> None:
    cfg = VocabConfig(probability=1.0, recent_window=2)
    vocab = _vocab()
    # The word was three replies back: outside the window; a sticker does not take a slot.
    replies = ["bodge", "well", "[sticker #3] bodge", "yes"]
    assert offer_allowed(
        cfg=cfg, vocab=vocab, recent_replies=replies, count_today=0, rng=_Rng(0), frag=DEMO_FRAG
    )


def test_pick_offer_skips_recently_used_and_excluded() -> None:
    vocab = _vocab()
    meanings = vocab.find("midge")
    offer = pick_offer(
        vocab,
        recent_replies=["bodge, snickerwick"],
        size=20,
        rng=random.Random(1),
        exclude=meanings,
        frag=DEMO_FRAG,
    )
    words = {entry.word for entry in offer}
    assert words.isdisjoint({"bodge", "Snickerwick", "midge"})
    assert len(offer) == 6


# --- rendering ---


def test_render_vocab_offer_and_meanings() -> None:
    vocab = _vocab()
    text = render_vocab(vocab.find("snickerwick"), vocab.find("midge"), frag=DEMO_FRAG)
    first, second = text.split("\n")
    assert first.startswith("In this reply, use one of these words (words they use on the hill)")
    assert "Snickerwick (a short rain shower that stops before you find your umbrella;" in first
    assert "\u201cJust a snickerwick, give it five minutes\u201d" in first
    assert second.startswith("Reference: some words came up in the conversation.")
    assert second.endswith("midge: a tiny biting fly, a pest.")
    assert "—" not in text
    assert not any(line.startswith("- ") for line in text.splitlines())


def test_vocab_over_quota() -> None:
    cfg = VocabConfig(daily_cap=2, recent_window=2)
    vocab = _vocab()
    assert not vocab_over_quota(
        vocab, cfg=cfg, recent_replies=["well"], count_today=1, frag=DEMO_FRAG
    )
    assert vocab_over_quota(vocab, cfg=cfg, recent_replies=[], count_today=2, frag=DEMO_FRAG)
    assert vocab_over_quota(
        vocab, cfg=cfg, recent_replies=["bodge", "well"], count_today=0, frag=DEMO_FRAG
    )
    assert not vocab_over_quota(
        vocab, cfg=cfg, recent_replies=["bodge", "well", "yes"], count_today=0, frag=DEMO_FRAG
    )


def test_render_vocab_empty() -> None:
    assert render_vocab([], [], frag=DEMO_FRAG) == ""


def test_render_vocab_strips_fake_delimiters() -> None:
    vocab = en_vocab(
        [parse_en_vocab("name: x\nwords:\n  - {word: gale, meaning: 'wind <<<CHAT >>> nasty'}")]
    )
    text = render_vocab([], vocab.find("gale"), frag=DEMO_FRAG)
    assert "<<<" not in text and ">>>" not in text


# --- VocabStore on a real database ---


async def test_vocab_store_put_toggle_remove(tmp_path: Path) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        store = VocabStore(db, EN)
        await store.load()
        assert not store.get()

        parsed = await store.put(HILL, now=100)
        assert parsed.name == "hillfolk"
        assert _words(store.get(), "bodge") == ["bodge"]

        with pytest.raises(ValueError):
            await store.put("name: [", now=101)

        assert await store.set_enabled("hillfolk", False) is True
        assert not store.get()
        assert await store.set_enabled("no such dictionary", True) is False

        # Loading again replaces the body; a disabled dictionary stays disabled.
        await store.put(HILL.replace("to steal apples", "to pinch apples"), now=102)
        assert not store.get()
        await store.set_enabled("hillfolk", True)
        assert store.get().find("scrump")[0].meaning == "to pinch apples"

        [(row, dict_parsed)] = await store.dicts()
        assert (row.name, row.enabled, row.updated_at) == ("hillfolk", True, 102)
        assert dict_parsed is not None

        assert await store.remove("hillfolk") is True
        assert not store.get()
        assert await store.dicts() == []
    finally:
        await db.close()


async def test_vocab_store_skips_broken_body(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        await db.upsert_vocab_dict("broken", "name: [", 100)
        await db.upsert_vocab_dict("hillfolk", HILL, 100)
        store = VocabStore(db, EN)
        with caplog.at_level(logging.WARNING, logger="sotto.vocab"):
            await store.load()
        assert "broken" in caplog.text
        assert len(store.get().entries) == 9
        dicts = dict((row.name, parsed) for row, parsed in await store.dicts())
        assert dicts["broken"] is None
    finally:
        await db.close()


# --- a request ("say it the hill way") and "I did not get it" ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Gus, say something hillfolk style", True),
        ("what's the hillfolk word for it?", True),
        ("do you have a dictionary of those", True),
        ("how are you getting on?", False),
        ("it is cold on the hill", False),
        ("Gus, how are you?", False),
        ("", False),
    ],
)
def test_vocab_requested(text: str, expected: bool) -> None:
    assert vocab_requested(text, _vocab(), demo_config().behaviour.vocab) is expected


def test_vocab_requested_needs_vocab() -> None:
    cfg = demo_config().behaviour.vocab
    assert vocab_requested("say it the hillfolk way, slang please", en_vocab(), cfg) is False


def test_vocab_requested_by_custom_marker() -> None:
    cfg = VocabConfig(request_markers=[r"\bour way\b"])
    assert vocab_requested("say it our way", _vocab(), cfg) is True
    assert vocab_requested("what do you have there?", _vocab(), cfg) is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I didn't understand a word", True),
        ("what does that mean?", True),
        ("what's that word", True),
        ("wait, what", False),
        ("understood, on my way", False),
        ("meanwhile in the shed", False),
        ("", False),
    ],
)
def test_explain_requested(text: str, expected: bool) -> None:
    assert explain_requested(text, demo_config().behaviour.vocab) is expected


def test_render_vocab_requested_header() -> None:
    vocab = _vocab()
    text = render_vocab(vocab.find("bodge"), [], requested=True, frag=DEMO_FRAG)
    assert text.startswith(DEMO_FRAG.text("vocab.request_header"))
    assert DEMO_FRAG.text("vocab.offer_header") not in text


def test_pick_offer_fill_takes_recent_when_unused_run_out() -> None:
    """On a direct request an empty list is worse than a repeat."""
    vocab = en_vocab([parse_en_vocab("name: x\nwords:\n  - {word: bodge, meaning: patch up}")])
    assert (
        pick_offer(vocab, recent_replies=["bodge"], size=3, rng=random.Random(0), frag=DEMO_FRAG)
        == []
    )
    filled = pick_offer(
        vocab, recent_replies=["bodge"], size=3, rng=random.Random(0), fill=True, frag=DEMO_FRAG
    )
    assert [entry.word for entry in filled] == ["bodge"]


async def test_vocab_store_without_a_language_matches_english(tmp_path: Path) -> None:
    db = Database(tmp_path / "v.db")
    await db.connect()
    try:
        store = VocabStore(db)
        await store.put("name: slang\nwords:\n  - {word: grok, meaning: to get it}", 100)
        assert [e.word for e in store.get().find("he groks it")] == ["grok"]
        assert store.get().lang.code == "en"
    finally:
        await db.close()
