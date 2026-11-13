"""The Russian language object: plural, folding, stop words, vocabulary hooks, default lists."""

from __future__ import annotations

import re

import pytest

from sotto.lang.ru import RussianLanguage
from tests.packs import RU


@pytest.mark.parametrize(
    ("n", "category"),
    [
        (0, "many"),
        (1, "one"),
        (2, "few"),
        (4, "few"),
        (5, "many"),
        (11, "many"),
        (12, "many"),
        (21, "one"),
        (22, "few"),
        (25, "many"),
        (101, "one"),
        (111, "many"),
        (-1, "one"),
    ],
)
def test_plural(n: int, category: str) -> None:
    assert RU.plural(n) == category


def test_fold_lowers_strips_stress_and_yo() -> None:
    assert RU.fold("Сырю́ка  ЁЖ") == "сырюка  еж"
    assert RU.fold("ЁЛКА") == "елка"
    assert RU.fold("MiXed") == "mixed"


def test_language_attributes() -> None:
    assert RU.code == "ru"
    assert RU.foreign_letters == "A-Za-z"
    assert RU.reflexive_suffixes == ("ся", "сь")
    assert "и" in RU.stopwords and "не" in RU.stopwords
    assert isinstance(RU, RussianLanguage)


def test_fact_re_allows_cyrillic_spaces_comma_hyphen_only() -> None:
    fact = RU.fact_re()
    assert fact.fullmatch("тихо, по-домашнему")
    assert fact.fullmatch("Ёж")
    for bad in ["quiet", "тихо 5", "тихо!", "", "тихо\nшумно"]:
        assert not fact.fullmatch(bad)


def test_default_filters_lists_compile_and_are_copies() -> None:
    lists = RU.default_filters()
    assert set(lists) == {
        "injection_markers",
        "weather_request",
        "model_talk",
        "assistant_markers",
        "self_markers",
    }
    for key, patterns in lists.items():
        assert patterns, key
        for pattern in patterns:
            re.compile(pattern, re.IGNORECASE)
    lists["model_talk"].append("changed")
    assert "changed" not in RU.default_filters()["model_talk"]


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("дворовые", "дворо"),
        ("лесные", "лесны"),
        ("слова", "слова"),
        ("тихо", None),
        ("", None),
    ],
)
def test_vocab_name_stem(word: str, stem: str | None) -> None:
    assert RU.vocab_name_stem(word) == stem


def test_vocab_prefix_tails_need_a_long_tail_and_keep_priority_order() -> None:
    assert RU.vocab_prefix_tails("помарганил") == ["марганил"]
    assert RU.vocab_prefix_tails("перепоправил") == ["поправил"]
    assert RU.vocab_prefix_tails("запоздало") == ["поздало"]
    assert RU.vocab_prefix_tails("пода") == []
    assert RU.vocab_prefix_tails("слово") == []


def test_vocab_rules_for_a_plain_verb_and_a_reflexive_one() -> None:
    plain = RU.vocab_rules("гурхать")
    assert plain[0].stem == "гурха" and plain[0].reflexive is False
    reflexive = RU.vocab_rules("бураться")
    assert reflexive[0].stem == "бура" and reflexive[0].reflexive is True


def test_vocab_rules_fleeting_vowel_genitive_goes_to_exact_forms() -> None:
    rules = RU.vocab_rules("ягушка")
    assert rules[0].exact_forms == ("ягушек", "ягушок")
    assert RU.vocab_rules("мошка")[0].exact_forms == ("мошек", "мошок")
    # a short stem or a verb gets none
    assert RU.vocab_rules("тка")[0].exact_forms == ()
    assert RU.vocab_rules("гурхать")[0].exact_forms == ()


def test_vocab_rules_adjective_and_mya_noun() -> None:
    adjective = RU.vocab_rules("дворовый")
    assert [r.stem for r in adjective] == ["дворов"] and adjective[0].reflexive is None
    assert [r.stem for r in RU.vocab_rules("кремя")] == ["кремен"]
    # «-ой» is also a noun: both readings
    assert [r.stem for r in RU.vocab_rules("орехобой")] == ["орехоб", "орехобо"]


def test_vocab_rules_verb_present_tense_variants() -> None:
    assert [r.stem for r in RU.vocab_rules("копировать")] == ["копирова", "копиру"]
    assert [r.stem for r in RU.vocab_rules("гулять")][1] == "гул"
    assert [r.stem for r in RU.vocab_rules("фырнуть")][1] == "фырн"
    assert next(r.stem for r in RU.vocab_rules("гурхати")) == "гурха"


def test_name_declension_through_the_language_object() -> None:
    assert RU.name_forms("отец") == ["отец", "отца", "отцу", "отцом", "отце"]
    assert re.search(RU.name_pattern("федя"), "про Федю", re.IGNORECASE)
    assert not re.search(RU.name_pattern("федя"), "Федяев", re.IGNORECASE)
    # letters that do not decline, and an empty word, stay literal
    assert RU.name_forms("федь") == ["федь"]
    assert RU.name_pattern("") == r"\b\b"
