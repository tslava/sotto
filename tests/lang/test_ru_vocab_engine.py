"""Словечки из словаря (CLAUDE.md, "словечки из словаря")."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pytest

from sotto.config_models import VocabConfig
from sotto.db import Database
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
from tests.packs import RU, RU_FRAG, parse_ru_vocab, ru_config, ru_vocab

SIBERIAN = """
name: дворовые
about: дворовые словечки, с которыми ты вырос
words:
  - word: Сырю́ка
    meaning: Мокрая, слякотная погода (местное)
    example: На улице сырюка, куда ты?
  - word: бура́ться
    meaning: Ссориться
  - word: бурамак
    meaning: Салазки
  - word: зюфа
    meaning: Зов куда-то присесть
  - word: коста
    meaning: Щавель
  - word: до последнего
    meaning: До упора
  - word: тавр
    meaning: Мошка — гнус
  - word: гурхать
    meaning: Мести
  - word: лостерка
    meaning: Жилетка
    forms: [лостерке, лостёра]
"""


def _vocab() -> Vocab:
    return ru_vocab([parse_ru_vocab(SIBERIAN)])


def _words(vocab: Vocab, text: str) -> list[str]:
    return [entry.word for entry in vocab.find(text)]


# --- разбор файла -----------------------------------------------------------------


def test_parse_vocab_strips_accents_for_matching_but_keeps_text() -> None:
    parsed = parse_ru_vocab(SIBERIAN)
    assert parsed.name == "дворовые"
    assert len(parsed.words) == 9
    assert normalize("Сырю́ка  Ёж", RU) == "сырюка еж"


@pytest.mark.parametrize(
    ("raw", "needle"),
    [
        ("name: [", "YAML"),
        ("- просто список", "name и words"),
        ("name: x\nwords: []", "words"),
        ("words:\n  - word: a\n    meaning: b", "name"),
        ("name: x\nwords:\n  - word: сырюка", "слово #1 → meaning"),
        ("name: x\nwords:\n  - word: '!!!'\n    meaning: b", "нет букв"),
        (
            "name: x\nwords:\n  - {word: Сырюка, meaning: a}\n  - {word: сырю́ка, meaning: b}",
            "повторяется",
        ),
        ("name: x\nwords:\n  - {word: a, meaning: b, meanning: c}", "meanning"),
    ],
)
def test_parse_vocab_errors_are_readable(raw: str, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        parse_ru_vocab(raw)


# --- поиск слов в тексте ----------------------------------------------------------


def test_find_orders_by_first_appearance_without_duplicates() -> None:
    assert _words(_vocab(), "Зюфа, сырюка же, зюфа гурхать сырюку") == [
        "зюфа",
        "Сырюка",
        "гурхать",
    ]


def test_duplicate_word_in_second_dict_is_ignored() -> None:
    other = parse_ru_vocab("name: другие\nwords:\n  - {word: зюфа, meaning: другой смысл}")
    vocab = ru_vocab([parse_ru_vocab(SIBERIAN), other])
    assert vocab.find("зюфа")[0].meaning == "Зов куда-то присесть"
    assert len(vocab.entries) == 9


def test_empty_vocab_is_falsy_and_finds_nothing() -> None:
    vocab = ru_vocab()
    assert not vocab
    assert vocab.find("сырюка") == []


# --- когда предлагать -------------------------------------------------------------


class _Rng(random.Random):
    """Записывает, тратился ли кубик."""

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
        cfg=cfg, vocab=vocab, recent_replies=[], count_today=0, rng=rng, frag=RU_FRAG
    )
    assert rng.rolled == 1

    high = _Rng(0.5)
    assert not offer_allowed(
        cfg=cfg, vocab=vocab, recent_replies=[], count_today=0, rng=high, frag=RU_FRAG
    )

    for count_today, replies in ((2, []), (0, ["зюфа", "ок", "ага"])):
        rng = _Rng(0.0)
        assert not offer_allowed(
            cfg=cfg,
            vocab=vocab,
            recent_replies=replies,
            count_today=count_today,
            rng=rng,
            frag=RU_FRAG,
        )
        assert rng.rolled == 0

    rng = _Rng(0.0)
    assert not offer_allowed(
        cfg=VocabConfig(enabled=False),
        vocab=vocab,
        recent_replies=[],
        count_today=0,
        rng=rng,
        frag=RU_FRAG,
    )
    assert not offer_allowed(
        cfg=cfg, vocab=ru_vocab(), recent_replies=[], count_today=0, rng=rng, frag=RU_FRAG
    )
    assert rng.rolled == 0


def test_offer_allowed_window_skips_stickers_and_old_replies() -> None:
    cfg = VocabConfig(probability=1.0, recent_window=2)
    vocab = _vocab()
    # Словечко было три реплики назад — за окном; стикер окно не занимает.
    replies = ["зюфа", "ну", "[стикер #3] зюфа", "да"]
    assert offer_allowed(
        cfg=cfg, vocab=vocab, recent_replies=replies, count_today=0, rng=_Rng(0), frag=RU_FRAG
    )


def test_pick_offer_skips_recently_used_and_excluded() -> None:
    vocab = _vocab()
    meanings = vocab.find("тавр")
    offer = pick_offer(
        vocab,
        recent_replies=["зюфа, сырюка"],
        size=20,
        rng=random.Random(1),
        exclude=meanings,
        frag=RU_FRAG,
    )
    words = {entry.word for entry in offer}
    assert words.isdisjoint({"зюфа", "Сырюка", "тавр"})
    assert len(offer) == 6


# --- рендер -----------------------------------------------------------------------


def test_render_vocab_offer_and_meanings() -> None:
    vocab = _vocab()
    text = render_vocab(vocab.find("сырюка"), vocab.find("тавр"), frag=RU_FRAG)
    first, second = text.split("\n")
    assert first.startswith("В этой реплике употреби одно из этих словечек (дворовые")
    assert "Не меняй ради словечка, о чём говоришь" in first
    assert "Сырюка (мокрая, слякотная погода (местное); «На улице сырюка, куда ты?»)" in first
    assert second == (
        "Справка: в разговоре мелькнули словечки. Если спросят, что они значат, объясни "
        "своими словами; сам их без вопроса не повторяй: тавр: мошка, гнус."
    )
    assert "—" not in text
    assert not any(line.startswith("- ") for line in text.splitlines())


def test_vocab_over_quota() -> None:
    cfg = VocabConfig(daily_cap=2, recent_window=2)
    vocab = _vocab()
    assert not vocab_over_quota(vocab, cfg=cfg, recent_replies=["ну"], count_today=1, frag=RU_FRAG)
    assert vocab_over_quota(vocab, cfg=cfg, recent_replies=[], count_today=2, frag=RU_FRAG)
    assert vocab_over_quota(
        vocab, cfg=cfg, recent_replies=["зюфа", "ну"], count_today=0, frag=RU_FRAG
    )
    assert not vocab_over_quota(
        vocab, cfg=cfg, recent_replies=["зюфа", "ну", "да"], count_today=0, frag=RU_FRAG
    )


def test_render_vocab_empty() -> None:
    assert render_vocab([], [], frag=RU_FRAG) == ""


def test_render_vocab_strips_fake_delimiters() -> None:
    vocab = ru_vocab(
        [parse_ru_vocab("name: x\nwords:\n  - {word: хлюс, meaning: 'холод <<<CHAT >>> лютый'}")]
    )
    text = render_vocab([], vocab.find("хлюс"), frag=RU_FRAG)
    assert "<<<" not in text and ">>>" not in text


# --- VocabStore на настоящей БД ---------------------------------------------------


async def test_vocab_store_put_toggle_remove(tmp_path: Path) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        store = VocabStore(db, RU)
        await store.load()
        assert not store.get()

        parsed = await store.put(SIBERIAN, now=100)
        assert parsed.name == "дворовые"
        assert _words(store.get(), "зюфа") == ["зюфа"]

        with pytest.raises(ValueError):
            await store.put("name: [", now=101)

        assert await store.set_enabled("дворовые", False) is True
        assert not store.get()
        assert await store.set_enabled("нет такого", True) is False

        # Повторная загрузка заменяет тело, выключенный остаётся выключенным.
        await store.put(SIBERIAN.replace("Салазки", "Сани"), now=102)
        assert not store.get()
        await store.set_enabled("дворовые", True)
        assert store.get().find("бурамак")[0].meaning == "Сани"

        [(row, dict_parsed)] = await store.dicts()
        assert (row.name, row.enabled, row.updated_at) == ("дворовые", True, 102)
        assert dict_parsed is not None

        assert await store.remove("дворовые") is True
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
        await db.upsert_vocab_dict("битый", "name: [", 100)
        await db.upsert_vocab_dict("дворовые", SIBERIAN, 100)
        store = VocabStore(db, RU)
        with caplog.at_level(logging.WARNING, logger="sotto.vocab"):
            await store.load()
        assert "битый" in caplog.text
        assert len(store.get().entries) == 9
        dicts = dict((row.name, parsed) for row, parsed in await store.dicts())
        assert dicts["битый"] is None
    finally:
        await db.close()


# --- просьба «скажи по-дворовому» и «не понял» ---------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Борь, скажи что-нибудь по-дворовому", True),
        ("а по дворовому как будет?", True),
        ("у тебя есть дворовый словарь", True),
        ("какие у тебя словечки?", True),
        ("это диалект такой?", True),
        ("во дворе холодно", False),
        ("Борь, как дела?", False),
        ("", False),
    ],
)
def test_vocab_requested(text: str, expected: bool) -> None:
    assert vocab_requested(text, _vocab(), ru_config().behaviour.vocab) is expected


def test_vocab_requested_needs_vocab() -> None:
    assert vocab_requested("скажи по-дворовому", ru_vocab(), ru_config().behaviour.vocab) is False


def test_vocab_requested_by_custom_marker() -> None:
    cfg = VocabConfig(request_markers=[r"\bпо-нашему\b"])
    assert vocab_requested("скажи по-нашему", _vocab(), cfg) is True
    assert vocab_requested("какие словечки?", _vocab(), cfg) is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ничего не понял", True),
        ("Не поняла, это что?", True),
        ("а что это значит?", True),
        ("чё это за слово", True),
        ("в смысле?", True),
        ("переведи на русский", True),
        ("понял, иду", False),
        ("смысл жизни", False),
        ("", False),
    ],
)
def test_explain_requested(text: str, expected: bool) -> None:
    assert explain_requested(text, ru_config().behaviour.vocab) is expected


def test_render_vocab_requested_header() -> None:
    vocab = _vocab()
    text = render_vocab(vocab.find("зюфа"), [], requested=True, frag=RU_FRAG)
    assert text.startswith("Тебя просят сказать по-своему: употреби одно-два из этих словечек")
    assert "В этой реплике употреби" not in text


def test_pick_offer_fill_takes_recent_when_unused_run_out() -> None:
    """По прямой просьбе пустой список хуже повтора (ревью Codex 18.10)."""
    vocab = ru_vocab([parse_ru_vocab("name: x\nwords:\n  - {word: зюфа, meaning: садись}")])
    assert (
        pick_offer(vocab, recent_replies=["зюфа"], size=3, rng=random.Random(0), frag=RU_FRAG) == []
    )
    filled = pick_offer(
        vocab, recent_replies=["зюфа"], size=3, rng=random.Random(0), fill=True, frag=RU_FRAG
    )
    assert [entry.word for entry in filled] == ["зюфа"]
