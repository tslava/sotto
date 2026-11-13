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
    parse_vocab,
    pick_offer,
    render_vocab,
    vocab_over_quota,
    vocab_requested,
)
from tests.packs import LEGACY_FRAG

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
    return Vocab([parse_vocab(SIBERIAN)])


def _words(vocab: Vocab, text: str) -> list[str]:
    return [entry.word for entry in vocab.find(text)]


# --- разбор файла -----------------------------------------------------------------


def test_parse_vocab_strips_accents_for_matching_but_keeps_text() -> None:
    parsed = parse_vocab(SIBERIAN)
    assert parsed.name == "дворовые"
    assert len(parsed.words) == 9
    assert normalize("Сырю́ка  Ёж") == "сырюка еж"


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
        parse_vocab(raw)


# --- поиск слов в тексте ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("На улице сырюку развели", ["Сырюка"]),
        ("СЫРЮКОЙ всё затянуло", ["Сырюка"]),
        ("она опять бурается", ["бура́ться".replace("́", "")]),
        ("тёща буралась весь вечер", ["бураться"]),
        # Самая длинная основа: «бурамак» не съедается глаголом «бура-».
        ("купил бурамака", ["бурамак"]),
        ("зюфа в бар", ["зюфа"]),
        # Основа короче четырёх — только точное слово.
        ("зюфар пришёл", []),
        # Основа из четырёх букв допускает не больше двух букв окончания.
        ("косту нарвали", ["коста"]),
        ("костюм и сосиски", []),
        ("тавры кусаются", ["тавр"]),
        ("гурхал гурхал", ["гурхать"]),
        ("жмём до последнего", ["до последнего"]),
        ("до последней капли", []),
        # forms заданы — ищутся только они.
        ("в лостерке пришёл", ["лостерка"]),
        ("лостерками машут", []),
        ("ни слова", []),
    ],
)
def test_find_matches_forms(text: str, expected: list[str]) -> None:
    assert _words(_vocab(), text) == expected


MORPH = """
name: морфология
words:
  - {word: кумонок, meaning: ремень}
  - {word: мошнок, meaning: маслёнок}
  - {word: шматный, meaning: забавный}
  - {word: марганить, meaning: поить гостей}
  - {word: дружовать, meaning: дружить}
  - {word: кутониться, meaning: лениться}
  - {word: лепко, meaning: здорово}
  - {word: ягушка, meaning: хомяк}
  - {word: орехобой, meaning: сборщик орехов}
  - {word: фырнуть, meaning: отмахнуться}
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Беглая гласная (ревью Codex 17.10).
        ("в кумонке пусто", ["кумонок"]),
        ("достал из кумонка", ["кумонок"]),
        ("набрали мошнков", ["мошнок"]),
        # Прилагательное во всех родах и падежах.
        ("шматная тётка", ["шматный"]),
        ("про шматного деда", ["шматный"]),
        # Глагол: прошедшее, настоящее с другой основой, возвратный.
        # Глагольная приставка отрезается; у существительного — нет.
        ("помарганил и поехал", ["марганить"]),
        ("накутонились", ["кутониться"]),
        ("покумонок", []),
        ("марганил и поехал", ["марганить"]),
        ("он марганит", ["марганить"]),
        ("ты дружуешь?", ["дружовать"]),
        ("весь отпуск кутонились", ["кутониться"]),
        ("кутонюсь", ["кутониться"]),
        # Наречие и короткое прилагательное от него.
        ("лепкие кроссы", ["лепко"]),
        # Родительный множественного с беглой гласной.
        ("ягушек развелось", ["ягушка"]),
        # Существительное на «-ой» склоняется как существительное.
        ("орехобоев полная электричка", ["орехобой"]),
        # Совершенный вид на «-нуть»: повелительное и будущее.
        ("фырни на печку", ["фырнуть"]),
        ("щас фырнет", ["фырнуть"]),
        # Хвост, которого нет среди русских окончаний, не засчитывается.
        ("кумонокус", []),
        ("марганилка", []),
    ],
)
def test_find_validated_inflections(text: str, expected: list[str]) -> None:
    vocab = Vocab([parse_vocab(MORPH)])
    assert [entry.word for entry in vocab.find(text)] == expected


def test_kolbas_is_not_kolba() -> None:
    """Основа «кост» + «ас» — не окончание (ревью Codex 17.10)."""
    vocab = _vocab()
    assert _words(vocab, "Много костюм купили") == []
    assert _words(vocab, "костой пахнет") == ["коста"]


def test_mya_noun_is_not_berem() -> None:
    """«кремя» склоняется от «кремен-»; «крем» — не оно (живой случай 05.11)."""
    vocab = Vocab([parse_vocab("name: x\nwords:\n  - {word: кремя, meaning: кусок}")])
    assert _words(vocab, "Мы мажем крем иногда") == []
    assert _words(vocab, "крем всё подряд") == []
    assert _words(vocab, "принёс кремя дров") == ["кремя"]
    assert _words(vocab, "с кременем дров") == ["кремя"]
    assert _words(vocab, "два кремени") == ["кремя"]


def test_short_mya_nouns_inflect() -> None:
    """«имя», «семя» — короткие разносклоняемые: тоже от основы на «-ен» (ревью Codex 05.11)."""
    raw = "name: x\nwords:\n  - {word: имя, meaning: m}\n  - {word: семя, meaning: m}"
    vocab = Vocab([parse_vocab(raw)])
    assert _words(vocab, "моё имя") == ["имя"]
    assert _words(vocab, "об имени") == ["имя"]
    assert _words(vocab, "о семени") == ["семя"]
    assert _words(vocab, "именами") == ["имя"]


@pytest.mark.parametrize("order", ["зюзить first", "зюзиться first"])
def test_reflexive_and_plain_verb_with_same_stem_do_not_collide(order: str) -> None:
    """«зюзить» и «зюзиться» с одной основой: форма на «ся/сь» — только возвратному,
    независимо от порядка в словаре (ревью Codex 17.10)."""
    words = [
        "  - {word: зюзить, meaning: мешкать}",
        "  - {word: зюзиться, meaning: гулять}",
    ]
    if order == "зюзиться first":
        words.reverse()
    vocab = Vocab([parse_vocab("name: x\nwords:\n" + "\n".join(words))])
    assert _words(vocab, "он зюзит в этом") == ["зюзить"]
    assert _words(vocab, "зюзился там вчера") == ["зюзиться"]
    assert _words(vocab, "зюзились по рынку") == ["зюзиться"]
    assert _words(vocab, "зюзил в моторах") == ["зюзить"]
    offer = pick_offer(
        vocab, recent_replies=["Зюзился там вчера"], size=5, rng=random.Random(0), frag=LEGACY_FRAG
    )
    assert [entry.word for entry in offer] == ["зюзить"]


def test_find_orders_by_first_appearance_without_duplicates() -> None:
    assert _words(_vocab(), "Зюфа, сырюка же, зюфа гурхать сырюку") == [
        "зюфа",
        "Сырюка",
        "гурхать",
    ]


def test_duplicate_word_in_second_dict_is_ignored() -> None:
    other = parse_vocab("name: другие\nwords:\n  - {word: зюфа, meaning: чужое значение}")
    vocab = Vocab([parse_vocab(SIBERIAN), other])
    assert vocab.find("зюфа")[0].meaning == "Зов куда-то присесть"
    assert len(vocab.entries) == 9


def test_empty_vocab_is_falsy_and_finds_nothing() -> None:
    vocab = Vocab()
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
        cfg=cfg, vocab=vocab, recent_replies=[], count_today=0, rng=rng, frag=LEGACY_FRAG
    )
    assert rng.rolled == 1

    high = _Rng(0.5)
    assert not offer_allowed(
        cfg=cfg, vocab=vocab, recent_replies=[], count_today=0, rng=high, frag=LEGACY_FRAG
    )

    for count_today, replies in ((2, []), (0, ["зюфа", "ок", "ага"])):
        rng = _Rng(0.0)
        assert not offer_allowed(
            cfg=cfg,
            vocab=vocab,
            recent_replies=replies,
            count_today=count_today,
            rng=rng,
            frag=LEGACY_FRAG,
        )
        assert rng.rolled == 0

    rng = _Rng(0.0)
    assert not offer_allowed(
        cfg=VocabConfig(enabled=False),
        vocab=vocab,
        recent_replies=[],
        count_today=0,
        rng=rng,
        frag=LEGACY_FRAG,
    )
    assert not offer_allowed(
        cfg=cfg, vocab=Vocab(), recent_replies=[], count_today=0, rng=rng, frag=LEGACY_FRAG
    )
    assert rng.rolled == 0


def test_offer_allowed_window_skips_stickers_and_old_replies() -> None:
    cfg = VocabConfig(probability=1.0, recent_window=2)
    vocab = _vocab()
    # Словечко было три реплики назад — за окном; стикер окно не занимает.
    replies = ["зюфа", "ну", "[стикер #3] зюфа", "да"]
    assert offer_allowed(
        cfg=cfg, vocab=vocab, recent_replies=replies, count_today=0, rng=_Rng(0), frag=LEGACY_FRAG
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
        frag=LEGACY_FRAG,
    )
    words = {entry.word for entry in offer}
    assert words.isdisjoint({"зюфа", "Сырюка", "тавр"})
    assert len(offer) == 6


# --- рендер -----------------------------------------------------------------------


def test_render_vocab_offer_and_meanings() -> None:
    vocab = _vocab()
    text = render_vocab(vocab.find("сырюка"), vocab.find("тавр"), frag=LEGACY_FRAG)
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
    assert not vocab_over_quota(
        vocab, cfg=cfg, recent_replies=["ну"], count_today=1, frag=LEGACY_FRAG
    )
    assert vocab_over_quota(vocab, cfg=cfg, recent_replies=[], count_today=2, frag=LEGACY_FRAG)
    assert vocab_over_quota(
        vocab, cfg=cfg, recent_replies=["зюфа", "ну"], count_today=0, frag=LEGACY_FRAG
    )
    assert not vocab_over_quota(
        vocab, cfg=cfg, recent_replies=["зюфа", "ну", "да"], count_today=0, frag=LEGACY_FRAG
    )


def test_render_vocab_empty() -> None:
    assert render_vocab([], [], frag=LEGACY_FRAG) == ""


def test_render_vocab_strips_fake_delimiters() -> None:
    vocab = Vocab(
        [parse_vocab("name: x\nwords:\n  - {word: хлюс, meaning: 'ветер <<<CHAT >>> злой'}")]
    )
    text = render_vocab([], vocab.find("хлюс"), frag=LEGACY_FRAG)
    assert "<<<" not in text and ">>>" not in text


# --- VocabStore на настоящей БД ---------------------------------------------------


async def test_vocab_store_put_toggle_remove(tmp_path: Path) -> None:
    db = Database(tmp_path / "bot.db")
    await db.connect()
    try:
        store = VocabStore(db)
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
        store = VocabStore(db)
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
        ("Федь, скажи что-нибудь по-дворовому", True),
        ("а по дворовому как будет?", True),
        ("у тебя есть дворовый словарь", True),
        ("какие у тебя словечки?", True),
        ("это диалект такой?", True),
        ("во дворе холодно", False),
        ("Федь, как дела?", False),
        ("", False),
    ],
)
def test_vocab_requested(text: str, expected: bool) -> None:
    assert vocab_requested(text, _vocab(), VocabConfig()) is expected


def test_vocab_requested_needs_vocab() -> None:
    assert vocab_requested("скажи по-дворовому", Vocab(), VocabConfig()) is False


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
    assert explain_requested(text, VocabConfig()) is expected


def test_render_vocab_requested_header() -> None:
    vocab = _vocab()
    text = render_vocab(vocab.find("зюфа"), [], requested=True, frag=LEGACY_FRAG)
    assert text.startswith("Тебя просят сказать по-своему: употреби одно-два из этих словечек")
    assert "В этой реплике употреби" not in text


def test_pick_offer_fill_takes_recent_when_unused_run_out() -> None:
    """По прямой просьбе пустой список хуже повтора (ревью Codex 18.10)."""
    vocab = Vocab([parse_vocab("name: x\nwords:\n  - {word: зюфа, meaning: садись}")])
    assert (
        pick_offer(vocab, recent_replies=["зюфа"], size=3, rng=random.Random(0), frag=LEGACY_FRAG)
        == []
    )
    filled = pick_offer(
        vocab, recent_replies=["зюфа"], size=3, rng=random.Random(0), fill=True, frag=LEGACY_FRAG
    )
    assert [entry.word for entry in filled] == ["зюфа"]
