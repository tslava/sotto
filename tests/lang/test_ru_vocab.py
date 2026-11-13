"""Russian vocabulary matching (sotto.lang.ru.vocab_rules): inflections, prefixes, reflexive verbs.

Moved from tests/test_vocab.py when the morphology tables went to ``sotto/lang/ru.py``; the
engine-level tests (offer, render, store) stay there.
"""

from __future__ import annotations

import random

import pytest

from sotto.vocab import Vocab, pick_offer
from tests.packs import RU_FRAG, parse_ru_vocab, ru_vocab

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
    vocab = ru_vocab([parse_ru_vocab(MORPH)])
    assert [entry.word for entry in vocab.find(text)] == expected


def test_kolbas_is_not_kolba() -> None:
    """Основа «кост» + «ас» — не окончание (ревью Codex 17.10)."""
    vocab = _vocab()
    assert _words(vocab, "Много костюм купили") == []
    assert _words(vocab, "костой пахнет") == ["коста"]


def test_mya_noun_is_not_berem() -> None:
    """«кремя» склоняется от «кремен-»; «крем» — не оно (живой случай 05.11)."""
    vocab = ru_vocab([parse_ru_vocab("name: x\nwords:\n  - {word: кремя, meaning: кусок}")])
    assert _words(vocab, "Мы мажем крем иногда") == []
    assert _words(vocab, "крем всё подряд") == []
    assert _words(vocab, "принёс кремя дров") == ["кремя"]
    assert _words(vocab, "с кременем дров") == ["кремя"]
    assert _words(vocab, "два кремени") == ["кремя"]


def test_short_mya_nouns_inflect() -> None:
    """«имя», «семя» — короткие разносклоняемые: тоже от основы на «-ен» (ревью Codex 05.11)."""
    raw = "name: x\nwords:\n  - {word: имя, meaning: m}\n  - {word: семя, meaning: m}"
    vocab = ru_vocab([parse_ru_vocab(raw)])
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
    vocab = ru_vocab([parse_ru_vocab("name: x\nwords:\n" + "\n".join(words))])
    assert _words(vocab, "он зюзит в этом") == ["зюзить"]
    assert _words(vocab, "зюзился там вчера") == ["зюзиться"]
    assert _words(vocab, "зюзились по рынку") == ["зюзиться"]
    assert _words(vocab, "зюзил в моторах") == ["зюзить"]
    offer = pick_offer(
        vocab, recent_replies=["Зюзился там вчера"], size=5, rng=random.Random(0), frag=RU_FRAG
    )
    assert [entry.word for entry in offer] == ["зюзить"]
