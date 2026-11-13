"""Russian: name declension, vocabulary inflection, stop words and the generic marker lists.

Everything here moved verbatim from ``morph.py``, ``vocab.py`` (tables and ``_rules``),
``filters.py`` (stop words) and ``config_models.py`` (default marker lists); behaviour is
unchanged. This module and ``tests/lang`` are the only places in the engine that may
contain Cyrillic (``tools/check_language.py``).

Name declension (CLAUDE.md, "имя в падежах и проверка обращения по имени"): a trigger
word is searched in all singular cases, but with a CLOSED list of endings, so that
«федерация», «дедлайн», «Федяев», «отечество», «батюшкам» never match. The letter «ё» is
not normalised here: the caller keeps both «фёдор» and «федор» in ``name_triggers``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from sotto.lang.base import VocabRule

# --- name declension (moved from morph.py) ---------------------------------

# Согласные кириллицы: «ь» и «ъ» сюда не входят намеренно — слово на мягкий знак
# («федь») склоняется не по этому образцу, для него остаётся точное совпадение.
_CONSONANTS = frozenset("бвгджзйклмнпрстфхцчшщ")

# Окончания по образцам склонения. Пустая строка в _CONSONANT_ENDINGS — сама
# именительная форма («федор»), она же делает группу в регулярке необязательной.
_EC_ENDINGS = ("а", "у", "ом", "е")  # отец -> отца, отцу, отцом, отце
_KA_ENDINGS = ("а", "и", "е", "у", "ой")  # батюшка -> батюшки, батюшке, батюшку, батюшкой
_YA_ENDINGS = ("я", "и", "е", "ю", "ей")  # федя -> феди, феде, федю, федей
_A_ENDINGS = ("а", "ы", "е", "у", "ой")
_CONSONANT_ENDINGS = ("", "а", "у", "ом", "е", "ы")  # федор -> федора, федору, федором, ...


@dataclass(frozen=True, slots=True)
class _Declension:
    """Разбор слова на стем и закрытый список окончаний.

    ``keep_word`` — нужно ли считать само слово отдельной формой: у беглой
    гласной («отец» -> «отц-») именительный падеж стемом не покрывается.
    """

    stem: str
    endings: tuple[str, ...]
    keep_word: bool


def _analyze(word: str) -> _Declension | None:
    """Образец склонения для слова или None, если склонять его нечем (мягкий знак,
    «отче», латиница, что угодно ещё) — тогда остаётся точное совпадение."""
    if not word:
        return None
    if len(word) > 2 and word.endswith("ец"):
        return _Declension(stem=word[:-2] + "ц", endings=_EC_ENDINGS, keep_word=True)
    if len(word) > 2 and word.endswith("ка"):
        return _Declension(stem=word[:-1], endings=_KA_ENDINGS, keep_word=False)
    if len(word) > 1 and word.endswith("я"):
        return _Declension(stem=word[:-1], endings=_YA_ENDINGS, keep_word=False)
    if len(word) > 1 and word.endswith("а"):
        return _Declension(stem=word[:-1], endings=_A_ENDINGS, keep_word=False)
    if word[-1] in _CONSONANTS:
        return _Declension(stem=word, endings=_CONSONANT_ENDINGS, keep_word=False)
    return None


def _group(endings: tuple[str, ...]) -> str:
    return "|".join(re.escape(ending) for ending in endings if ending)


def name_pattern(word: str) -> str:
    """Регулярка (строкой, без ``re.compile``) для слова-триггера во всех падежах
    единственного числа. Регистр не учитывается вызывающим через ``re.IGNORECASE``.

    Все части экранируются ``re.escape``: триггеры задаются в ``config.yaml`` и
    попасть туда может что угодно, а падать на кривой регулярке гейт не должен.
    """
    analysis = _analyze(word)
    if analysis is None:
        return rf"\b{re.escape(word)}\b"
    stem = re.escape(analysis.stem)
    group = _group(analysis.endings)
    if analysis.keep_word:
        return rf"\b({re.escape(word)}|{stem}({group}))\b"
    if "" in analysis.endings:
        return rf"\b{stem}({group})?\b"
    return rf"\b{stem}({group})\b"


def name_forms(word: str) -> list[str]:
    """Сами формы слова — то же множество, которое матчит ``name_pattern``.

    Нужны тестам (проверять список форм нагляднее, чем строку регулярки) и
    больше нигде: в рантайме ищется всегда регуляркой.
    """
    analysis = _analyze(word)
    if analysis is None:
        return [word]
    forms = [word] if analysis.keep_word else []
    forms += [analysis.stem + ending for ending in analysis.endings]
    seen: set[str] = set()
    unique: list[str] = []
    for form in forms:
        if form not in seen:
            seen.add(form)
            unique.append(form)
    return unique


# --- vocabulary inflection (moved from vocab.py) ----------------------------

# Формы слова ищутся по основе и ЗАКРЫТОМУ списку русских окончаний (ревью Codex
# 17.10: «любой хвост до N букв» путал «костюм» с «коста»). Основа короче трёх букв —
# только точное совпадение.
_REFLEXIVE = ("ся", "сь")
_VERB_ENDINGS = ("ть", "ти", "чь")
_ADJ_ENDINGS_WORD = ("ый", "ий", "ой")
_FLEETING = ("ок", "ек", "ец")  # беглая гласная: кумонок -> кумонка, мошнок -> мошнков
_SOFT_ENDINGS = frozenset("аеиоуыэюяйь")
_NOUN_ENDINGS = frozenset(
    {"", "а", "я", "о", "е", "ы", "и", "у", "ю", "ь", "й", "ой", "ей", "ою", "ею", "ом", "ем"}
    | {"ам", "ям", "ах", "ях", "ами", "ями", "ов", "ев", "ью"}
)
_ADJ_ENDINGS = frozenset(
    {"ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ого", "его", "ому", "ему", "ую"}
    | {"юю", "ых", "их", "ым", "им", "ыми", "ими", "ом", "ем", "", "а", "о", "ы", "и"}
)
_NOMINAL_ENDINGS = _NOUN_ENDINGS | _ADJ_ENDINGS
# Разносклоняемые на «-мя» (кремя): косвенные падежи от основы на «-ен».
_MYA_ENDINGS = frozenset({"и", "ем", "а", "ам", "ами", "ах"})
# Окончания на основе инфинитива: гурха|л, бура|ет(ся), маргани|ла, скруди|.
_VERB_FORMS = frozenset(
    {"", "ть", "ти", "чь", "л", "ла", "ло", "ли", "в", "вши", "я", "й", "йте"}
    | {"ю", "у", "ешь", "ет", "ем", "ете", "ют", "ут"}
)
# Настоящее время, где основа инфинитива меняется: марган|ит, друж|у|ешь, ша|ют.
_PRESENT_E = frozenset({"ю", "у", "ешь", "ет", "ем", "ете", "ют", "ут", "й", "йте", "я"})
_PRESENT_I = frozenset({"ю", "у", "ишь", "ит", "им", "ите", "ят", "ат", "ь", "я"})
# Глагольные приставки: «по|марганил», «на|трунил», «от|кружили». Отрезаются только
# перед глагольной основой и только если после них остаётся длинное слово.
_VERB_PREFIXES = ("пере", "при", "раз", "рас", "под", "над", "об", "от", "вы", "за", "на", "по")
_PREFIXED_MIN = 5

# Имя словаря как признак просьбы: «дворовые» -> «дворо» ловит «по-дворовому»,
# «дворовое», но не «двора».
_NAME_STEM_MIN = 5
_NAME_STEM_CUT = 3


def _rules(word: str) -> list[tuple[str, frozenset[str], bool | None]]:
    """Основы и допустимые окончания для слова (CLAUDE.md, "словечки из словаря")."""
    base, reflexive = word, False
    for suffix in _REFLEXIVE:
        if base.endswith(suffix) and base[: -len(suffix)].endswith(_VERB_ENDINGS):
            base, reflexive = base[: -len(suffix)], True
            break
    for ending in _VERB_ENDINGS:
        if base.endswith(ending) and len(base) > len(ending) + 2:
            stem = base[: -len(ending)]
            rules: list[tuple[str, frozenset[str], bool | None]] = [(stem, _VERB_FORMS, reflexive)]
            if stem.endswith(("ова", "ева")):
                rules.append(
                    (stem[:-3] + ("у" if stem.endswith("ова") else "ю"), _PRESENT_E, reflexive)
                )
            elif stem.endswith(("и", "е")):
                rules.append((stem[:-1], _PRESENT_I, reflexive))
            elif stem.endswith("я"):
                rules.append((stem[:-1], _PRESENT_E, reflexive))
            elif stem.endswith("ну"):  # фырну|ть -> фырн|и, фырн|ешь
                rules.append((stem[:-1], _PRESENT_E | {"и", "ите"}, reflexive))
            return rules
    rules = []
    if base.endswith(_ADJ_ENDINGS_WORD) and len(base) >= 5:
        rules.append((base[:-2], _NOMINAL_ENDINGS, None))
        if not base.endswith("ой"):  # «орехобой» — существительное: орехобо|ев
            return rules
    if base.endswith("мя") and len(base) >= 3:
        # Кремя, бремя, имя, семя: кремени, кременем, об имени. Общая основа «крем»
        # с пустым окончанием путала его с «крем» (05.11, живой случай).
        rules.append((base[:-1] + "ен", _MYA_ENDINGS, None))
        return rules
    stem = base[:-1] if base[-1] in _SOFT_ENDINGS else base
    rules.append((stem, _NOMINAL_ENDINGS, None))
    if base.endswith(_FLEETING) and len(base) > 4:
        rules.append((base[:-2] + base[-1], _NOMINAL_ENDINGS - {""}, None))
    return rules


# --- stop words (moved from filters.py) ------------------------------------

# Стоп-слова для dedup:self_echo (CLAUDE.md, "Интерфейсы этапа 4", правки): 4-грамма,
# состоящая только из этих слов, слишком общая, чтобы считаться самоповтором.
_SELF_ECHO_STOPWORDS = frozenset(
    {
        "я",
        "и",
        "в",
        "на",
        "не",
        "что",
        "это",
        "у",
        "меня",
        "тебя",
        "ты",
        "а",
        "но",
        "да",
        "нет",
        "же",
        "бы",
        "как",
        "так",
        "то",
        "все",
        "всё",
    }
)

# --- generic marker lists (moved from config_models.py) --------------------

_INJECTION_MARKERS_DEFAULT = [
    r"забудь (инструкции|правила|всё)",
    r"\bигнорируй (все|всё|предыдущ\w*|инструкци\w*|правил\w*)\b",
    r"\bты теперь (бот|пират|ассистент|помощник|нейросеть|другой человек"
    r"|другой персонаж|играешь роль)\b",
    r"системн\w+ промпт",
    r"повтори за мной",
    r"скажи дословно",
    r"напиши слово",
    r"ignore (previous|all)",
    r"как языковая модель",
    r"\bпредставь,? что ты\b",
    r"\bпритворись\b",
    r"\bиграй роль\b",
]

# Повод посмотреть погоду в месте, про которое спросили (CLAUDE.md, "погода в
# другом месте"). Регулярка только включает дешёвый вызов, который достаёт из
# текста название места, — само решение «спрашивали ли про погоду» за моделью.
_WEATHER_REQUEST_DEFAULT = [
    r"\bпогод",
    r"\bдожд",
    r"\bснег",
    r"\bжар[аеуы]\b",
    r"\bupał",
    r"\bтемператур",
    r"\bградус",
    r"\bтепл[оаы]",
    r"\bхолодн",
]

_MODEL_TALK_DEFAULT = [
    r"языковая модель",
    r"нейросет\w*",
    r"инструкци\w*",
    r"промпт\w*",
    r"\bИИ\b",
    r"OpenAI",
    r"Anthropic",
    r"OpenRouter",
]

_ASSISTANT_MARKERS_DEFAULT = [
    "важно отметить",
    "стоит учесть",
    "рекомендую",
    "могу помочь",
    "дай знать",
    "надеюсь это поможет",
    "во-первых",
    "конечно!",
    "отличный вопрос",
    "если у тебя есть вопросы",
]

# Маркеры «говорит о себе» (CLAUDE.md, "про себя через раз") — только для метрики в
# /status, ничего не режут. Местоимения первого лица и его родня; «мне кажется» не
# считается. Формы перечислены закрытым списком окончаний («Внуково» и «нашёл» мимо).
# Ложные срабатывания («у нас» про весь чат) допустимы: метрика грубая и
# нужна, чтобы видеть тренд, а не судить отдельную реплику.
_SELF_MARKERS_DEFAULT = [
    r"\bя\b",
    r"\bмен[яе]\b",
    r"\bмне\b(?!\s+кажется)",
    r"\bмной\b",
    r"\bмо(й|я|ё|е|и|ю|его|ей|их|им|ими|ему|ём|ем)\b",
    r"\bмы\b",
    r"\bу нас\b",
    r"\bнаш(а|е|и|у|его|ей|их|им|ими|ему|ем|ём)?\b",
    r"\bвнук(а|у|ом|е|и|ов|ам|ами|ах)?\b",
    r"\bвнучк(а|и|е|у|ой)\b",
    r"\bжен[аеуы]\b",
    r"\bжено[йю]\b",
    r"\bсын(а|у|ом|е|ы|ов|овья|овей|овьям|ок|ка|ку|ком|ке)?\b",
]


# --- the language object ------------------------------------------------------

# Знаки ударения (острое и тупое) — в публичных глоссариях они в каждом слове.
_ACCENTS_RE = re.compile("[\u0300\u0301]")
_FACT_RE = re.compile(r"^[а-яёА-ЯЁ ,\-]+$")
_FLEETING_GENITIVE_MIN = 3


class RussianLanguage:
    code = "ru"
    stopwords = _SELF_ECHO_STOPWORDS
    foreign_letters: str | None = "A-Za-z"
    reflexive_suffixes = _REFLEXIVE

    def plural(self, n: int) -> str:
        count = abs(n)
        if count % 10 == 1 and count % 100 != 11:
            return "one"
        if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
            return "few"
        return "many"

    def fold(self, text: str) -> str:
        return _ACCENTS_RE.sub("", text).casefold().replace("ё", "е")

    def name_pattern(self, word: str) -> str:
        return name_pattern(word)

    def name_forms(self, word: str) -> list[str]:
        return name_forms(word)

    def vocab_rules(self, word: str) -> list[VocabRule]:
        rules: list[VocabRule] = []
        for stem, endings, reflexive in _rules(word):
            exact: tuple[str, ...] = ()
            # Родительный множественного с беглой гласной: ягушк|а -> ягушек.
            if reflexive is None and len(stem) > _FLEETING_GENITIVE_MIN and stem.endswith("к"):
                exact = tuple(f"{stem[:-1]}{vowel}к" for vowel in ("е", "о"))
            rules.append(VocabRule(stem, endings, reflexive, exact))
        return rules

    def vocab_prefix_tails(self, token: str) -> list[str]:
        return [
            token[len(prefix) :]
            for prefix in _VERB_PREFIXES
            if token.startswith(prefix) and len(token) - len(prefix) >= _PREFIXED_MIN
        ]

    def vocab_name_stem(self, word: str) -> str | None:
        if len(word) < _NAME_STEM_MIN:
            return None
        return word[: max(_NAME_STEM_MIN, len(word) - _NAME_STEM_CUT)]

    def fact_re(self) -> re.Pattern[str]:
        return _FACT_RE

    def default_filters(self) -> Mapping[str, list[str]]:
        return {
            "injection_markers": list(_INJECTION_MARKERS_DEFAULT),
            "weather_request": list(_WEATHER_REQUEST_DEFAULT),
            "model_talk": list(_MODEL_TALK_DEFAULT),
            "assistant_markers": list(_ASSISTANT_MARKERS_DEFAULT),
            "self_markers": list(_SELF_MARKERS_DEFAULT),
        }


# --- CHANGELOG.md block headers of a pre-split Russian pack (read by changelog.py) ---------------

LEGACY_FOR_CHAT_HEADER = "Для чата"
LEGACY_FOR_OWNER_HEADER = "Для владельца"
