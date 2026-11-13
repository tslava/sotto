"""English: deliberately small.

An English pack is the demo and the quickstart, not a product: names get an optional
possessive, vocabulary words an optional plain ending, and the generic marker lists are
short. Limits are documented in ``docs/creating-a-persona.md``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from sotto.lang.base import VocabRule

_STOPWORDS = frozenset(
    {
        "i",
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "to",
        "of",
        "in",
        "on",
        "at",
        "is",
        "it",
        "that",
        "this",
        "my",
        "me",
        "you",
        "your",
        "not",
        "no",
        "yes",
        "so",
        "as",
        "do",
        "be",
    }
)

# Plain endings a vocabulary word may carry: plural, past, progressive.
_WORD_ENDINGS = frozenset({"", "s", "es", "ed", "ing", "d"})
_NAME_STEM_MIN = 4
_NAME_STEM_CUT = 2
_FACT_RE = re.compile(r"^[A-Za-z ,\-]+$")

_INJECTION_MARKERS = [
    r"\bforget (your |all |the )?(instructions|rules|everything)\b",
    r"\bignore (all|previous|prior|your)\b",
    r"\byou are now\b",
    r"\bsystem prompt\b",
    r"\brepeat after me\b",
    r"\bsay exactly\b",
    r"\bpretend (that )?you\b",
    r"\bact as\b",
]
_WEATHER_REQUEST = [
    r"\bweather\b",
    r"\brain",
    r"\bsnow",
    r"\btemperature\b",
    r"\bdegrees\b",
    r"\bforecast\b",
]
_MODEL_TALK = [
    r"language model",
    r"neural net\w*",
    r"\binstructions?\b",
    r"\bprompts?\b",
    r"\bAI\b",
    r"OpenAI",
    r"Anthropic",
    r"OpenRouter",
]
_ASSISTANT_MARKERS = [
    "it is important to note",
    "it's worth noting",
    "i recommend",
    "i can help",
    "let me know",
    "hope this helps",
    "great question",
    "feel free to ask",
]
_SELF_MARKERS = [
    r"\bI\b",
    r"\bI'(m|ve|ll|d)\b",
    r"\bme\b",
    r"\bmy\b",
    r"\bmine\b",
    r"\bwe\b",
    r"\bour\b",
]


class EnglishLanguage:
    code = "en"
    stopwords = _STOPWORDS
    # No foreign script to guard against: ``regex:venue`` and ``regex:latin`` (and the latin-token
    # frequency rule) are off for English. A capitalised-word venue check would fire on every
    # sentence start, weekday and month without a stop list, so it is left out on purpose; the
    # invented-venue guard for English personas is ``places`` in the prompt plus the judge.
    foreign_letters: str | None = None
    reflexive_suffixes: tuple[str, ...] = ()

    def plural(self, n: int) -> str:
        return "one" if abs(n) == 1 else "other"

    def fold(self, text: str) -> str:
        return text.replace("́", "").casefold()

    def name_pattern(self, word: str) -> str:
        return rf"\b{re.escape(word)}(?:['’]s)?\b"

    def name_forms(self, word: str) -> list[str]:
        return [word, f"{word}'s"]

    def vocab_rules(self, word: str) -> list[VocabRule]:
        return [VocabRule(word, _WORD_ENDINGS, None)]

    def vocab_prefix_tails(self, token: str) -> list[str]:
        return []

    def vocab_name_stem(self, word: str) -> str | None:
        if len(word) < _NAME_STEM_MIN:
            return None
        return word[: max(_NAME_STEM_MIN, len(word) - _NAME_STEM_CUT)]

    def fact_re(self) -> re.Pattern[str]:
        return _FACT_RE

    def default_filters(self) -> Mapping[str, list[str]]:
        return {
            "injection_markers": list(_INJECTION_MARKERS),
            "weather_request": list(_WEATHER_REQUEST),
            "model_talk": list(_MODEL_TALK),
            "assistant_markers": list(_ASSISTANT_MARKERS),
            "self_markers": list(_SELF_MARKERS),
        }
