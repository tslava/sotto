"""Compiled gate and filter patterns (CHARACTER.md section 6, PLAN.md stage 2).

Everything is compiled once in __init__; after that, pure search methods over ready
regex objects. Implements gate_types.PatternsLike.
"""

from __future__ import annotations

import re

from sotto.config_models import FiltersConfig
from sotto.lang import Language, get_language


def _compile_regex_list(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def _compile_phrase_list(phrases: list[str]) -> list[re.Pattern[str]]:
    """Assistant markers: phrases, escaped through re.escape when compiled."""
    return [re.compile(re.escape(phrase), re.IGNORECASE) for phrase in phrases]


def _first_match(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        if pattern.search(text):
            return pattern.pattern
    return None


class Patterns:
    """Compiles the filter regexes and triggers once; after that only searches."""

    def __init__(
        self,
        filters: FiltersConfig,
        name_triggers: list[str],
        bot_username: str,
        lang: Language | None = None,
    ) -> None:
        # The pack's language: name declensions and everything that depends on the alphabet
        # and morphology. Without a language (tests, tools without a pack) it is English, the
        # neutral default.
        lang = lang if lang is not None else get_language("en")
        self.lang = lang
        self._topic_stop = _compile_regex_list(filters.topic_stop)
        self._injection_markers = _compile_regex_list(filters.injection_markers)
        self._logistics = _compile_regex_list(filters.logistics)
        self._urgent = _compile_regex_list(filters.urgent)
        self._places_request = _compile_regex_list(filters.places_request)
        # A reason to look up the weather at a named place (CLAUDE.md, "weather somewhere else").
        self._weather_request = _compile_regex_list(filters.weather_request)
        self._model_talk = _compile_regex_list(filters.model_talk)
        self._assistant_markers = _compile_phrase_list(filters.assistant_markers)
        self._grumpy_markers = _compile_regex_list(filters.grumpy_markers)
        # Props and tall tales (CLAUDE.md, "less and more varied", measure 5) are
        # compiled here too, so that both the prompt (the {avoid} slot) and the output filter
        # (dedup:motif / style:story_quota) take one and the same ready set.
        self._motifs = {
            label: _compile_regex_list(patterns) for label, patterns in filters.motifs.items()
        }
        self._story_markers = _compile_regex_list(filters.story_markers)
        # "About myself" metric (CLAUDE.md, "about myself every other time"): /status only.
        self._self_markers = _compile_regex_list(filters.self_markers)
        # The name is looked up in all singular cases (CLAUDE.md, "name in cases"): "about
        # Fedor" and "ask the old man" are the same addressing as "Fedor" and "old man".
        # morph.name_pattern gives the forms as a closed list of endings, so that unrelated
        # words that merely start with the name still do not match.
        self._name_triggers = [
            re.compile(lang.name_pattern(trigger), re.IGNORECASE) for trigger in name_triggers
        ]
        # "\b" does not work before "@" (@ is not a word character), so on the left we use a
        # lookbehind "a non-word character before @", and "\b" on the right.
        self._bot_mention = (
            re.compile(rf"(?<!\w)@{re.escape(bot_username)}\b", re.IGNORECASE)
            if bot_username
            else None
        )

    def topic_stop(self, text: str) -> str | None:
        return _first_match(self._topic_stop, text)

    def injection(self, text: str) -> str | None:
        return _first_match(self._injection_markers, text)

    def logistics(self, text: str) -> str | None:
        return _first_match(self._logistics, text)

    def name_trigger(self, text: str) -> str | None:
        return _first_match(self._name_triggers, text)

    def mentions_bot(self, text: str) -> bool:
        if self._bot_mention is None:
            return False
        return self._bot_mention.search(text) is not None

    def urgent(self, text: str) -> bool:
        return _first_match(self._urgent, text) is not None

    def places_request(self, text: str) -> bool:
        return _first_match(self._places_request, text) is not None

    def weather_request(self, text: str) -> bool:
        return _first_match(self._weather_request, text) is not None

    def model_talk(self, text: str) -> str | None:
        return _first_match(self._model_talk, text)

    def assistant_marker(self, text: str) -> str | None:
        return _first_match(self._assistant_markers, text)

    def unsafe_note(self, text: str) -> str | None:
        """A note extracted by the model (a diary fact, a chat joke, a person's
        thread) goes into the system prompt for a long time: commands and talk about the
        model/assistant are not let in (Codex review)."""
        return self.injection(text) or self.model_talk(text) or self.assistant_marker(text)

    def grumpy(self, text: str) -> str | None:
        return _first_match(self._grumpy_markers, text)

    @property
    def motifs(self) -> dict[str, list[re.Pattern[str]]]:
        """Motif label -> compiled regexes (for motifs.py)."""
        return self._motifs

    @property
    def story_markers(self) -> list[re.Pattern[str]]:
        """Compiled tall-tale markers (for motifs.story_count)."""
        return self._story_markers

    @property
    def self_markers(self) -> list[re.Pattern[str]]:
        """Compiled "talks about itself" markers (for motifs.self_share)."""
        return self._self_markers
