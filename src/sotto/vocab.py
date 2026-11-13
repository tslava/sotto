"""Vocabulary words from a dictionary (CLAUDE.md, "vocabulary words from a dictionary").

The owner uploads dictionaries as a file in a private chat (``/words``); there can be several.
Now and then, in the right place, the persona uses a word: in some replies (a die roll, a
daily cap, a pause of a few replies) the model is offered a couple of words with their
meanings. It also knows the meaning if asked: a dictionary word that came up in conversation
goes into the prompt with its meaning. There are no new model calls, only substitution into
the ``{vocab}`` slot.

A dictionary is the owner's data (like ``{life}``), not participant input, so it lives in
the system prompt. Dictionaries are not committed to the public repository: they are someone
else's copyrighted text and live in the DB (``vocab_dicts``); the repo holds only
``personas/demo/vocab.example.yaml``.

The pure part (parsing, matching, rendering) does no I/O; ``VocabStore`` is a thin wrapper
over the DB modelled on ``stores.PromptStore``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from sotto.config_models import VocabConfig
from sotto.errors import OwnerError
from sotto.fragments import Fragments
from sotto.i18n import Catalog
from sotto.lang import Language, get_language

if TYPE_CHECKING:
    from sotto.db import Database, VocabDictRow

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Stress marks (acute and grave): public glossaries put them in every word.
_ACCENTS_RE = re.compile("[\u0300\u0301]")
_SPACES_RE = re.compile(r"\s+")
_LT_RUN_RE = re.compile(r"<{3,}")
_GT_RUN_RE = re.compile(r">{3,}")
# The model copies dashes from the prompt into replies (CLAUDE.md, "post-processing").
_DASH_RE = re.compile(r"\s*[‒–—―]\s*")

# Word forms are looked up by stem and a CLOSED list of the language's endings
# (``Language.vocab_rules``; Codex review: "any tail up to N letters" confused one word
# with a different, shorter one). A stem shorter than three letters matches exactly only.
_MIN_STEM = 3
_PREFIX_KEY = 3

_WORD_MAX = 40
_MEANING_MAX = 300
_EXAMPLE_MAX = 200
_ABOUT_MAX = 200
_NAME_MAX = 40
_FORMS_MAX = 30
_WORDS_MAX = 2000
_ERRORS_SHOWN = 5


def normalize(text: str, lang: Language) -> str:
    """``lang.fold`` (case, "yo" letter, stress marks), spaces collapsed."""
    return _SPACES_RE.sub(" ", lang.fold(text)).strip()


def _tokens(text: str, lang: Language) -> list[str]:
    return _TOKEN_RE.findall(normalize(text, lang))


def _display(text: str) -> str:
    """Text for display: no stress marks, spaces collapsed, case preserved."""
    return _SPACES_RE.sub(" ", _ACCENTS_RE.sub("", text)).strip()


def _context_lang(info: ValidationInfo) -> Language:
    """Parsing language from ``model_validate(context=...)``; without a context (the model was
    built by hand) it is English: the language hardly affects "does the word contain letters"
    and word repetition."""
    context = info.context
    if isinstance(context, dict) and context.get("lang") is not None:
        return cast(Language, context["lang"])
    return get_language("en")


class VocabWord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str = Field(min_length=1, max_length=_WORD_MAX)
    meaning: str = Field(min_length=1, max_length=_MEANING_MAX)
    example: str = Field(default="", max_length=_EXAMPLE_MAX)
    forms: list[str] = Field(default_factory=list, max_length=_FORMS_MAX)

    @field_validator("word", "meaning", "example", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("word")
    @classmethod
    def _has_letters(cls, value: str, info: ValidationInfo) -> str:
        if not _tokens(value, _context_lang(info)):
            raise OwnerError("vocab_no_letters")
        return value

    @field_validator("forms")
    @classmethod
    def _forms(cls, value: list[str], info: ValidationInfo) -> list[str]:
        lang = _context_lang(info)
        cleaned = [form.strip() for form in value if form.strip()]
        for form in cleaned:
            if len(form) > _WORD_MAX or not _tokens(form, lang):
                raise OwnerError("vocab_bad_form", form=form[:_WORD_MAX])
        return cleaned


class VocabDict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=_NAME_MAX)
    about: str = Field(default="", max_length=_ABOUT_MAX)
    words: list[VocabWord] = Field(min_length=1, max_length=_WORDS_MAX)

    @field_validator("name", "about", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return " ".join(value.split()) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _unique_words(self, info: ValidationInfo) -> VocabDict:
        lang = _context_lang(info)
        seen: set[str] = set()
        for item in self.words:
            key = normalize(item.word, lang)
            if key in seen:
                raise OwnerError("vocab_duplicate", word=_display(item.word))
            seen.add(key)
        return self


_LocPart = int | str
_Issue = tuple[tuple[_LocPart, ...], "str | OwnerError"]


class VocabError(OwnerError):
    """A vocabulary file that does not validate; ``render`` composes the owner text.

    ``issues`` holds the pydantic error locations with either pydantic's own message (kept as is)
    or the ``OwnerError`` raised by a validator here (rendered through the catalog); ``more`` is
    how many further errors were not listed.
    """

    def __init__(self, issues: list[_Issue], more: int) -> None:
        super().__init__("vocab_invalid", issues=issues, more=more)
        self.issues = issues
        self.more = more

    def render(self, catalog: Catalog) -> str:
        lines = []
        for loc, msg in self.issues:
            text = f"Value error, {msg.render(catalog)}" if isinstance(msg, OwnerError) else msg
            where = _format_loc(loc, catalog)
            lines.append(f"{where}: {text}" if where else text)
        if self.more > 0:
            lines.append(catalog.t("errors.vocab_more", more=self.more))
        return "; ".join(lines)


def _format_loc(loc: tuple[_LocPart, ...], catalog: Catalog) -> str:
    parts: list[str] = []
    for i, part in enumerate(loc):
        if isinstance(part, int) and i > 0 and loc[i - 1] == "words":
            parts.append(catalog.t("errors.vocab_word_loc", n=part + 1))
        elif isinstance(part, int):
            parts.append(f"#{part + 1}")
        else:
            parts.append(part)
    return " → ".join(parts)


def parse_vocab(raw: str, lang: Language) -> VocabDict:
    """YAML -> VocabDict. Any error is an ``OwnerError`` (a ``ValueError``) the owner UI renders."""
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise OwnerError("vocab_yaml", error=exc) from exc
    if not isinstance(data, dict):
        raise OwnerError("vocab_shape")
    try:
        return VocabDict.model_validate(data, context={"lang": lang})
    except ValidationError as exc:
        issues: list[_Issue] = []
        for error in exc.errors()[:_ERRORS_SHOWN]:
            original = (error.get("ctx") or {}).get("error")
            message: str | OwnerError = (
                original if isinstance(original, OwnerError) else str(error["msg"])
            )
            issues.append((tuple(error["loc"]), message))
        raise VocabError(issues, len(exc.errors()) - _ERRORS_SHOWN) from exc


@dataclass(frozen=True, slots=True)
class VocabEntry:
    word: str  # without stress marks, as in the dictionary
    meaning: str
    example: str
    about: str
    dict_name: str


def _name_stem(name: str, lang: Language) -> str | None:
    tokens = _tokens(name, lang)
    if not tokens:
        return None
    return lang.vocab_name_stem(tokens[0])


@functools.lru_cache(maxsize=32)
def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


def vocab_requested(text: str, vocab: Vocab, cfg: VocabConfig) -> bool:
    """Asked to say something with a word: "say it the local way", "what words do you
    have?": ``request_markers`` or the stem of an enabled dictionary's name."""
    if not text or not vocab:
        return False
    if any(p.search(text) for p in _compile(tuple(cfg.request_markers))):
        return True
    tokens = _tokens(text, vocab.lang)
    return any(token.startswith(stem) for stem in vocab.name_stems for token in tokens)


def explain_requested(text: str, cfg: VocabConfig) -> bool:
    """ "I did not understand", "what does it mean", "meaning?": by ``explain_markers``. The
    regexes run on the original text (not on the normalized one, which has folded letters)."""
    return bool(text) and any(p.search(text) for p in _compile(tuple(cfg.explain_markers)))


@dataclass(frozen=True, slots=True)
class _Rule:
    stem: str
    endings: frozenset[str]
    reflexive: bool | None  # None: not a verb; True: the form ends in a reflexive suffix
    entry: VocabEntry


class Vocab:
    """Enabled dictionaries assembled into one matcher. A word repeated across two
    dictionaries takes the first (dictionaries go by name)."""

    def __init__(self, dicts: Sequence[VocabDict] = (), *, lang: Language) -> None:
        self.lang = lang
        entries: list[VocabEntry] = []
        self._exact: dict[str, VocabEntry] = {}
        self._phrases: dict[tuple[str, ...], VocabEntry] = {}
        self._rules: dict[str, list[_Rule]] = {}
        seen: set[str] = set()
        for vocab_dict in dicts:
            for item in vocab_dict.words:
                key = normalize(item.word, lang)
                if key in seen:
                    continue
                seen.add(key)
                entry = VocabEntry(
                    word=_display(item.word),
                    meaning=_display(item.meaning),
                    example=_display(item.example),
                    about=vocab_dict.about,
                    dict_name=vocab_dict.name,
                )
                entries.append(entry)
                self._index(entry, item)
        self.entries: tuple[VocabEntry, ...] = tuple(entries)
        self._max_phrase = max((len(p) for p in self._phrases), default=0)
        self.name_stems: tuple[str, ...] = tuple(
            stem for stem in (_name_stem(d.name, lang) for d in dicts) if stem is not None
        )

    def _index(self, entry: VocabEntry, item: VocabWord) -> None:
        for variant in (item.word, *item.forms):
            tokens = tuple(_tokens(variant, self.lang))
            if len(tokens) > 1:
                self._phrases.setdefault(tokens, entry)
            elif tokens:
                self._exact.setdefault(tokens[0], entry)
        word_tokens = _tokens(item.word, self.lang)
        if item.forms or len(word_tokens) != 1:
            return
        for vocab_rule in self.lang.vocab_rules(word_tokens[0]):
            if len(vocab_rule.stem) >= _MIN_STEM:
                rule = _Rule(vocab_rule.stem, vocab_rule.endings, vocab_rule.reflexive, entry)
                self._rules.setdefault(vocab_rule.stem[:_PREFIX_KEY], []).append(rule)
            for form in vocab_rule.exact_forms:
                self._exact.setdefault(form, entry)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def _match_token(self, token: str) -> VocabEntry | None:
        exact = self._exact.get(token)
        if exact is not None:
            return exact
        found = self._match_rules(token, verbs_only=False)
        if found is not None:
            return found
        for tail in self.lang.vocab_prefix_tails(token):
            found = self._match_rules(tail, verbs_only=True)
            if found is not None:
                return found
        return None

    def _match_rules(self, token: str, *, verbs_only: bool) -> VocabEntry | None:
        # Several matching words: the longest stem wins (a long word against a shorter
        # prefix of it). A reflexive and a non-reflexive verb with one stem do not compete:
        # the reflexive-suffix form fits only the reflexive one.
        best: _Rule | None = None
        for rule in self._rules.get(token[:_PREFIX_KEY], ()):
            if not token.startswith(rule.stem) or (verbs_only and rule.reflexive is None):
                continue
            rest = token[len(rule.stem) :]
            if rule.reflexive:
                suffix = next((x for x in self.lang.reflexive_suffixes if rest.endswith(x)), None)
                if suffix is None:
                    continue
                rest = rest[: -len(suffix)]
            if rest in rule.endings and (best is None or len(rule.stem) > len(best.stem)):
                best = rule
        return best.entry if best is not None else None

    def find(self, text: str) -> list[VocabEntry]:
        """Dictionary words mentioned in ``text``, in order of first appearance, no repeats."""
        if not self.entries or not text:
            return []
        tokens = _tokens(text, self.lang)
        found: list[VocabEntry] = []
        seen: set[int] = set()
        i = 0
        while i < len(tokens):
            matched: VocabEntry | None = None
            width = 1
            for size in range(min(self._max_phrase, len(tokens) - i), 1, -1):
                phrase = self._phrases.get(tuple(tokens[i : i + size]))
                if phrase is not None:
                    matched, width = phrase, size
                    break
            if matched is None:
                matched = self._match_token(tokens[i])
            if matched is not None and id(matched) not in seen:
                seen.add(id(matched))
                found.append(matched)
            i += width
        return found


def _speech(replies: Sequence[str], frag: Fragments) -> list[str]:
    """Replies without sticker entries (the prefix comes from the package's
    ``placeholders.sticker_tag``)."""
    prefix = frag.prefix("placeholders.sticker_tag")
    return [reply for reply in replies if not reply.startswith(prefix)]


def offer_allowed(
    *,
    cfg: VocabConfig,
    vocab: Vocab,
    recent_replies: Sequence[str],
    count_today: int,
    rng: random.Random,
    frag: Fragments,
) -> bool:
    """Whether to offer the model words in this reply. The die roll is last: ``rng`` is
    consumed only when everything else allows (like ``reactions.pick_reaction``)."""
    if not cfg.enabled or not vocab:
        return False
    if count_today >= cfg.daily_cap:
        return False
    window = _speech(recent_replies, frag)[-cfg.recent_window :] if cfg.recent_window > 0 else []
    if any(vocab.find(reply) for reply in window):
        return False
    return rng.random() < cfg.probability


def pick_offer(
    vocab: Vocab,
    *,
    recent_replies: Sequence[str],
    size: int,
    rng: random.Random,
    exclude: Sequence[VocabEntry] = (),
    fill: bool = False,
    frag: Fragments,
) -> list[VocabEntry]:
    """``size`` random words that were not in ``recent_replies`` and are not in ``exclude``.
    ``fill``: if there are not enough of those (a small dictionary, everything sounded
    recently), top up from the rest: on a direct request an empty list is worse than a
    repeat (Codex review)."""
    used = {id(entry) for reply in _speech(recent_replies, frag) for entry in vocab.find(reply)}
    used.update(id(entry) for entry in exclude)
    pool = [entry for entry in vocab.entries if id(entry) not in used]
    picked = rng.sample(pool, min(size, len(pool)))
    if fill and len(picked) < size:
        chosen = {id(entry) for entry in picked}
        rest = [entry for entry in vocab.entries if id(entry) not in chosen]
        picked.extend(rng.sample(rest, min(size - len(picked), len(rest))))
    return picked


def _clean(text: str) -> str:
    cleaned = _GT_RUN_RE.sub(" ", _LT_RUN_RE.sub(" ", text))
    cleaned = _DASH_RE.sub(", ", cleaned)
    return " ".join(cleaned.split()).strip().strip('«»"').rstrip(".;").strip()


def _meaning(text: str) -> str:
    """Meaning in lower case: dictionaries capitalize it ("Wet, slushy weather") while the
    prompt uses it mid-sentence. Abbreviations ("NATO") are left alone."""
    cleaned = _clean(text)
    if len(cleaned) > 1 and cleaned[0].isupper() and not cleaned[1].isupper():
        return cleaned[0].lower() + cleaned[1:]
    return cleaned


def render_vocab(
    offer: Sequence[VocabEntry],
    meanings: Sequence[VocabEntry],
    *,
    requested: bool = False,
    frag: Fragments,
) -> str:
    """Block for the ``{vocab}`` slot: an offer of words and/or the meanings of words that came up.
    ``requested`` means the persona was asked outright to use a word (a different heading).
    Empty -> "". No "- " at line starts (regex:prompt_leak) and no dashes."""
    parts: list[str] = []
    if offer:
        abouts = {_clean(entry.about) for entry in offer}
        about = abouts.pop() if len(abouts) == 1 else ""
        quote_open, quote_close = frag.text("render.quote_open"), frag.text("render.quote_close")
        items = []
        for entry in offer:
            example = _clean(entry.example)
            meaning = _meaning(entry.meaning)
            detail = f"{meaning}; {quote_open}{example}{quote_close}" if example else meaning
            items.append(f"{_clean(entry.word)} ({detail})")
        kind = "request" if requested else "offer"
        base, tail = frag.get(f"vocab.{kind}_header"), frag.get(f"vocab.{kind}_tail")
        header = f"{base} ({about})" if about else base
        parts.append(header + tail + "; ".join(items) + ".")
    if meanings:
        items = [f"{_clean(entry.word)}: {_meaning(entry.meaning)}" for entry in meanings]
        parts.append(frag.text("vocab.meanings_header") + "; ".join(items) + ".")
    return "\n".join(parts)


class VocabStore:
    """Dictionaries from the DB assembled into a ``Vocab``. The source of truth is the
    ``vocab_dicts`` table; memory holds only the parsed enabled dictionaries, rebuilt after
    every edit."""

    def __init__(self, db: Database, lang: Language | None = None) -> None:
        lang = lang if lang is not None else get_language("en")
        self._db = db
        self._lang = lang
        self._lock = asyncio.Lock()
        self._vocab = Vocab(lang=lang)

    def get(self) -> Vocab:
        return self._vocab

    async def load(self) -> None:
        async with self._lock:
            await self._load_locked()

    async def _load_locked(self) -> None:
        dicts: list[VocabDict] = []
        for row in await self._db.vocab_dicts():
            if not row.enabled:
                continue
            try:
                dicts.append(parse_vocab(row.body, self._lang))
            except ValueError as exc:
                logger.warning("vocab dict %r did not parse, skipped: %s", row.name, exc)
        self._vocab = Vocab(dicts, lang=self._lang)
        logger.info("vocab: %d dicts, %d words", len(dicts), len(self._vocab.entries))

    async def put(self, raw: str, now: int) -> VocabDict:
        """Parse and save a dictionary (the same ``name`` replaces). ``ValueError`` propagates."""
        parsed = parse_vocab(raw, self._lang)
        async with self._lock:
            await self._db.upsert_vocab_dict(parsed.name, raw, now)
            await self._load_locked()
        return parsed

    async def set_enabled(self, name: str, enabled: bool) -> bool:
        async with self._lock:
            changed = await self._db.set_vocab_dict_enabled(name, enabled)
            if changed:
                await self._load_locked()
            return changed

    async def remove(self, name: str) -> bool:
        async with self._lock:
            removed = await self._db.delete_vocab_dict(name)
            if removed:
                await self._load_locked()
            return removed

    async def dicts(self) -> list[tuple[VocabDictRow, VocabDict | None]]:
        result: list[tuple[VocabDictRow, VocabDict | None]] = []
        for row in await self._db.vocab_dicts():
            try:
                parsed: VocabDict | None = parse_vocab(row.body, self._lang)
            except ValueError:
                parsed = None
            result.append((row, parsed))
        return result


def vocab_over_quota(
    vocab: Vocab,
    *,
    cfg: VocabConfig,
    recent_replies: Sequence[str],
    count_today: int,
    frag: Fragments,
) -> bool:
    """Enough words for today: the daily cap is used up, or the word already sounded in the
    last ``recent_window`` replies. Then the output filter cuts a reply with an unsolicited
    word (``dedup:vocab``): a hint in the prompt alone does not hold the limit
    (Codex review)."""
    if count_today >= cfg.daily_cap:
        return True
    window = _speech(recent_replies, frag)[-cfg.recent_window :] if cfg.recent_window > 0 else []
    return any(vocab.find(reply) for reply in window)
