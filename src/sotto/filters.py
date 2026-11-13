"""Output filter: three layers.

Layer 1 (``layer_regex``) and layer 2 (``layer_rules``) are pure synchronous functions,
0 ms, no I/O. Layer 3 (the LLM judge, ``judge.py``) is called through ``check_output``
only if layers 1-2 passed or shadow mode is on. A failure of any layer means
silence, no retries. ``FilterContext``/``FilterVerdict`` are the contract dataclasses,
``check_output`` is the single entry point for ``responder.py``.

Word normalization for echo/dedup/prompt leak is the same everywhere: lower
case, language-specific folding, punctuation stripped, split on whitespace
(see ``_normalize_words``). The whitelist for "invented venues" and foreign-script words is
``places_names`` + ``filters.places_whitelist`` + ``filters.known_places`` (venues
from the persona card, word by word) + ``filters.polish_words`` (the persona's
vocabulary) + districts + foreign-script tokens that participants themselves
already used in ``trigger_text``/``context_rows`` (if a person named the place or
word, the bot repeats it rather than inventing it). For ``dedup:polish_freq`` the
whitelist is narrower: ``places_names``/``places_whitelist``/``known_places`` without
``polish_words``: those foreign words deliberately count as foreign tokens there,
which is the point of the rule.
"""

from __future__ import annotations

import functools
import re
import string
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol, cast

from sotto.config_models import Config
from sotto.db import MessageRow
from sotto.fragments import Fragments
from sotto.gate_types import PatternsLike
from sotto.lang import Language, get_language
from sotto.motifs import motifs_in, story_count, used_motifs
from sotto.patterns import Patterns
from sotto.vocab import Vocab, VocabEntry

_MAX_LEN = 300

_MARKDOWN_LIST_RE = re.compile(r"(^|\n)\s*[-*•] ")
_MARKDOWN_NUM_RE = re.compile(r"(^|\n)\s*\d+\.\s")
_MARKDOWN_HEADER_RE = re.compile(r"(^|\n)\s*#")

_EMOJI_RANGES = ((0x1F300, 0x1FAFF), (0x2600, 0x27BF))
_EMOJI_CATEGORIES = frozenset({"So", "Sk"})
# The variation selector U+FE0F and the skin-tone modifiers U+1F3FB-U+1F3FF that
# directly follow an emoji (e.g. a thumbs-up with a modifier) are part of the same
# symbol, not a separate emoji.
_VARIATION_SELECTOR_16 = "\ufe0f"
_SKIN_TONE_LO, _SKIN_TONE_HI = 0x1F3FB, 0x1F3FF

_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+(?:\s|$)")
# A fragment between [.!?…] separators counts as a sentence only from 3 words:
# a choppy story of short phrases ("Twice.", "It happens.") must not be falsely
# cut as a multi-sentence lecture.
_MIN_SENTENCE_WORDS = 3

_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")


@dataclass(frozen=True, slots=True)
class _Script:
    """Regexes of the "foreign" script: ``foreign_letters`` + ``filters.latin_extra_letters``."""

    # A token made entirely of alphabet letters.
    token: re.Pattern[str]
    # A word of three or more letters; a capital first letter is checked separately (a rough
    # proxy for "looks like a venue": a regex cannot recognize a venue).
    word: re.Pattern[str]
    # A token of two or more letters, for the frequency of foreign words (dedup:polish_freq).
    two_plus: re.Pattern[str]


@functools.lru_cache(maxsize=16)
def _script(letters: str) -> _Script:
    return _Script(
        token=re.compile(rf"[{letters}]+"),
        word=re.compile(rf"\b[{letters}]{{3,}}\b"),
        two_plus=re.compile(rf"[{letters}]{{2,}}"),
    )


def _foreign_script(ctx: FilterContext, lang: Language) -> _Script | None:
    """None: the language has no "foreign" script, so the latin and venue checks stay silent."""
    base = lang.foreign_letters
    if base is None:
        return None
    return _script(base + re.escape(ctx.cfg.filters.latin_extra_letters))


_PUNCT_CHARS = string.punctuation + "«»„”“—–…"

_NORMALIZE_STRIP_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SLOT_MARKER_RE = re.compile(r"\{[a-z_]+\}")


def _normalize_words(text: str, lang: Language) -> list[str]:
    """``lang.fold`` (case, yo, accents), strip punctuation, split on whitespace."""
    cleaned = _NORMALIZE_STRIP_RE.sub(" ", lang.fold(text))
    return cleaned.split()


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _first_word(text: str) -> str:
    """The first whitespace-delimited token of the text, without surrounding punctuation."""
    stripped = text.lstrip()
    match = re.match(r"\S+", stripped)
    if match is None:
        return ""
    return match.group(0).strip(_PUNCT_CHARS)


def _fold(text: str, lang: Language) -> str:
    return lang.fold(text.strip())


def _patterns_for(ctx: FilterContext) -> Patterns:
    """Return the caller's ready Patterns (``ctx.patterns``) if set.

    Otherwise build a new one from ``ctx.cfg``: a slow path, good only for tests
    that do not pass ``patterns`` explicitly. ``check_output`` calls this exactly
    once per check and hands the result to both layers, so the same set of
    regexes is not compiled twice for every candidate.
    """
    if ctx.patterns is not None:
        return cast(Patterns, ctx.patterns)
    lang = ctx.lang if ctx.lang is not None else get_language("en")
    return Patterns(ctx.cfg.filters, ctx.cfg.persona.name_triggers, "", lang)


def _latin_tokens_any(text: str, script: _Script) -> set[str]:
    return {m.group(0).casefold() for m in script.token.finditer(text)}


def _words_of(phrases: list[str]) -> set[str]:
    """The full phrase (casefold) + each separate word (casefold) from a list of phrases."""
    words: set[str] = set()
    for phrase in phrases:
        folded = phrase.casefold()
        if folded:
            words.add(folded)
        words |= {w.casefold() for w in phrase.split()}
    return words


def _venue_whitelist(ctx: FilterContext, script: _Script) -> set[str]:
    """Whitelist for regex:venue and regex:latin (shared, see the module docstring).

    places_names + filters.places_whitelist + filters.known_places (venues from the
    persona card, multi-word names word by word) + filters.districts +
    filters.polish_words (the persona's vocabulary) + foreign-script tokens that
    participants themselves already used in trigger_text/context_rows.
    """
    words = _words_of(
        [*ctx.places_names, *ctx.cfg.filters.places_whitelist, *ctx.cfg.filters.known_places]
    )
    words |= {name.casefold() for name in ctx.cfg.filters.districts}
    words |= {word.casefold() for word in ctx.cfg.filters.polish_words}
    words |= _latin_tokens_any(ctx.trigger_text, script)
    for row in ctx.context_rows:
        if row.text:
            words |= _latin_tokens_any(row.text, script)
    return words


def _places_whitelist(ctx: FilterContext) -> set[str]:
    """Narrower whitelist for dedup:polish_freq: places, not foreign words.

    places_names + filters.places_whitelist + filters.known_places, word by word (as
    in _venue_whitelist). filters.polish_words are deliberately not included: they
    are exactly the foreign tokens whose frequency dedup:polish_freq limits.
    """
    return _words_of(
        [*ctx.places_names, *ctx.cfg.filters.places_whitelist, *ctx.cfg.filters.known_places]
    )


# --------------------------------------------------------------------------- #
# Layer 1: regex:*
# --------------------------------------------------------------------------- #


def _has_markdown(text: str) -> bool:
    if _MARKDOWN_LIST_RE.search(text) or _MARKDOWN_NUM_RE.search(text):
        return True
    if "**" in text or "```" in text:
        return True
    return _MARKDOWN_HEADER_RE.search(text) is not None


def _is_emoji_char(ch: str) -> bool:
    code_point = ord(ch)
    if any(lo <= code_point <= hi for lo, hi in _EMOJI_RANGES):
        return True
    return unicodedata.category(ch) in _EMOJI_CATEGORIES


def _has_emoji(text: str) -> bool:
    return any(_is_emoji_char(ch) for ch in text)


def _is_emoji_modifier(ch: str) -> bool:
    """U+FE0F (variation selector) or a skin-tone modifier U+1F3FB-U+1F3FF directly
    following a base emoji is part of the same symbol."""
    if ch == _VARIATION_SELECTOR_16:
        return True
    return _SKIN_TONE_LO <= ord(ch) <= _SKIN_TONE_HI


def _emoji_tokens(text: str) -> list[tuple[str, int]]:
    """List of (base emoji symbol, end index) for every emoji in the text.
    Modifiers (``_is_emoji_modifier``) directly after the base symbol are absorbed
    by the same token and do not count as a separate emoji."""
    tokens: list[tuple[str, int]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if _is_emoji_char(ch):
            end = i + 1
            while end < n and _is_emoji_modifier(text[end]):
                end += 1
            tokens.append((ch, end))
            i = end
        else:
            i += 1
    return tokens


def _sentence_count(text: str) -> int:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return sum(1 for part in parts if len(part.split()) >= _MIN_SENTENCE_WORDS)


def _starts_name_set(ctx: FilterContext, lang: Language) -> set[str]:
    names: set[str] = set()
    for name in ctx.participant_names:
        folded = _fold(name, lang)
        if not folded:
            continue
        names.add(folded)
        words = folded.split()
        if words:
            names.add(words[0])
    for name in ctx.bot_names:
        folded = _fold(name, lang)
        if folded:
            names.add(folded)
    return names


def _check_starts_name(text: str, ctx: FilterContext, lang: Language) -> bool:
    first = _fold(_first_word(text), lang)
    if not first:
        return False
    return first in _starts_name_set(ctx, lang)


def _check_venue(text: str, whitelist: set[str], script: _Script) -> bool:
    for match in script.word.finditer(text):
        word = match.group(0)
        if word[0].isupper() and word.casefold() not in whitelist:
            return True
    return False


def _check_muted(text: str, muted_names: list[str]) -> bool:
    for name in muted_names:
        stripped = name.strip()
        if not stripped:
            continue
        if re.search(rf"\b{re.escape(stripped)}\b", text, re.IGNORECASE):
            return True
    return False


def _check_echo(text: str, ctx: FilterContext, lang: Language) -> bool:
    text_grams = _ngrams(_normalize_words(text, lang), 4)
    if not text_grams:
        return False
    for row in ctx.context_rows:
        if row.is_bot or not row.text:
            continue
        row_grams = _ngrams(_normalize_words(row.text, lang), 4)
        if text_grams & row_grams:
            return True
    return False


def _prompt_leak_source_lines(system_prompt: str) -> list[str]:
    """Only the instructive part of the prompt: bullet lines "- ..." (after strip) and
    lines containing "JSON" (the answer-format block). The biography is meant to be
    retold by the character and is deliberately not counted as a leak, even if the
    candidate repeats it almost verbatim.
    """
    return [
        line
        for raw_line in system_prompt.splitlines()
        for line in (raw_line.strip(),)
        if line.startswith("- ") or "JSON" in line
    ]


def _check_prompt_leak(text: str, system_prompt: str, lang: Language) -> bool:
    if not system_prompt:
        return False
    source_lines = _prompt_leak_source_lines(system_prompt)
    if not source_lines:
        return False
    cleaned_prompt = _SLOT_MARKER_RE.sub(" ", "\n".join(source_lines))
    prompt_grams = _ngrams(_normalize_words(cleaned_prompt, lang), 6)
    if not prompt_grams:
        return False
    text_grams = _ngrams(_normalize_words(text, lang), 6)
    return bool(prompt_grams & text_grams)


def _allowed_emoji_set(ctx: FilterContext) -> frozenset[str]:
    return frozenset(ctx.cfg.filters.allowed_emoji)


def _check_emoji_disallowed(text: str, allowed: frozenset[str]) -> bool:
    """regex:emoji: an emoji outside filters.allowed_emoji."""
    return any(ch not in allowed for ch, _ in _emoji_tokens(text))


def _check_latin_run(text: str, whitelist: set[str], script: _Script) -> bool:
    run = 0
    for raw_token in text.split():
        token = raw_token.strip(_PUNCT_CHARS)
        if token and script.token.fullmatch(token) and token.casefold() not in whitelist:
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def layer_regex(text: str, ctx: FilterContext, patterns: Patterns) -> list[str]:
    """Layer 1: regexes, 0 ms. Order as in the output filter contract.

    ``patterns`` is built once in ``check_output`` (``_patterns_for``) and passed
    here and to ``layer_rules``, so the regexes are not compiled twice for every
    candidate.
    """
    reasons: list[str] = []
    lang = patterns.lang
    script = _foreign_script(ctx, lang)

    if len(text) > _MAX_LEN:
        reasons.append("regex:length")
    if _has_markdown(text):
        reasons.append("regex:markdown")
    if _check_emoji_disallowed(text, _allowed_emoji_set(ctx)):
        reasons.append("regex:emoji")
    if _sentence_count(text) > 2:
        reasons.append("regex:sentences")
    if _check_starts_name(text, ctx, lang):
        reasons.append("regex:starts_name")
    if _PHONE_RE.search(text):
        reasons.append("regex:phone")

    whitelist = _venue_whitelist(ctx, script) if script is not None else set()
    if script is not None and _check_venue(text, whitelist, script):
        reasons.append("regex:venue")

    if patterns.topic_stop(text) is not None:
        reasons.append("regex:topic")
    if _check_muted(text, ctx.muted_names):
        reasons.append("regex:muted_name")
    if _check_echo(text, ctx, lang):
        reasons.append("regex:echo")
    if _check_prompt_leak(text, ctx.system_prompt, lang):
        reasons.append("regex:prompt_leak")
    if patterns.model_talk(text) is not None:
        reasons.append("regex:model_talk")
    if script is not None and _check_latin_run(text, whitelist, script):
        reasons.append("regex:latin")

    return reasons


# --------------------------------------------------------------------------- #
# Layer 2: dedup:* / style:*
# --------------------------------------------------------------------------- #


def _check_dedup_jaccard(text: str, recent_replies: list[str], lang: Language) -> bool:
    words = _normalize_words(text, lang)
    normalized = " ".join(words)
    if not normalized:
        return False

    if len(words) < 3:
        return any(
            " ".join(_normalize_words(reply, lang)) == normalized for reply in recent_replies
        )

    shingles = _ngrams(words, 3)
    for reply in recent_replies:
        reply_shingles = _ngrams(_normalize_words(reply, lang), 3)
        if _jaccard(shingles, reply_shingles) >= 0.6:
            return True
    return False


def _check_polish_freq(text: str, ctx: FilterContext, script: _Script) -> bool:
    whitelist = _places_whitelist(ctx)
    text_latin = {m.group(0).casefold() for m in script.two_plus.finditer(text)} - whitelist
    if not text_latin:
        return False

    last4 = ctx.recent_replies[-4:]
    last4_latin_sets = [
        {m.group(0).casefold() for m in script.two_plus.finditer(reply)} - whitelist
        for reply in last4
    ]

    if any(text_latin & reply_latin for reply_latin in last4_latin_sets):
        return True
    # "once per 5-6 replies": any foreign-script token in the last 4 replies is enough,
    # even if it does not literally match the candidate's token.
    return any(reply_latin for reply_latin in last4_latin_sets)


def _check_question_x2(text: str, recent_replies: list[str]) -> bool:
    if not text.rstrip().endswith("?"):
        return False
    if not recent_replies:
        return False
    return recent_replies[-1].rstrip().endswith("?")


_SELF_ECHO_RECENT = 20


def _check_self_echo(text: str, ctx: FilterContext, lang: Language) -> bool:
    """dedup:self_echo: the candidate retells its own recent story.

    The normalized 4-gram of the candidate (excluding grams made of stop-words only)
    is checked against the words of the last ``_SELF_ECHO_RECENT``
    ``ctx.recent_replies``: if all four words of the gram occur among the words of
    the same reply, the candidate is cut. The comparison is by word set, not by
    verbatim fragment (unlike ``regex:echo``): the goal is to catch a retelling of
    the same story in other words or order, not only a literal repetition.
    """
    grams = _ngrams(_normalize_words(text, lang), 4)
    grams = {gram for gram in grams if set(gram) - lang.stopwords}
    if not grams:
        return False
    for reply in ctx.recent_replies[-_SELF_ECHO_RECENT:]:
        reply_words = set(_normalize_words(reply, lang))
        if any(set(gram) <= reply_words for gram in grams):
            return True
    return False


def _check_motif(text: str, ctx: FilterContext, patterns: Patterns) -> bool:
    """dedup:motif: the candidate pulls the same prop that was in the last replies.

    A safety net for the prompt's ``{avoid}`` slot: the model was warned in advance
    not to mention certain props now, and if it brings them anyway, the reply is cut.
    The window ``filters.motif_recent_window`` is deliberately narrower than the
    ``{avoid}`` window (``motif_avoid_window``): asking for variety can be broad,
    but cutting only for a very fresh repeat.
    """
    window = ctx.cfg.filters.motif_recent_window
    if window <= 0:
        return False
    candidate = motifs_in(text, patterns.motifs)
    if not candidate:
        return False
    recent = used_motifs(ctx.recent_replies, window, patterns.motifs, frag=ctx.frag)
    return bool(set(candidate) & set(recent))


def _check_story_quota(text: str, ctx: FilterContext, patterns: Patterns) -> bool:
    """style:story_quota: a story when the story quota in the window is already used up.

    "A story not in every answer, one per several answers": if the last
    ``filters.story_window`` replies already hold ``filters.story_max`` stories or
    more, the next story is cut, while a short to-the-point answer passes.
    """
    filters_cfg = ctx.cfg.filters
    markers = patterns.story_markers
    if not markers:
        return False
    if not any(marker.search(text) for marker in markers):
        return False
    told = story_count(ctx.recent_replies, filters_cfg.story_window, markers, frag=ctx.frag)
    return told >= filters_cfg.story_max


def _check_emoji_count(text: str, allowed: frozenset[str], max_per_reply: int) -> bool:
    """style:emoji_count: more allowed emoji in the reply than emoji_max_per_reply."""
    count = sum(1 for ch, _ in _emoji_tokens(text) if ch in allowed)
    return count > max_per_reply


def _check_emoji_freq(text: str, recent_replies: list[str], window: int) -> bool:
    """style:emoji_freq: the text has an emoji AND at least one of the last
    ``window`` recent_replies also had an emoji (the "not more than once in five"
    rule is held by the output filter, not by the prompt)."""
    if window <= 0 or not _has_emoji(text):
        return False
    return any(_has_emoji(reply) for reply in recent_replies[-window:])


def _check_vocab(text: str, ctx: FilterContext) -> bool:
    """dedup:vocab: a dictionary word when today's quota of them is already used up. A word
    from the interlocutor's message ("what does X mean?") does not count: a question is answered."""
    if ctx.vocab is None or not ctx.vocab_over_quota:
        return False
    asked = set(ctx.vocab.find(ctx.trigger_text)) | ctx.vocab_allowed
    return any(entry not in asked for entry in ctx.vocab.find(text))


def layer_rules(text: str, ctx: FilterContext, patterns: Patterns) -> list[str]:
    """Layer 2: deterministic rules, 0 ms. ``patterns``: see ``layer_regex``."""
    reasons: list[str] = []
    lang = patterns.lang
    script = _foreign_script(ctx, lang)

    if _check_dedup_jaccard(text, ctx.recent_replies, lang):
        reasons.append("dedup:jaccard")
    if script is not None and _check_polish_freq(text, ctx, script):
        reasons.append("dedup:polish_freq")
    if _check_self_echo(text, ctx, lang):
        reasons.append("dedup:self_echo")
    if _check_motif(text, ctx, patterns):
        reasons.append("dedup:motif")
    if _check_vocab(text, ctx):
        reasons.append("dedup:vocab")

    if patterns.assistant_marker(text) is not None:
        reasons.append("style:assistant")
    if patterns.grumpy(text) is not None:
        reasons.append("style:grumpy")
    if _check_story_quota(text, ctx, patterns):
        reasons.append("style:story_quota")
    if _check_question_x2(text, ctx.recent_replies):
        reasons.append("style:question_x2")
    if text.count("!") > 1:
        reasons.append("style:exclaim")

    allowed_emoji = _allowed_emoji_set(ctx)
    if _check_emoji_count(text, allowed_emoji, ctx.cfg.filters.emoji_max_per_reply):
        reasons.append("style:emoji_count")
    if _check_emoji_freq(text, ctx.recent_replies, ctx.cfg.filters.emoji_recent_window):
        reasons.append("style:emoji_freq")

    return reasons


# --------------------------------------------------------------------------- #
# Contract: FilterContext / FilterVerdict / check_output
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FilterContext:
    cfg: Config
    recent_replies: list[str]  # the last 50 bot replies, chronological
    context_rows: list[MessageRow]  # fresh context (context_window)
    places_names: list[str]  # names from places
    participant_names: list[str] = field(default_factory=list)  # display_name of context authors
    bot_names: list[str] = field(default_factory=list)  # persona.name, display_name, name_triggers
    muted_names: list[str] = field(default_factory=list)  # display_name of muted participants
    # The caller's ready Patterns; None -> build from cfg (slow, for tests only).
    patterns: PatternsLike | None = None
    system_prompt: str = ""  # prompt body, for the leak check
    trigger_text: str = ""  # the trigger message ("" for ambient/spontaneous/morning)
    now: int = 0
    # Dictionary words: the matcher and the "today's quota is used up" flag
    # (vocab.vocab_over_quota). None/False: the dedup:vocab rule stays silent.
    vocab: Vocab | None = None
    vocab_over_quota: bool = False
    # Words allowed beyond the limit: those offered on request ("say it in
    # slang") and its own recent ones when asked to explain them.
    vocab_allowed: frozenset[VocabEntry] = frozenset()
    # Pack fragments: the sticker-entry prefix that motifs skips when counting.
    frag: Fragments = field(kw_only=True)
    # Language for the fallback ``Patterns`` (when ``patterns`` is not passed); None -> English.
    lang: Language | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class FilterVerdict:
    ok: bool
    reason: str  # the first cut reason or "pass"
    reasons: tuple[str, ...] = ()  # ALL reasons that fired (for shadow statistics)


class _JudgeLike(Protocol):
    """A narrow protocol instead of ``judge.Judge``: this module does not import it."""

    async def check(self, *, candidate: str, trigger_text: str, now: int) -> list[str]: ...


async def check_output(
    text: str, ctx: FilterContext, judge: _JudgeLike | None = None
) -> FilterVerdict:
    """Assemble the verdict from the three layers. Layer 3 is called only if layers 1-2

    passed (nothing to cut) or shadow mode is on (all layers are counted for
    statistics, but nothing is cut without the shadow exception in responder.py).
    """
    patterns = _patterns_for(ctx)
    reasons = layer_regex(text, ctx, patterns) + layer_rules(text, ctx, patterns)

    if judge is not None and (not reasons or ctx.cfg.filters.shadow):
        judge_reasons = await judge.check(
            candidate=text, trigger_text=ctx.trigger_text, now=ctx.now
        )
        reasons = reasons + judge_reasons

    ok = not reasons
    reason = reasons[0] if reasons else "pass"
    return FilterVerdict(ok=ok, reason=reason, reasons=tuple(reasons))
