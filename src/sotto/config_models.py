"""Pydantic models of config.yaml.

Ranges of numeric fields (Field(ge=..., le=...)) are sensible for a bot persona
in a single chat and are later used to validate the /set command.
"""

import re
from datetime import date, time

from pydantic import BaseModel, Field, field_validator, model_validator

from sotto.errors import OwnerError

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
# OpenRouter model id: "provider/model", e.g. "openai/gpt-4o-mini" or
# "anthropic/claude-3.5-sonnet:beta"; used by both llm.main_model and llm.judge_model.
_MODEL_ID_RE = re.compile(r"^[\w.-]+/[\w.:-]+$")


# The standard set of Telegram Bot API reactions (ReactionTypeEmoji.emoji). Anything else
# (e.g. a laughing-with-tears face) is rejected by Telegram with REACTION_INVALID. This is not
# filters.allowed_emoji: that list is for emoji in reply text.
TELEGRAM_REACTIONS: frozenset[str] = frozenset(
    {
        "👍",
        "👎",
        "❤",
        "🔥",
        "🥰",
        "👏",
        "😁",
        "🤔",
        "🤯",
        "😱",
        "🤬",
        "😢",
        "🎉",
        "🤩",
        "🤮",
        "💩",
        "🙏",
        "👌",
        "🕊",
        "🤡",
        "🥱",
        "🥴",
        "😍",
        "🐳",
        "❤‍🔥",
        "🌚",
        "🌭",
        "💯",
        "🤣",
        "⚡",
        "🍌",
        "🏆",
        "💔",
        "🤨",
        "😐",
        "🍓",
        "🍾",
        "💋",
        "🖕",
        "😈",
        "😴",
        "😭",
        "🤓",
        "👻",
        "👨‍💻",
        "👀",
        "🎃",
        "🙈",
        "😇",
        "😨",
        "🤝",
        "✍",
        "🤗",
        "🫡",
        "🎅",
        "🎄",
        "☃",
        "💅",
        "🤪",
        "🗿",
        "🆒",
        "💘",
        "🙉",
        "🦄",
        "😘",
        "💊",
        "🙊",
        "😎",
        "👾",
        "🤷‍♂",
        "🤷",
        "🤷‍♀",
        "😡",
    }
)

# Default reaction palette: emoji -> when it fits (the text the reaction model sees). The
# pack's config.yaml carries its own descriptions; these are generic. The model does not obey
# "do not repeat yourself", so repeats are limited in code (ReactionsConfig.max_repeat).
DEFAULT_REACTION_EMOJI: dict[str, str] = {
    "🤣": "very funny: a joke that really makes you laugh",
    "😁": "amusing: a good joke, but not a loud laugh",
    "👍": "genuine approval: a sound idea, a good deed",
    "💩": "playful approval, with a tease",
    "🔥": "impressive: a trip, a purchase, something made by hand",
    "👏": "well done: finished it, managed, achieved it",
    "🍾": "worth celebrating: a holiday, big good news",
}


def _validate_hhmm(value: str) -> str:
    if not _HHMM_RE.match(value):
        raise OwnerError("validation.time", value=value)
    return value


def parse_hhmm(value: str) -> time:
    """ "HH:MM" -> datetime.time."""
    hour_str, minute_str = value.split(":")
    return time(hour=int(hour_str), minute=int(minute_str))


class PersonaConfig(BaseModel):
    name: str = Field(default="Persona", description="Character name")
    display_name: str = Field(default="Persona", description="Bot profile name in Telegram")
    name_triggers: list[str] = Field(
        default_factory=list,
        description="Words that address the character without @, matched on word boundaries",
    )
    birth_date: date = Field(
        default=date(1970, 1, 1), description="Date of birth; the age is computed on the fly"
    )
    district: str = Field(default="", description="Home district the character stands up for")
    timezone: str = Field(
        default="UTC", description="Character's timezone for days and time windows"
    )

    def age(self, today: date) -> int:
        years = today.year - self.birth_date.year
        if (today.month, today.day) < (self.birth_date.month, self.birth_date.day):
            years -= 1
        return years


class LiveTalkConfig(BaseModel):
    """What counts as a "live conversation" for ambient replies."""

    min_messages: int = Field(
        default=2,
        ge=1,
        le=50,
        description="Minimum messages in the window to count as a live conversation",
    )
    min_people: int = Field(
        default=2,
        ge=1,
        le=50,
        description="Minimum distinct people in the window to count as a live conversation",
    )
    window_min: int = Field(
        default=20,
        ge=1,
        le=1440,
        description="Window in minutes over which a conversation is counted",
    )


class SpontaneousConfig(BaseModel):
    """Just because: the bot speaks on its own, without a reason, when the chat is quiet."""

    per_week: int = Field(
        default=2, ge=0, le=50, description="How many unprompted messages per week are allowed"
    )
    window: tuple[str, str] = Field(
        default=("10:00", "22:00"), description="Time window HH:MM-HH:MM for unprompted messages"
    )
    min_quiet_hours: int = Field(
        default=3, ge=0, le=24, description="Hours of silence in a row before an unprompted message"
    )

    @field_validator("window")
    @classmethod
    def _validate_window(cls, value: tuple[str, str]) -> tuple[str, str]:
        return (_validate_hhmm(value[0]), _validate_hhmm(value[1]))

    def window_time(self) -> tuple[time, time]:
        return (parse_hhmm(self.window[0]), parse_hhmm(self.window[1]))


class ReplyDelayBucket(BaseModel):
    """One bucket of the reply delay to an address."""

    weight: float = Field(
        ge=0.0, le=1.0, description="Weight of the bucket when picking a reply delay"
    )
    range_sec: tuple[int, int] = Field(description="Delay range in seconds, [min, max]")

    @field_validator("range_sec")
    @classmethod
    def _validate_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        lo, hi = value
        if not (0 <= lo <= hi <= 86400):
            raise OwnerError("validation.range_sec", value=value)
        return value


class ReactionsConfig(BaseModel):
    """Emoji reactions to messages cut by the gate on gate:dice/gate:ambient_cooldown.

    Instead of a text reply, only setMessageReaction: there is no new message,
    but there is an effect of presence. The reaction is not set instantly and not by
    dice: first a ``delay_sec`` pause (a person reads first), then with ``semantic`` a
    cheap model decides whether a reaction fits and which one.
    ``probability`` remains only for ``semantic: false`` (the old behaviour).
    ``emoji`` is validated at the ``Config`` level (see ``Config._check_reactions_emoji_telegram``):
    every element must be in ``TELEGRAM_REACTIONS`` (the Bot API reaction set),
    otherwise ``/set behaviour.reactions.emoji`` fails with a clear error.
    """

    enabled: bool = Field(default=True, description="Enables emoji reactions instead of silence")
    probability: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Chance to react when a reaction is possible"
    )
    cooldown_min: int = Field(
        default=60, ge=0, le=1440, description="At most one reaction per N minutes"
    )
    daily_cap: int = Field(
        default=8, ge=0, le=100, description="Reactions per day, separate from ambient/mention"
    )
    emoji: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_REACTION_EMOJI),
        description="Emoji -> when it fits (the model sees it); only Telegram reaction emoji",
    )
    max_repeat: int = Field(
        default=2,
        ge=0,
        le=20,
        description="The same emoji at most N times in a row; 0 means no limit",
    )
    delay_sec: tuple[int, int] = Field(
        default=(20, 120),
        description="Pause before a reaction: read it and nodded, not instant, [min, max]",
    )
    semantic: bool = Field(
        default=True, description="A cheap model picks the reaction; false means the old dice roll"
    )
    model: str = Field(default="", description="Reaction model; empty -> llm.judge_model")
    max_tokens: int = Field(
        default=600,
        ge=10,
        le=1000,
        description="Reaction model answer cap in tokens (the model may think first)",
    )
    semantic_daily_cap: int = Field(
        default=40, ge=0, le=1000, description="Reaction model calls per day"
    )
    context_messages: int = Field(
        default=6, ge=1, le=30, description="How many recent chat messages to show the model"
    )

    @field_validator("emoji", mode="before")
    @classmethod
    def _emoji_list_to_dict(cls, value: object) -> object:
        # The old format is a list without descriptions (`/set behaviour.reactions.emoji [👍, 🤣]`).
        if isinstance(value, list):
            return {str(e): "" for e in value}
        return value

    @field_validator("emoji")
    @classmethod
    def _validate_non_empty(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise OwnerError("validation.emoji_empty")
        # A phone keyboard appends the variation selector U+FE0F to an emoji, while the Telegram
        # reaction set has the emoji without it: strip it so /set does not reject a valid
        # reaction and exactly the form from the set goes to Telegram.
        return {e.replace("\ufe0f", ""): meaning.strip() for e, meaning in value.items()}

    @field_validator("delay_sec")
    @classmethod
    def _validate_delay_sec(cls, value: tuple[int, int]) -> tuple[int, int]:
        lo, hi = value
        if not (0 <= lo <= hi <= 3600):
            raise OwnerError("validation.delay_sec", value=value)
        return value


class StickersConfig(BaseModel):
    """Stickers instead of text: a second call to a cheap model. The main model knows
    nothing about the catalog; the text that already passed the output filter goes
    together with the trigger and the catalog to a second, cheap model
    (``model`` or, if empty, ``llm.judge_model``), which either names a sticker
    number or ``null``, and then the text goes out as usual.
    """

    enabled: bool = Field(default=True, description="Enables sending stickers instead of text")
    min_replies_between: int = Field(
        default=2,
        ge=0,
        le=50,
        description="Text replies that must pass after a sticker before the next one",
    )
    daily_cap: int = Field(default=1, ge=0, le=50, description="Stickers per day")
    recent_window: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Do not repeat this many of the latest used stickers",
    )
    model: str = Field(default="", description="Sticker chooser model; empty -> llm.judge_model")
    max_tokens: int = Field(
        default=60, ge=10, le=300, description="Token cap of the sticker chooser answer"
    )

    @field_validator("model")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if value == "":
            return value
        if not _MODEL_ID_RE.match(value):
            raise OwnerError("validation.model_id", value=value)
        return value


class HotWindowConfig(BaseModel):
    """Hot window after `/life` and `/say`: livelier than usual for a while.
    Outside the window the behaviour does not change at all.
    """

    enabled: bool = Field(default=True, description="Enables the hot window after /life and /say")
    minutes: int = Field(
        default=10, ge=0, le=720, description="Length of the hot window after /life and /say"
    )
    mention_max_delay_sec: int = Field(
        default=120,
        ge=0,
        le=3600,
        description="Cap on the reply delay to a direct address in the hot window",
    )
    ambient_probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Chance of an ambient reply in the hot window, replaces ambient_probability",
    )
    ambient_cap: int = Field(
        default=8, ge=0, le=50, description="Cap on ambient replies per hot window"
    )
    open_on_any_reply: bool = Field(
        default=False,
        description="Open the hot window after any reply, not only /life and /say",
    )


class FollowupConfig(BaseModel):
    """Cheap semantic check "is this for me or about my topic?" in the hot window.
    Messages that the ordinary gate did not recognize as a direct address are
    additionally checked by a small model during the hot window; a "yes" turns them
    into a ``Trigger.FOLLOWUP`` address.
    """

    enabled: bool = Field(
        default=True, description='Enables the cheap "is this for me?" check in the hot window'
    )
    model: str = Field(
        default="", description='Model for the "is this for me?" check; empty -> llm.judge_model'
    )
    max_tokens: int = Field(
        default=60, ge=10, le=300, description="Token cap of the cheap check answer"
    )
    daily_cap: int = Field(
        default=300,
        ge=0,
        le=5000,
        description="Cap on cheap-check calls per day, its own counter",
    )
    context_messages: int = Field(
        default=10, ge=1, le=50, description="How many chat messages to give the cheap check"
    )
    recent_replies: int = Field(
        default=3, ge=1, le=10, description="How many of the latest bot replies to give the check"
    )
    name_check: bool = Field(
        default=True,
        description="A cheap model confirms an address by name, not just the regex",
    )
    name_check_max_tokens: int = Field(
        default=60, ge=10, le=300, description="Token cap of the name-address check answer"
    )

    @field_validator("model")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if value == "":
            return value
        if not _MODEL_ID_RE.match(value):
            raise OwnerError("validation.model_id", value=value)
        return value


class CheckinConfig(BaseModel):
    """Came back to check: ``after_min`` minutes after the hot window closes, one
    call of the main model over everything written after the persona's last
    reply. Not to be confused with ``followup`` (a cheap check INSIDE the window):
    here the window is already closed, and the check is done by the main model and
    less often.
    """

    enabled: bool = Field(
        default=True, description='Enables the periodic "come back and look" check'
    )
    after_min: tuple[int, int] = Field(
        default=(120, 240),
        description="Minutes after the hot window closes until the check, [min, max]",
    )
    topic_max_hours: int = Field(
        default=48,
        ge=1,
        le=720,
        description="Hours without bot replies after which the topic counts as dead",
    )
    max_messages: int = Field(
        default=40, ge=1, le=200, description="How many messages after the last reply to take"
    )
    poll_sec: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Period of the background checkin_job loop, seconds",
    )
    quiet_min: int = Field(
        default=10,
        ge=0,
        le=180,
        description="Minutes of chat silence before the check; written later -> postpone",
    )
    prefilter: bool = Field(
        default=True, description="Cheap check of the message batch before the main model"
    )
    prefilter_max_tokens: int = Field(
        default=60, ge=10, le=300, description="Token cap of the cheap prefilter answer"
    )

    @field_validator("after_min")
    @classmethod
    def _validate_after_min(cls, value: tuple[int, int]) -> tuple[int, int]:
        lo, hi = value
        if not (0 <= lo <= hi):
            raise OwnerError("validation.after_min", value=value)
        return value


class PresenceConfig(BaseModel):
    """Presence cap: within a local day the bot does not say more than ``max_share``
    of the number of human messages plus ``free_replies`` in reserve. Under the cap
    only replies to the bot itself, ``/life`` and ``/say`` pass; everything else
    (addresses by name, ambient, "just because", the morning reply, "came back to
    check") stays silent until the end of the day.
    """

    enabled: bool = Field(default=True, description="Enables the daily presence cap of the bot")
    max_share: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Share of bot replies to people's messages over the local day",
    )
    free_replies: int = Field(
        default=2,
        ge=0,
        le=20,
        description="Replies per day on top of the share: for the morning of an empty chat",
    )

    def allowance(self, human_messages_today: int) -> int:
        """How many replies the bot can afford today given this number of human messages."""
        return int(human_messages_today * self.max_share) + self.free_replies

    def over_cap(self, *, human_messages_today: int, bot_replies_today: int) -> bool:
        """The cap is already reached (a disabled cap: never)."""
        if not self.enabled:
            return False
        return bot_replies_today >= self.allowance(human_messages_today)


class ChatMemoryConfig(BaseModel):
    """Long-term chat memory: once a week the model compresses past conversations
    into a few lines; the recap lives in the DB longer than the messages themselves
    (``message_retention_days``) and is mixed into the system prompt.
    """

    enabled: bool = Field(default=True, description="Enables recaps of past conversations")
    period_days: int = Field(
        default=7, ge=1, le=31, description="Length of one recap period in days"
    )
    run_window: tuple[str, str] = Field(
        default=("04:00", "06:00"), description="Local-time window HH:MM for the memory run"
    )
    in_prompt: int = Field(
        default=8, ge=0, le=52, description="How many of the latest recaps go into the prompt"
    )
    keep_days: int = Field(
        default=365, ge=7, le=3650, description="How many days recaps are kept in the database"
    )
    max_messages: int = Field(
        default=600,
        ge=50,
        le=5000,
        description="Cap on messages per recap, the latest are taken",
    )
    max_chars: int = Field(
        default=700, ge=100, le=3000, description="Cap on the length of one recap in characters"
    )
    max_tokens: int = Field(
        default=400, ge=50, le=2000, description="Token cap of the model answer for a recap"
    )
    model: str = Field(default="", description="Recap model; empty -> llm.main_model")
    backfill_periods: int = Field(
        default=4, ge=0, le=12, description="How many past periods to catch up on the first run"
    )

    @field_validator("run_window")
    @classmethod
    def _validate_run_window(cls, value: tuple[str, str]) -> tuple[str, str]:
        return (_validate_hhmm(value[0]), _validate_hhmm(value[1]))

    def run_window_time(self) -> tuple[time, time]:
        return (parse_hhmm(self.run_window[0]), parse_hhmm(self.run_window[1]))

    @field_validator("model")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if value == "":
            return value
        if not _MODEL_ID_RE.match(value):
            raise OwnerError("validation.model_id", value=value)
        return value


class DiaryConfig(BaseModel):
    """Day diary: after each of its replies a cheap model extracts a fact about the
    persona's own doings from it (e.g. "went to the sauna"), and the day's facts go
    into the system prompt in the {diary} slot so that it does not contradict itself.
    """

    enabled: bool = Field(
        default=True, description="Enables the day diary: facts about itself from its replies"
    )
    model: str = Field(default="", description="Fact extraction model; empty -> llm.judge_model")
    max_tokens: int = Field(
        default=80, ge=20, le=300, description="Token cap of the fact extraction answer"
    )
    daily_cap: int = Field(
        default=30, ge=0, le=500, description="Extraction calls per day, counter diary_calls"
    )
    week_days: int = Field(
        default=6, ge=0, le=30, description='How many past days to show in the "this week" line'
    )
    keep_days: int = Field(
        default=14, ge=1, le=90, description="How many days facts are kept in the database"
    )

    @field_validator("model")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if value == "":
            return value
        if not _MODEL_ID_RE.match(value):
            raise OwnerError("validation.model_id", value=value)
        return value


class JokesConfig(BaseModel):
    """Chat jokes: the weekly recap extracts the words that the chat picked up, and
    they go into the system prompt in the {jokes} slot.
    """

    enabled: bool = Field(default=True, description="Enables the memory of the chat's shared jokes")
    per_period: int = Field(
        default=3, ge=0, le=10, description="How many jokes to pull from one weekly period"
    )
    in_prompt: int = Field(
        default=6, ge=0, le=30, description="How many jokes to put into the system prompt"
    )
    cooldown_days: int = Field(
        default=5, ge=0, le=60, description="Days during which an already used joke is not shown"
    )


class VocabConfig(BaseModel):
    """Dictionary words: now and then, to the point, offer the model a couple of words
    from the dictionary loaded by the owner, and explain those that came up in conversation.
    """

    enabled: bool = Field(
        default=True, description="Enables vocabulary words from the /words dictionaries"
    )
    probability: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Chance to offer words to the model in one reply"
    )
    daily_cap: int = Field(
        default=2, ge=0, le=20, description="How many words per day the character may use"
    )
    recent_window: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Replies that must pass without a word before the next one",
    )
    offer_size: int = Field(
        default=8, ge=1, le=10, description="How many words to offer the model to choose from"
    )
    meaning_context: int = Field(
        default=5,
        ge=0,
        le=30,
        description="In how many recent messages to look for a word to give its meaning",
    )
    max_meanings: int = Field(
        default=3, ge=0, le=10, description="How many word meanings to put into the prompt at once"
    )
    request_markers: list[str] = Field(
        default_factory=list,
        description="Request to use dictionary words (regexes); the dictionary name counts too",
    )
    request_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many words to show the model when it was asked for them",
    )
    explain_markers: list[str] = Field(
        default_factory=list,
        description='Regexes for "what does that mean": after its word it is a question to it',
    )
    explain_window_min: int = Field(
        default=30,
        ge=0,
        le=1440,
        description='Minutes after its word during which "didn\'t get it" is a question to it',
    )
    explain_max_between: int = Field(
        default=2,
        ge=0,
        le=20,
        description='More messages than this after its word: "what does that mean" is not to it',
    )

    @field_validator("request_markers", "explain_markers")
    @classmethod
    def _validate_markers(cls, value: list[str]) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise OwnerError("validation.regex", pattern=pattern, error=str(exc)) from exc
        return value


class CallbackConfig(BaseModel):
    """So how did it go?: a few days later the persona itself asks a person how the
    story they told about ended.
    """

    enabled: bool = Field(
        default=True, description='Enables the "so how did it go?" questions to people'
    )
    min_days: int = Field(
        default=3,
        ge=1,
        le=30,
        description='At most one "so how did it go?" question per this many days',
    )
    max_age_days: int = Field(
        default=14, ge=1, le=90, description="A story older than this many days is not asked about"
    )
    author_active_days: int = Field(
        default=7,
        ge=1,
        le=60,
        description="The story's author must have written in the chat within this many days",
    )
    window: tuple[str, str] = Field(
        default=("11:00", "20:00"), description="Local-time window HH:MM for the question"
    )

    @field_validator("window")
    @classmethod
    def _validate_window(cls, value: tuple[str, str]) -> tuple[str, str]:
        return (_validate_hhmm(value[0]), _validate_hhmm(value[1]))


class VisionConfig(BaseModel):
    """Vision for photos. A chat photo is described by a vision model in a sentence
    or two, and the description becomes the message text ("[photo: ...]"); the gate
    and generation then work with it as with ordinary text. It costs money, so not
    every photo is described: a reason (an address or the hot window), sometimes
    dice, and always under a daily cap with its own ``vision_calls`` counter.
    """

    enabled: bool = Field(default=True, description="Enables photo descriptions by a vision model")
    model: str = Field(
        default="", description="Vision model for photo descriptions; empty -> llm.main_model"
    )
    max_tokens: int = Field(
        default=120, ge=20, le=500, description="Token cap of the vision model answer"
    )
    daily_cap: int = Field(
        default=20, ge=0, le=500, description="Photo descriptions per day, its own counter"
    )
    ambient_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Chance to describe a photo sent without a reason (no address, no window)",
    )
    max_width: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="The largest photo size not wider than this, in pixels, is used",
    )
    max_chars: int = Field(
        default=200, ge=40, le=500, description="Cap on the photo description length, characters"
    )

    @field_validator("model")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if value == "":
            return value
        if not _MODEL_ID_RE.match(value):
            raise OwnerError("validation.model_id", value=value)
        return value


class WeatherConfig(BaseModel):
    """Weather as the background of the persona's life. The source is Open-Meteo, a
    public API with no key.

    The home point coordinates are NOT here and are not hardcoded in the code: the
    repository is public, and where the owner lives is personal data. Latitude,
    longitude and name live in ``.env`` (``WEATHER_LATITUDE``/``WEATHER_LONGITUDE``/
    ``WEATHER_HOME_NAME``, see settings.py). If unset, the weather block is empty, while
    weather for a place named in a question still works.
    """

    enabled: bool = Field(
        default=True, description="Enables weather in the system prompt (Open-Meteo, no key)"
    )
    ttl_min: int = Field(
        default=30, ge=1, le=1440, description="Minutes a weather answer counts as fresh"
    )
    timeout_sec: int = Field(
        default=5, ge=1, le=60, description="Open-Meteo request timeout, seconds"
    )
    place_lookup: bool = Field(
        default=True, description="Look up the weather for a place asked about in an address"
    )
    lookup_model: str = Field(
        default="",
        description="Cheap model that extracts the place from a question; empty -> llm.judge_model",
    )
    lookup_max_tokens: int = Field(
        default=40, ge=10, le=200, description="Token cap of the place-extraction model answer"
    )
    lookup_daily_cap: int = Field(
        default=30, ge=0, le=500, description="Cap on such calls per day, counter weather_calls"
    )
    geocode_ttl_days: int = Field(
        default=30,
        ge=1,
        le=3650,
        description="Days the coordinates of a found place are remembered",
    )
    # The geocoder language and "home country" (formerly hardcoded in weather.py): the
    # language of place names and the country that is not appended to a name.
    geocode_language: str = Field(
        default="en", description="Language code in which the geocoder returns place names"
    )
    home_country_codes: list[str] = Field(
        default_factory=list,
        description="Country codes (ISO) that are not appended to a place name",
    )
    home_country_names: list[str] = Field(
        default_factory=list,
        description="Home country names (case-insensitive) when the geocoder gives no code",
    )

    @field_validator("lookup_model")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if value == "":
            return value
        if not _MODEL_ID_RE.match(value):
            raise OwnerError("validation.model_id", value=value)
        return value


class BehaviourConfig(BaseModel):
    quiet_window: tuple[str, str] = Field(
        default=("02:00", "07:00"), description="Window of full silence, direct addresses included"
    )
    morning_reply_window: tuple[str, str] = Field(
        default=("07:00", "08:00"),
        description="Window of the morning reply to everyone who called at night; random moment",
    )

    live_talk: LiveTalkConfig = Field(
        default_factory=LiveTalkConfig,
        description="What counts as a live conversation for ambient replies",
    )
    ambient_probability: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Chance of an ambient reply inside a live conversation",
    )
    chat_cooldown_min: int = Field(
        default=25, ge=0, le=1440, description="Minimum minutes between ambient replies"
    )
    daily_cap: int = Field(
        default=3, ge=0, le=50, description="Daily limit of ambient and unprompted replies together"
    )

    spontaneous: SpontaneousConfig = Field(
        default_factory=SpontaneousConfig,
        description="Unprompted: the bot writes by itself when the chat is quiet",
    )
    reactions: ReactionsConfig = Field(
        default_factory=ReactionsConfig,
        description="Emoji reactions instead of silence when the gate drops on the dice",
    )
    stickers: StickersConfig = Field(
        default_factory=StickersConfig, description="Stickers instead of text, a second model call"
    )
    hot_window: HotWindowConfig = Field(
        default_factory=HotWindowConfig,
        description="Livelier-than-usual window after /life and /say",
    )
    followup: FollowupConfig = Field(
        default_factory=FollowupConfig,
        description='Cheap "is this for me?" check for messages without an address in the window',
    )
    checkin: CheckinConfig = Field(
        default_factory=CheckinConfig,
        description='Periodic "come back and look" check after the hot window closes',
    )
    presence: PresenceConfig = Field(
        default_factory=PresenceConfig,
        description="Daily presence cap: share of bot replies to people's messages",
    )
    chat_memory: ChatMemoryConfig = Field(
        default_factory=ChatMemoryConfig,
        description="Long-term chat memory: recaps of past conversations by week",
    )
    diary: DiaryConfig = Field(
        default_factory=DiaryConfig,
        description="Day diary: facts about itself from its replies, slot {diary} in the prompt",
    )
    jokes: JokesConfig = Field(
        default_factory=JokesConfig,
        description="Shared chat jokes: the weekly recap pulls them, slot {jokes} in the prompt",
    )
    vocab: VocabConfig = Field(
        default_factory=VocabConfig,
        description="Words from /words dictionaries: now and then in a reply, meaning on request",
    )
    callback: CallbackConfig = Field(
        default_factory=CallbackConfig,
        description='"So how did it go?": a question to a person about their open story',
    )
    vision: VisionConfig = Field(
        default_factory=VisionConfig,
        description="Photo description by a vision model instead of the photo placeholder",
    )
    weather: WeatherConfig = Field(
        default_factory=WeatherConfig,
        description="Weather as background in the system prompt; home coordinates are in .env",
    )

    # Cooldown = delay, not refusal: an address always gets an answer, the cooldown
    # only shifts due_at in the responder (earliest).
    mention_cooldown_sec: int = Field(
        default=60,
        ge=0,
        le=86400,
        description="Cooldown of a reply to a direct address, per person, seconds",
    )
    mention_chat_cooldown_sec: int = Field(
        default=90,
        ge=0,
        le=86400,
        description="Cooldown of a reply to a direct address, per chat, seconds",
    )
    # While the chat members are still playing with it; lower later.
    mention_daily_cap: int = Field(
        default=50, ge=0, le=200, description="Daily limit of replies to direct addresses"
    )
    reply_as_reply_after_sec: int = Field(
        default=300,
        ge=0,
        le=86400,
        description="After how many seconds to answer as a reply, not just in the chat",
    )

    debounce_sec: tuple[int, int] = Field(
        default=(3, 7), description="Debounce range: collects the message queue before replying"
    )
    reply_delay_buckets: list[ReplyDelayBucket] = Field(
        default_factory=lambda: [
            ReplyDelayBucket(weight=0.6, range_sec=(30, 180)),
            ReplyDelayBucket(weight=0.3, range_sec=(180, 900)),
            ReplyDelayBucket(weight=0.1, range_sec=(900, 3600)),
        ],
        description="Reply delay buckets for an address, weight and range in seconds",
    )
    urgent_max_delay_sec: int = Field(
        default=180,
        ge=0,
        le=86400,
        description='Delay cap for urgent messages ("today", "now")',
    )
    late_reply_threshold_sec: int = Field(
        default=600, ge=0, le=86400, description="After how many seconds a reply counts as late"
    )
    context_window: int = Field(
        default=30, ge=1, le=200, description="How many latest messages the model gets as context"
    )
    recent_replies_memory: int = Field(
        default=20,
        ge=1,
        le=200,
        description="How many latest bot replies are remembered for dedup",
    )
    topic_cooldown_min: int = Field(
        default=45,
        ge=0,
        le=1440,
        description="Minutes of silence after the input stop list fires",
    )
    message_retention_days: int = Field(
        default=30, ge=1, le=3650, description="How many days messages are kept before deletion"
    )
    min_gap_sec: int = Field(
        default=300,
        ge=0,
        le=3600,
        description="Minimum seconds between any two bot messages in the chat",
    )

    @field_validator("quiet_window", "morning_reply_window")
    @classmethod
    def _validate_windows(cls, value: tuple[str, str]) -> tuple[str, str]:
        return (_validate_hhmm(value[0]), _validate_hhmm(value[1]))

    @field_validator("debounce_sec")
    @classmethod
    def _validate_debounce(cls, value: tuple[int, int]) -> tuple[int, int]:
        lo, hi = value
        if not (0 <= lo <= hi <= 600):
            raise OwnerError("validation.debounce_sec", value=value)
        return value

    @model_validator(mode="after")
    def _check_bucket_weights(self) -> "BehaviourConfig":
        total = sum(bucket.weight for bucket in self.reply_delay_buckets)
        if abs(total - 1.0) > 0.01:
            raise OwnerError("validation.bucket_weights", total=total)
        return self

    def quiet_window_time(self) -> tuple[time, time]:
        return (parse_hhmm(self.quiet_window[0]), parse_hhmm(self.quiet_window[1]))

    def morning_reply_window_time(self) -> tuple[time, time]:
        return (parse_hhmm(self.morning_reply_window[0]), parse_hhmm(self.morning_reply_window[1]))


class LlmConfig(BaseModel):
    provider: str = Field(default="openrouter", description="LLM provider, OpenAI-compatible API")
    main_model: str = Field(
        default="anthropic/claude-opus-5", description="Main model id in OpenRouter"
    )
    judge_model: str = Field(
        default="openai/gpt-5.4-nano",
        description="Small cheap judge model (output filter), another vendor",
    )
    timeout_sec: int = Field(default=30, ge=1, le=120, description="Model request timeout, seconds")
    max_tokens: int = Field(
        default=200, ge=1, le=4000, description="Token cap of the main model answer"
    )
    # 50 addresses + 3 ambient + morning + "just because" is about 55 main calls and
    # as many for the judge, about 110; a cap of 150 leaves headroom.
    daily_calls_cap: int = Field(
        default=150,
        ge=0,
        le=1000,
        description="Cap on model call attempts per day, the judge counts",
    )
    daily_budget_usd: float = Field(
        default=2.0,
        ge=0.0,
        le=1000.0,
        description="Daily budget in dollars; beyond it the bot stays silent",
    )
    circuit_errors: int = Field(
        default=5, ge=1, le=100, description="Errors in a row that trip the circuit breaker"
    )
    circuit_pause_min: int = Field(
        default=30, ge=0, le=1440, description="Minutes of pause after the circuit breaker trips"
    )
    alert_owner: bool = Field(
        default=True, description="Message the owner privately when the provider refuses (4xx)"
    )
    alert_interval_hours: int = Field(
        default=24, ge=1, le=168, description="At most once per this many hours for one status code"
    )
    # Fallback prices per 1M tokens: used only if the provider did not return usage.cost.
    price_in_usd_per_1m: float = Field(
        default=5.0,
        ge=0.0,
        description="Fallback price of input tokens per 1M if there is no usage.cost",
    )
    price_out_usd_per_1m: float = Field(
        default=25.0,
        ge=0.0,
        description="Fallback price of output tokens per 1M if there is no usage.cost",
    )

    @field_validator("main_model", "judge_model")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        if value == "":
            return value
        if not _MODEL_ID_RE.match(value):
            raise OwnerError("validation.model_id", value=value)
        return value


class PlacesConfig(BaseModel):
    min_rating: float = Field(
        default=4.2, ge=0.0, le=5.0, description="Minimum venue rating for the whitelist"
    )
    min_reviews: int = Field(
        default=50,
        ge=0,
        le=100_000,
        description="Minimum number of venue reviews for the whitelist",
    )
    require_operational: bool = Field(
        default=True, description="Count only operating (not closed) venues"
    )
    cache_ttl_days: int = Field(
        default=30,
        ge=1,
        le=3650,
        description="How often, in days, to refresh the cache with the fill script",
    )
    max_per_reply: int = Field(
        default=2, ge=0, le=20, description="Maximum venues mentioned in one reply"
    )
    # Search queries of the offline venue cache fill (places_fill.py); the pack lists them.
    queries: list[str] = Field(
        default_factory=list,
        description="Search queries for the offline venue cache fill (places_fill.py)",
    )
    # Google Places search anchor (centre, language, region); the pack sets all three.
    search_center: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Latitude and longitude of the venue search centre (places_fill.py)",
    )
    region_code: str = Field(
        default="", description="Google Places search region code (ISO 3166-1 alpha-2)"
    )
    language_code: str = Field(
        default="en", description="Language code of Google Places answers (names, reviews)"
    )
    district_aliases: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Districts: canonical name -> spellings (for picking places and district)",
    )
    default_district: str = Field(
        default="", description="Default district when no district is found in the address"
    )
    quiet_keywords: list[str] = Field(
        default_factory=list,
        description="Words in a request that select only quiet places",
    )
    outskirts_keywords: list[str] = Field(
        default_factory=list,
        description="Words in a request that select places outside the city",
    )
    address_outskirts_markers: list[str] = Field(
        default_factory=list,
        description="Address fragments that mark a place as outside the city (places_fill.py)",
    )

    @field_validator("search_center")
    @classmethod
    def _validate_search_center(cls, value: tuple[float, float]) -> tuple[float, float]:
        lat, lon = value
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise OwnerError("validation.search_center", value=value)
        return value


# Output filter stages that cut even with filters.shadow: true: repeats and broken
# style are exactly what made a chat stop the bot, and shadow let them through.
_ENFORCE_STAGES_DEFAULT = ["dedup", "style"]

_REGEX_LIST_FIELDS = (
    "topic_stop",
    "injection_markers",
    "logistics",
    "urgent",
    "places_request",
    "weather_request",
    "model_talk",
    "grumpy_markers",
    "story_markers",
    "self_markers",
)

# The allowed emoji set: small, used rarely and to the point. The order is fixed.
# See prompts/system.txt for what the thumbs-up and the poop emoji mean in the chat.
_ALLOWED_EMOJI_DEFAULT = ["🙁", "🙂", "😂", "😀", "🤪", "💩", "👍"]


class FiltersConfig(BaseModel):
    shadow: bool = Field(
        default=True, description="true = filters log but do not cut the reply (gentle start)"
    )
    places_whitelist: list[str] = Field(
        default_factory=list,
        description="Everyday brands that are not venues from the character card",
    )
    # Allowed emoji for regex:emoji/style:emoji_*.
    allowed_emoji: list[str] = Field(
        default_factory=lambda: list(_ALLOWED_EMOJI_DEFAULT),
        description="Emoji the character may use: a small set, rarely and to the point",
    )
    emoji_max_per_reply: int = Field(
        default=1, ge=0, le=3, description="Maximum allowed emoji in one reply"
    )
    # If at least one of the last N replies had an emoji, a new one is cut (style:emoji_freq).
    emoji_recent_window: int = Field(
        default=4, ge=0, le=20, description="Window of latest replies for the emoji frequency check"
    )
    # Venues from the character card: the whitelist of regex:venue/regex:latin.
    known_places: list[str] = Field(
        default_factory=list,
        description="Venues from the character card, whitelist of regex:venue/regex:latin",
    )
    # The persona's foreign vocabulary: the same whitelist, while for dedup:polish_freq (the
    # other way round) these are exactly the words whose frequency the rule limits.
    polish_words: list[str] = Field(
        default_factory=list,
        description="Character's foreign-word list: whitelist and subject of dedup:polish_freq",
    )
    districts: list[str] = Field(
        default_factory=list,
        description="City districts: in Latin script they do not count as an invented venue",
    )
    latin_extra_letters: str = Field(
        default="",
        description="Letters of a foreign alphabet beyond A-Za-z (diacritics), e.g. Polish",
    )
    topic_stop: list[str] = Field(
        default_factory=list,
        description="Topic stop list (input and output), regexes on word boundaries",
    )
    injection_markers: list[str] = Field(
        default_factory=list,
        description="Markers of bot control commands (input gate, step 5a)",
    )
    logistics: list[str] = Field(
        default_factory=list,
        description='Logistics phrases: time, "who is going", "count me out": do not interfere',
    )
    urgent: list[str] = Field(
        default_factory=list,
        description='Urgency markers ("today", "now"): speed up the reply delay bucket',
    )
    places_request: list[str] = Field(
        default_factory=list,
        description='Markers of a venue request ("where to go", "recommend")',
    )
    weather_request: list[str] = Field(
        default_factory=list,
        description="Markers of weather talk: a reason to look up the weather somewhere",
    )
    model_talk: list[str] = Field(
        default_factory=list,
        description="Markers of talk about the model/AI that the character does not know (output)",
    )
    assistant_markers: list[str] = Field(
        default_factory=list,
        description="Assistant marker phrases, out of character (output filter)",
    )
    # Dry/irritated wording (style:grumpy, output filter): a good-natured
    # character does not brush off questions and does not get angry at repeats.
    grumpy_markers: list[str] = Field(
        default_factory=list,
        description="Dry or irritated wording, out of character (output filter)",
    )
    # Shadow stays gentle for new layers, but dedup and style always cut:
    # otherwise repeats go out to the chat.
    enforce_stages: list[str] = Field(
        default_factory=lambda: list(_ENFORCE_STAGES_DEFAULT),
        description="Output filter stages that cut the reply even with shadow: true",
    )
    # Props (recurring story topics): label -> regexes.
    motifs: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Motif label -> regexes of props the character keeps coming back to",
    )
    motif_recent_window: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Window of latest replies: a motif from it cuts the candidate (dedup:motif)",
    )
    motif_avoid_window: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Window of latest replies whose motifs go into the prompt slot {avoid}",
    )
    story_markers: list[str] = Field(
        default_factory=list,
        description='Markers of a tale: "back in the day", "I remember" (tale quota)',
    )
    story_window: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Window of latest replies in which the tale quota is counted",
    )
    self_markers: list[str] = Field(
        default_factory=list,
        description='Markers of "talks about itself" (I, my, we): a /status metric only',
    )
    story_max: int = Field(
        default=1,
        ge=0,
        le=20,
        description="Tales allowed per story_window, beyond it style:story_quota cuts",
    )

    @field_validator("motifs")
    @classmethod
    def _validate_motifs(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        """Motif regexes are compiled with IGNORECASE, like all other lists.

        The check is here and not in ``_validate_regex_list``: that validator gets a
        flat ``list[str]``, while motifs are a "label -> list of regexes" dict.
        """
        for label, patterns in value.items():
            for pattern in patterns:
                try:
                    re.compile(pattern, re.IGNORECASE)
                except re.error as exc:
                    raise OwnerError(
                        "validation.motif_regex", pattern=pattern, label=label, error=str(exc)
                    ) from exc
        return value

    @field_validator(*_REGEX_LIST_FIELDS)
    @classmethod
    def _validate_regex_list(cls, value: list[str]) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise OwnerError("validation.regex", pattern=pattern, error=str(exc)) from exc
        return value


class Config(BaseModel):
    persona: PersonaConfig = Field(
        default_factory=PersonaConfig,
        description="Who the character is: name, age, district, timezone",
    )
    behaviour: BehaviourConfig = Field(
        default_factory=BehaviourConfig, description="When and how often the bot writes and replies"
    )
    llm: LlmConfig = Field(
        default_factory=LlmConfig,
        description="Model, budget and protection against LLM provider failures",
    )
    places: PlacesConfig = Field(
        default_factory=PlacesConfig,
        description="Filling and selection of venues for recommendations",
    )
    filters: FiltersConfig = Field(
        default_factory=FiltersConfig,
        description="Input and output filters, whitelists and stop lists",
    )

    @model_validator(mode="after")
    def _check_reactions_emoji_telegram(self) -> "Config":
        """behaviour.reactions.emoji is a subset of TELEGRAM_REACTIONS.

        Telegram accepts only its own standard set in reactions; anything else
        (e.g. a laughing face) gives REACTION_INVALID and the reaction is silently
        not set. So we check against the Bot API set, not against filters.allowed_emoji
        (that one is for text).
        """
        bad = [e for e in self.behaviour.reactions.emoji if e not in TELEGRAM_REACTIONS]
        if bad:
            raise OwnerError(
                "validation.emoji_not_telegram", key="behaviour.reactions.emoji", bad=bad
            )
        return self
