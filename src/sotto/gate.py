"""Gate: decides whether a message is a reason to reply.

A pure function with no I/O. It does not change state: changes are returned in
``Decision.state_changes`` and applied by the caller.

Order of steps (step 0, the chat_id filter, is already in the handler, before the gate is called):

1.  ``is_bot``                              -> DROP ``gate:is_bot``
2.  ``panic`` / ``stop_until > now``         -> DROP ``gate:panic`` / ``gate:stop``
3.  author in ``muted_user_ids``               -> DROP ``gate:muted``
4.  topic stop-list                            -> DROP ``gate:topic`` (+ topic_cooldown_until)
5a. command markers (injection)                -> DROP ``gate:injection``
6.  direct address (reply > mention > name):
    night                                     -> QUEUE_NIGHT ``gate:night_queued``
    presence cap (except REPLY/MENTION)       -> DROP ``gate:presence_cap``
    daily address limit                       -> DROP ``gate:mention_cap``
    otherwise                                 -> PASS ``pass:<trigger>``

    A direct address is never dropped by a cooldown (owner decision): the chat
    and per-person cooldowns (``mention_chat_cooldown_sec``/``mention_cooldown_sec``)
    are not checked here at all. The gate always lets an address through, and the
    cooldown becomes a reply delay in the responder (``earliest`` when a pending
    task is created or collapsed).
    The topic stop cooldown (``topic_cooldown_until``) does not silence addresses:
    it is only checked below, in the ambient branch.
7.  night (no address)                         -> DROP ``gate:night``
8.  logistics                                  -> DROP ``gate:logistics``
7a. ``topic_cooldown_until > now``             -> DROP ``gate:topic_cooldown`` (ambient only)
8a. presence cap                               -> DROP ``gate:presence_cap``

    Presence cap: within a local day the bot does not say more than
    ``presence.max_share`` of the number of human messages plus
    ``presence.free_replies``. A reply to the bot itself (``Trigger.REPLY``) and ``@``
    (``Trigger.MENTION``) always pass under the cap: the person wrote to it
    and expects an answer. Everything else (the name, since people often speak
    about it in the third person, and ambient) stays silent under the cap. No
    reactions are set on ``gate:presence_cap``: ``reactions.REACT_REASONS`` is unchanged.

    Hot window after ``/life``/``/say`` (``state.hot_until``): while it is open,
    steps 9-11 are replaced with:
    9h. ambient replies limit per window     -> DROP ``gate:hot_cap``
    10h. dice with ``hot_window.ambient_probability`` -> DROP ``gate:dice``;
         otherwise PASS ``pass:ambient_hot`` (step 9, "not a live conversation", is skipped).

    Outside the window, as before:
9.  not a live conversation                  -> DROP ``gate:not_live``
10. daily ambient limit / chat cooldown      -> DROP ``gate:ambient_cap``
    / ``gate:ambient_cooldown``
11. dice                                     -> DROP ``gate:dice``; otherwise PASS ``pass:ambient``
"""

from __future__ import annotations

import random

from sotto.config_models import Config
from sotto.gate_types import (
    Decision,
    GateMessage,
    GateState,
    PatternsLike,
    StateChange,
    Trigger,
    Verdict,
)
from sotto.timeutil import in_window

# Triggers that pass under the presence cap: a reply to the bot and @. The name does
# not: people often speak about it in the third person. responder._recheck uses
# the same constant.
PRESENCE_EXEMPT: frozenset[Trigger] = frozenset({Trigger.REPLY, Trigger.MENTION})


def _drop(reason: str) -> Decision:
    return Decision(verdict=Verdict.DROP, trigger=None, reason=reason)


def _direct_trigger(msg: GateMessage, patterns: PatternsLike) -> Trigger | None:
    """Direct address type, priority reply > mention > name; None if there is no address."""
    if msg.reply_to_bot:
        return Trigger.REPLY
    if patterns.mentions_bot(msg.text):
        return Trigger.MENTION
    if patterns.name_trigger(msg.text) is not None:
        return Trigger.NAME
    return None


def _over_presence_cap(state: GateState, cfg: Config) -> bool:
    """The presence cap for the local day is reached. The counters are already in
    ``GateState``, collected by ``gate_state``."""
    return cfg.behaviour.presence.over_cap(
        human_messages_today=state.human_messages_today,
        bot_replies_today=state.bot_replies_today,
    )


def _is_live(state: GateState, cfg: Config) -> bool:
    """Live conversation: minimum messages and distinct authors within the live_talk window."""
    live = cfg.behaviour.live_talk
    if len(state.recent) < live.min_messages:
        return False
    distinct_users = {activity.user_id for activity in state.recent}
    return len(distinct_users) >= live.min_people


def _direct_address_decision(
    msg: GateMessage,
    state: GateState,
    cfg: Config,
    now: int,
    trigger: Trigger,
) -> Decision:
    """Step 6: a separate budget for a direct address; logistics and live-talk do not apply.

    The chat cooldown (``mention_chat_cooldown_sec``) and the per-person one
    (``mention_cooldown_sec``) no longer come in here: a direct address is never
    dropped by a cooldown, only the daily cap and night can stop it.
    A cooldown shifts the moment of the reply instead of cancelling it; that is done
    by ``responder.py`` (``earliest`` when a pending task is created or collapsed)."""
    behaviour = cfg.behaviour
    if in_window(now, cfg.persona.timezone, behaviour.quiet_window):
        return Decision(verdict=Verdict.QUEUE_NIGHT, trigger=trigger, reason="gate:night_queued")
    # Presence cap: a reply to the bot and @ always pass (the person wrote
    # to it), an address by name does not.
    if trigger not in PRESENCE_EXEMPT and _over_presence_cap(state, cfg):
        return _drop("gate:presence_cap")
    if state.mention_count_today >= behaviour.mention_daily_cap:
        return _drop("gate:mention_cap")
    return Decision(verdict=Verdict.PASS, trigger=trigger, reason=f"pass:{trigger.value}")


def should_consider(
    msg: GateMessage,
    state: GateState,
    cfg: Config,
    patterns: PatternsLike,
    now: int,
    rng: random.Random,
) -> Decision:
    """Decide whether to reply to a message. See the module docstring for the order of steps."""
    behaviour = cfg.behaviour
    tz = cfg.persona.timezone

    # 1. another bot
    if msg.is_bot:
        return _drop("gate:is_bot")

    # 2. panic / temporary stop
    if state.panic:
        return _drop("gate:panic")
    if state.stop_until is not None and state.stop_until > now:
        return _drop("gate:stop")

    # 3. muted author
    if msg.user_id in state.muted_user_ids:
        return _drop("gate:muted")

    # 4. topic stop-list on input: sets a cooldown
    if patterns.topic_stop(msg.text) is not None:
        cooldown_until = now + behaviour.topic_cooldown_min * 60
        return Decision(
            verdict=Verdict.DROP,
            trigger=None,
            reason="gate:topic",
            state_changes=(StateChange("topic_cooldown_until", str(cooldown_until)),),
        )

    # 5a. command markers: silence without a cooldown
    if patterns.injection(msg.text) is not None:
        return _drop("gate:injection")

    # 6. direct address: its own budget, steps 7-11 do not apply
    trigger = _direct_trigger(msg, patterns)
    if trigger is not None:
        return _direct_address_decision(msg, state, cfg, now, trigger)

    # 7. night window for ambient
    if in_window(now, tz, behaviour.quiet_window):
        return _drop("gate:night")

    # 7a. cooldown after the stop-list: ambient only, addresses do not check it
    if state.topic_cooldown_until is not None and state.topic_cooldown_until > now:
        return _drop("gate:topic_cooldown")

    # 8. logistics filter
    if patterns.logistics(msg.text) is not None:
        return _drop("gate:logistics")

    # 8a. presence cap: before both the live conversation and the hot window; if the bot
    # has already said its share for the day, no unaddressed reason makes it talk more.
    if _over_presence_cap(state, cfg):
        return _drop("gate:presence_cap")

    # Hot window after /life and /say: while it is open, chat liveness (step 9) is
    # not checked, and the daily limit/cooldown of ambient (step 10) are replaced
    # with a per-window budget and its own probability.
    hot_window = behaviour.hot_window
    hot = hot_window.enabled and state.hot_until is not None and now < state.hot_until
    if hot:
        if state.hot_ambient_count >= hot_window.ambient_cap:
            return _drop("gate:hot_cap")
        if rng.random() >= hot_window.ambient_probability:
            return _drop("gate:dice")
        return Decision(verdict=Verdict.PASS, trigger=Trigger.AMBIENT, reason="pass:ambient_hot")

    # 9. live conversation
    if not _is_live(state, cfg):
        return _drop("gate:not_live")

    # 10. daily ambient limit and chat cooldown
    if state.ambient_count_today >= behaviour.daily_cap:
        return _drop("gate:ambient_cap")
    if (
        state.last_ambient_at is not None
        and state.last_ambient_at + behaviour.chat_cooldown_min * 60 > now
    ):
        return _drop("gate:ambient_cooldown")

    # 11. dice
    if rng.random() >= behaviour.ambient_probability:
        return _drop("gate:dice")
    return Decision(verdict=Verdict.PASS, trigger=Trigger.AMBIENT, reason="pass:ambient")
