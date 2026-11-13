"""Reply delays: weighted buckets, a fast bucket for collapsing, debounce.

Pure functions. ``rng`` is passed in from outside so tests are deterministic.
"""

from __future__ import annotations

import random

from sotto.config_models import BehaviourConfig


def pick_delay(cfg: BehaviourConfig, rng: random.Random, *, urgent: bool) -> int:
    """Pick a bucket by weight (cumulative rng.random()) and seconds inside it.

    urgent=True caps the result at cfg.urgent_max_delay_sec: an urgent message
    ("today", "where are we going"...) must not wait an hour if a long bucket came up.
    """
    roll = rng.random()
    cumulative = 0.0
    buckets = cfg.reply_delay_buckets
    chosen = buckets[-1]
    for bucket in buckets:
        cumulative += bucket.weight
        if roll < cumulative:
            chosen = bucket
            break

    lo, hi = chosen.range_sec
    delay = rng.randint(lo, hi)
    if urgent:
        delay = min(delay, cfg.urgent_max_delay_sec)
    return delay


def fast_delay(cfg: BehaviourConfig, rng: random.Random) -> int:
    """Delay from the first (fastest) bucket, used when collapsing repeated addresses."""
    lo, hi = cfg.reply_delay_buckets[0].range_sec
    return rng.randint(lo, hi)


def debounce_seconds(cfg: BehaviourConfig, rng: random.Random) -> float:
    """Debounce pause before the context is assembled: rng.uniform within debounce_sec."""
    lo, hi = cfg.debounce_sec
    return rng.uniform(lo, hi)
