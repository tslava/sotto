"""Direct message to the owner when the model provider refuses requests.

A provider refusal (4xx other than timeout and rate limit, e.g. 402 when credits run
out) will not clear by itself and would otherwise go unnoticed while the bot stays
silent, so the bot messages the owner, at most once per ``llm.alert_interval_hours``
per status code.

Design: sending runs as a background task (a model call holding the reply lock must not
wait on Telegram); time comes from our own clock (the caller's ``now`` can be the date
of an old message); the "next alert not before" slot is taken BEFORE sending, so a
delivered message whose response got lost is not repeated. If sending fails, the next
attempt comes in an hour, not a day.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import Protocol

from sotto.config_models import Config
from sotto.i18n import Catalog, engine_catalog
from sotto.persona import Persona

logger = logging.getLogger(__name__)

# Transient failures clear on their own; alerting on them is noise.
_TRANSIENT_STATUSES = frozenset({408, 429})
_MESSAGE_LIMIT = 200
_RETRY_AFTER_FAILURE_SEC = 3600
# Statuses with their own hint text in the catalog (``alerts.hint_<status>``).
_HINT_STATUSES = frozenset({401, 402})


def is_refusal(status: int) -> bool:
    """A provider refusal that will not clear by itself: 4xx except timeout and rate limit."""
    return 400 <= status < 500 and status not in _TRANSIENT_STATUSES


def render_alert(
    *,
    status: int,
    model: str,
    message: str,
    interval_hours: int,
    catalog: Catalog | None = None,
) -> str:
    """The owner message about a provider refusal, in the pack's locale (English without one)."""
    cat = catalog if catalog is not None else engine_catalog()
    hint = cat.t(f"alerts.hint_{status}" if status in _HINT_STATUSES else "alerts.hint_default")
    lines = [cat.t("alerts.refused", status=status, hint=hint), cat.t("alerts.model", model=model)]
    if message:
        lines.append(cat.t("alerts.reply", message=message[:_MESSAGE_LIMIT]))
    lines.append(cat.t("alerts.tail", hours=interval_hours))
    return "\n".join(lines)


class _SenderLike(Protocol):
    async def send_message(self, chat_id: int, text: str) -> object: ...


class _StateStore(Protocol):
    async def get_state(self, key: str) -> str | None: ...

    async def set_state(self, key: str, value: str) -> None: ...


class OwnerAlerter:
    def __init__(
        self,
        *,
        bot: _SenderLike,
        db: _StateStore,
        admin_user_id: int,
        cfg_getter: Callable[[], Config],
        clock: Callable[[], int] = lambda: int(time.time()),
        persona: Persona | None = None,
    ) -> None:
        self._catalog = persona.catalog if persona is not None else None
        self._bot = bot
        self._db = db
        self._admin_user_id = admin_user_id
        self._cfg_getter = cfg_getter
        self._clock = clock
        # Parallel model calls get 402 at the same moment; without the lock several
        # would take the slot and send several identical messages.
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    async def provider_refused(self, *, status: int, model: str, message: str) -> None:
        cfg = self._cfg_getter().llm
        if not cfg.alert_owner or self._admin_user_id == 0 or not is_refusal(status):
            return
        key = f"alert_next:llm_{status}"
        async with self._lock:
            now = self._clock()
            next_raw = await self._db.get_state(key)
            if next_raw is not None and now < int(next_raw):
                return
            await self._db.set_state(key, str(now + cfg.alert_interval_hours * 3600))
        text = render_alert(
            status=status,
            model=model,
            message=message,
            interval_hours=cfg.alert_interval_hours,
            catalog=self._catalog,
        )
        task = asyncio.ensure_future(self._send(key, text, status=status, model=model))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send(self, key: str, text: str, *, status: int, model: str) -> None:
        try:
            await self._bot.send_message(self._admin_user_id, text)
        except asyncio.CancelledError:
            # Process stopped mid-send: the slot is held for the whole interval but the
            # alert may not have gone out, so after a restart retry in an hour, not a day.
            await self._db.set_state(key, str(self._clock() + _RETRY_AFTER_FAILURE_SEC))
            raise
        except Exception as exc:
            # The alert was not sent (or it is unknown whether it was): retry in an hour.
            # Any exception: a background task has nobody to hand it to.
            logger.warning("owner alert not sent: %s", type(exc).__name__)
            await self._db.set_state(key, str(self._clock() + _RETRY_AFTER_FAILURE_SEC))
            return
        logger.info("owner alert sent: llm status=%s model=%s", status, model)

    async def shutdown(self) -> None:
        """Cancel unsent alerts: they are not needed once the process stops."""
        tasks = [task for task in self._tasks if not task.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
