"""Сообщение владельцу в личку, когда провайдер модели отказывает (CLAUDE.md,
"Интерфейсы: тревога владельцу").

Решение владельца 26.10.2026: с 22.10 OpenRouter четыре дня отвечал 402 (кончились
кредиты), бот молчал, а заметили это только по разбору логов. Отказ провайдера
(4xx, кроме таймаута и rate limit) — не случайный сбой, сам он не пройдёт, поэтому
бот пишет владельцу сам: не чаще раза в ``llm.alert_interval_hours`` на код ответа.

Ревью Codex 26.10: отправка идёт фоновой задачей (вызов модели под локом ответов не
ждёт Telegram), время — по своим часам (``now`` вызывающего бывает датой старого
сообщения), а слот «следующая тревога не раньше» занимается ДО отправки: если Telegram
доставил, но ответ потерялся, повтора не будет. Не отправилось — следующая попытка
через час, а не через сутки.
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

# Временные сбои: проходят сами, тревога по ним — шум.
_TRANSIENT_STATUSES = frozenset({408, 429})
_MESSAGE_LIMIT = 200
_RETRY_AFTER_FAILURE_SEC = 3600
# Statuses with their own hint text in the catalog (``alerts.hint_<status>``).
_HINT_STATUSES = frozenset({401, 402})


def is_refusal(status: int) -> bool:
    """Отказ провайдера, который сам не пройдёт: 4xx, кроме таймаута и rate limit."""
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
        # Параллельные вызовы модели получают 402 одновременно — без лока слот заняли
        # бы несколько и ушло бы несколько одинаковых сообщений.
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
            # Остановка процесса посреди отправки: слот занят на весь интервал, а тревога,
            # возможно, не ушла — после рестарта повтор через час, не через сутки.
            await self._db.set_state(key, str(self._clock() + _RETRY_AFTER_FAILURE_SEC))
            raise
        except Exception as exc:
            # Тревога не ушла (или неизвестно, ушла ли) — следующая попытка через час.
            # Любое исключение: фоновая задача никому его не отдаст.
            logger.warning("owner alert not sent: %s", type(exc).__name__)
            await self._db.set_state(key, str(self._clock() + _RETRY_AFTER_FAILURE_SEC))
            return
        logger.info("owner alert sent: llm status=%s model=%s", status, model)

    async def shutdown(self) -> None:
        """Отменить неотправленные тревоги: при остановке процесса они не нужны."""
        tasks = [task for task in self._tasks if not task.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
