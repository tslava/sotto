"""LLMClient: the single place that calls the model in the whole project.

The order of checks and side effects in call() is a contract: circuit -> calls_cap ->
budget -> increment llm_calls (before the request) -> POST. There are no retries on
any layer (budget protection).

The DB dependency is narrowed to the _StateStore protocol (get_state/set_state/
increment_state/add_state_float) instead of the concrete Database, so tests of this
module only need structural conformance to the signatures.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from sotto.alerts import is_refusal
from sotto.config_models import Config
from sotto.timeutil import day_key

logger = logging.getLogger(__name__)

_URL = "https://openrouter.ai/api/v1/chat/completions"
_LOG_BODY_LIMIT = 200
_TEMPERATURE = 0.8


class _StateStore(Protocol):
    """The subset of Database methods that LLMClient needs: get_state/set_state/
    increment_state/add_state_float."""

    async def get_state(self, key: str) -> str | None: ...

    async def set_state(self, key: str, value: str) -> None: ...

    async def increment_state(self, key: str, by: int = 1) -> int: ...

    async def add_state_float(self, key: str, by: float) -> float: ...


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


class LLMError(Exception):
    """reason is one of "llm:timeout" | "llm:http" | "llm:budget" | "llm:calls_cap" |
    "llm:circuit_open" | "llm:empty"."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# Keys and the authorization header, in case the provider or a proxy echoes them in
# the error body: the body goes to the log and to the owner in a private message.
_SECRET_RE = re.compile(r"sk-or-[\w-]+|Bearer\s+\S+", re.IGNORECASE)
_REDACTED = "***"


def _redact(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, _REDACTED)
    return _SECRET_RE.sub(_REDACTED, text)


def _truncate_body(text: str) -> str:
    return text[:_LOG_BODY_LIMIT]


def _error_message(response: httpx.Response, api_key: str) -> str:
    """``error.message`` from the OpenRouter response body, else the start of the body; no secrets.
    Secrets are cut out before truncation so no visible piece of a key is left."""
    try:
        data: Any = response.json()
        message = str(data["error"]["message"])
    except (ValueError, KeyError, TypeError):
        message = response.text
    return _truncate_body(_redact(message, api_key))


class RefusalHook(Protocol):
    """Owner alert, implemented by alerts.OwnerAlerter.provider_refused. Called on
    every non-2xx provider response; the implementation decides which of them count
    as a refusal. It gets no time: the caller's ``now`` can be the date of an old
    message (photo description), while the alert uses its own clock."""

    async def __call__(self, *, status: int, model: str, message: str) -> None: ...


class LLMClient:
    def __init__(
        self,
        api_key: str,
        cfg_getter: Callable[[], Config],
        db: _StateStore,
        http: httpx.AsyncClient | None = None,
        on_refusal: RefusalHook | None = None,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        self._api_key = api_key
        self._on_refusal = on_refusal
        self._clock = clock
        self._cfg_getter = cfg_getter
        self._db = db
        self._http = (
            http if http is not None else httpx.AsyncClient(timeout=cfg_getter().llm.timeout_sec)
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        max_tokens: int,
        now: int,
        counter_key: str = "llm_calls",
        calls_cap: int | None = None,
    ) -> LLMResult:
        """Wrapper over ``call_raw`` for text messages (``content: str``).

        Existing callers (Responder, Judge, places_fill) work with text only; the
        narrow parameter type is kept here so their code does not change.
        """
        return await self.call_raw(
            messages,
            model=model,
            max_tokens=max_tokens,
            now=now,
            counter_key=counter_key,
            calls_cap=calls_cap,
        )

    async def call_raw(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
        now: int,
        counter_key: str = "llm_calls",
        calls_cap: int | None = None,
    ) -> LLMResult:
        """Same as ``call``, but a message ``content`` may be more than a string:
        a content array ``[{"type": "text", ...}, {"type": "image_url", ...}]``
        for vision requests (``stickers_fill.py``, reading the caption on a sticker).
        All the logic (budget, calls_cap, circuit, spend accounting) lives here, the
        single place that calls the model; ``call`` does not duplicate it, only
        narrows the type.

        ``counter_key``/``calls_cap`` give a separate daily attempts counter instead
        of the shared ``llm_calls``, with its own cap (``None`` ->
        ``cfg.llm.daily_calls_cap``, as before). Cheap checks such as the followup
        checker need it so they do not eat the main model's call budget; the dollar
        budget, the circuit and increment-before-request stay shared by all counters."""
        if not model:
            # An empty model is a configuration error (main_model not chosen yet), not
            # a call failure: the caller (Responder) decides; there are no retries here.
            raise ValueError("LLMClient.call: model must not be empty")

        cfg = self._cfg_getter()
        tz = cfg.persona.timezone

        circuit_until_raw = await self._db.get_state("llm_circuit_until")
        if circuit_until_raw is not None and int(circuit_until_raw) > now:
            raise LLMError("llm:circuit_open")

        cap = calls_cap if calls_cap is not None else cfg.llm.daily_calls_cap
        calls_key = day_key(counter_key, now, tz)
        calls_raw = await self._db.get_state(calls_key)
        calls_count = int(calls_raw) if calls_raw is not None else 0
        if calls_count >= cap:
            raise LLMError("llm:calls_cap")

        spent_key = day_key("llm_spent_usd", now, tz)
        spent_raw = await self._db.get_state(spent_key)
        spent = float(spent_raw) if spent_raw is not None else 0.0
        if spent >= cfg.llm.daily_budget_usd:
            raise LLMError("llm:budget")

        # Counted per attempt, not per success: the increment happens strictly before the request.
        await self._db.increment_state(calls_key, by=1)

        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": _TEMPERATURE,
            "usage": {"include": True},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Title": "sotto",
        }

        try:
            response = await self._http.post(
                _URL, json=body, headers=headers, timeout=cfg.llm.timeout_sec
            )
        except httpx.TimeoutException:
            logger.warning("llm timeout: model=%s timeout_sec=%s", model, cfg.llm.timeout_sec)
            await self._register_error(now, cfg)
            raise LLMError("llm:timeout") from None
        except httpx.HTTPError as exc:
            logger.warning("llm transport error: model=%s error=%s", model, type(exc).__name__)
            await self._register_error(now, cfg)
            raise LLMError("llm:http") from None

        if not (200 <= response.status_code < 300):
            logger.warning(
                "llm http error: status=%s body=%s",
                response.status_code,
                _truncate_body(_redact(response.text, self._api_key)),
            )
            await self._register_error(now, cfg)
            if is_refusal(response.status_code):
                # Refusals only (402, 401...): a fresh 429 or 5xx must not overwrite the real
                # refusal shown in /status. Time comes from the clock, not the caller's now
                # (for photos that is the message date).
                await self._db.set_state(
                    "llm_last_refusal", f"{self._clock()}|{response.status_code}"
                )
            await self._notify_refusal(response, model=model)
            raise LLMError("llm:http")

        try:
            data: Any = response.json()
        except ValueError:
            logger.warning(
                "llm invalid json: status=%s body=%s",
                response.status_code,
                _truncate_body(_redact(response.text, self._api_key)),
            )
            await self._register_error(now, cfg)
            raise LLMError("llm:http") from None

        try:
            text = str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError):
            logger.warning(
                "llm malformed response: status=%s body=%s",
                response.status_code,
                _truncate_body(_redact(response.text, self._api_key)),
            )
            await self._register_error(now, cfg)
            raise LLMError("llm:http") from None

        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        cost_raw = usage.get("cost")
        if cost_raw is None:
            # Fallback cost: used only if the provider did not return
            # usage.cost. Prices come from cfg.llm.price_in_usd_per_1m/price_out_usd_per_1m
            # (per 1M tokens) and change via /set without a restart.
            cost = (
                prompt_tokens * cfg.llm.price_in_usd_per_1m
                + completion_tokens * cfg.llm.price_out_usd_per_1m
            ) / 1_000_000
            logger.debug(
                "llm usage.cost missing, using price fallback: model=%s prompt_tokens=%s "
                "completion_tokens=%s cost_usd=%.6f",
                model,
                prompt_tokens,
                completion_tokens,
                cost,
            )
        else:
            cost = float(cost_raw)

        # The real request already happened and cost money regardless of what
        # ended up in content, so the spend is always counted.
        await self._db.add_state_float(spent_key, cost)

        if not text:
            # An empty answer is not a network failure; leave the error streak alone.
            raise LLMError("llm:empty")

        await self._db.set_state("llm_error_streak", "0")
        logger.info(
            "llm call ok: model=%s prompt_tokens=%s completion_tokens=%s cost_usd=%.6f",
            model,
            prompt_tokens,
            completion_tokens,
            cost,
        )
        return LLMResult(
            text=text,
            cost_usd=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def _notify_refusal(self, response: httpx.Response, *, model: str) -> None:
        if self._on_refusal is None:
            return
        try:
            await self._on_refusal(
                status=response.status_code,
                model=model,
                message=_error_message(response, self._api_key),
            )
        except Exception:
            # The alert is a side channel: its failure must not change the model call outcome.
            logger.exception("llm refusal hook failed")

    async def _register_error(self, now: int, cfg: Config) -> None:
        streak = await self._db.increment_state("llm_error_streak", by=1)
        if streak >= cfg.llm.circuit_errors:
            circuit_until = now + cfg.llm.circuit_pause_min * 60
            await self._db.set_state("llm_circuit_until", str(circuit_until))
            logger.warning(
                "llm circuit opened: streak=%s pause_min=%s until=%s",
                streak,
                cfg.llm.circuit_pause_min,
                circuit_until,
            )
