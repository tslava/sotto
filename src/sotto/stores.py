"""State that changes hot: config and prompt/few-shot versions.

ConfigStore holds the current Config (yaml + config_overrides from the DB) and the
rebuilt Patterns; PromptStore holds the active system prompt and few-shot version,
loading them from the DB and seeding from files on first start. Both survive a restart
through the DB, which after seeding is always the source of truth, not the files.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import yaml

from sotto.config import ConfigError, KeyInfo, describe_key, flatten_config, load_config
from sotto.config_models import Config
from sotto.db import Database
from sotto.errors import OwnerError
from sotto.few_shot import FewShot, render_few_shot
from sotto.patterns import Patterns
from sotto.persona import Persona

_ALLOWED_PREFIXES = ("behaviour.", "llm.", "places.", "filters.")


def _rebuild_patterns(cfg: Config, bot_username: str, persona: Persona) -> Patterns:
    return Patterns(cfg.filters, cfg.persona.name_triggers, bot_username, persona.lang)


def _first_error_message(exc: ValueError, key: str) -> str:
    """Readable error text: strips the load_config wrapper and key duplication.

    On ValidationError load_config gives "invalid config: loc1: msg1; loc2: msg2";
    for an unknown key/invalid YAML it gives plain text without the wrapper. In both
    cases take the first message and, if it already starts with key, do not repeat it.
    """
    message = str(exc)
    prefix = "invalid config: "
    if message.startswith(prefix):
        message = message[len(prefix) :]
    first = message.split("; ")[0]
    if first.startswith(f"{key}: "):
        first = first[len(key) + 2 :]
    return first


def _parse_few_shot_yaml(body_yaml: str) -> list[FewShot]:
    """Few-shot parser from YAML text: like load_few_shot but without reading a file."""
    loaded = yaml.safe_load(body_yaml)
    if loaded is None:
        return []
    if not isinstance(loaded, list):
        raise ValueError("few-shot body must contain a YAML list")
    return [FewShot.model_validate(item) for item in loaded]


class ConfigStore:
    """Config.yaml + config_overrides from the DB, with hot /set and a rebuilt Patterns."""

    def __init__(self, persona: Persona, db: Database) -> None:
        self._persona = persona
        self._db = db
        self._bot_username = ""
        self._overrides: dict[str, str] = {}
        self.current: Config = Config()
        self._base: Config = Config()
        self._patterns = _rebuild_patterns(self.current, self._bot_username, self._persona)
        # Serializes load/set/unset: without the lock two parallel /set calls (or /set
        # and a restart load) could read the same _overrides snapshot, both pass
        # validation, and one override would be lost when writing self.current.
        self._lock = asyncio.Lock()

    async def load(self) -> Config:
        async with self._lock:
            return await self._load_locked()

    async def _load_locked(self) -> Config:
        """Body of load() without taking self._lock, for set()/unset() which already
        hold it. asyncio.Lock is not reentrant: a nested load() inside set() would
        deadlock the coroutine."""
        overrides = await self._db.get_overrides()
        layers = self._persona.config_layers()
        cfg = load_config(layers, overrides)
        self._overrides = overrides
        self.current = cfg
        self._base = load_config(layers)
        self._patterns = _rebuild_patterns(cfg, self._bot_username, self._persona)
        return cfg

    def get(self) -> Config:
        return self.current

    def patterns(self) -> Patterns:
        return self._patterns

    def describe(self, key: str) -> KeyInfo | None:
        return describe_key(
            self.current, self._base, self._overrides, key, catalog=self._persona.catalog
        )

    def set_bot_username(self, username: str) -> None:
        self._bot_username = username
        self._patterns = _rebuild_patterns(self.current, self._bot_username, self._persona)

    async def set(
        self, key: str, raw_value: str, changed_by: int, now: int
    ) -> tuple[str | None, str]:
        async with self._lock:
            if key.startswith("persona."):
                raise OwnerError("set_persona")
            if not key.startswith(_ALLOWED_PREFIXES):
                raise OwnerError("set_unknown_section", key=key)

            candidate_overrides = dict(self._overrides)
            candidate_overrides[key] = raw_value
            try:
                load_config(self._persona.config_layers(), candidate_overrides)
            except ConfigError as exc:
                if exc.owner is not None:
                    raise exc.owner from exc
                raise ValueError(f"{key}: {_first_error_message(exc, key)}") from exc
            except ValueError as exc:
                raise ValueError(f"{key}: {_first_error_message(exc, key)}") from exc

            old = await self._db.set_override(key, raw_value, changed_by, now)
            await self._load_locked()
            new = flatten_config(self.current).get(key, raw_value)
            return (old, new)

    async def unset(self, key: str, changed_by: int, now: int) -> str | None:
        async with self._lock:
            await self._db.delete_override(key, changed_by, now)
            await self._load_locked()
            return flatten_config(self.current).get(key)

    def flat(self) -> list[tuple[str, str, bool]]:
        return [
            (key, value, key in self._overrides)
            for key, value in flatten_config(self.current).items()
        ]


SeedAction = Literal["keep", "seed"]


@dataclass(frozen=True)
class SeedPlan:
    """What ``PromptStore.load`` will do with the pack files against the stored versions.

    ``seed`` means a new version is created from the file (and becomes active); the deploy gate
    fails on it unless the owner asked for the change. The sha256 values identify the file text
    that was compared (``read_text`` form, as ``load`` sees it).
    """

    prompt: SeedAction
    few_shot: SeedAction
    prompt_sha256: str
    few_shot_sha256: str
    prompt_active_version: int | None
    few_shot_active_version: int | None


def decide_seed(file_text: str, stored_bodies: Sequence[str]) -> SeedAction:
    """Seed only when no version exists or the file (after strip) equals none of the stored ones."""
    if not stored_bodies:
        return "seed"
    return "keep" if file_text.strip() in {body.strip() for body in stored_bodies} else "seed"


def build_seed_plan(
    *,
    prompt_text: str,
    few_shot_text: str,
    prompt_bodies: Sequence[str],
    few_shot_bodies: Sequence[str],
    prompt_active_version: int | None,
    few_shot_active_version: int | None,
) -> SeedPlan:
    """Pure core shared by ``PromptStore.load`` and the read-only ``packcheck --seed-dry-run``."""
    return SeedPlan(
        prompt=decide_seed(prompt_text, prompt_bodies),
        few_shot=decide_seed(few_shot_text, few_shot_bodies),
        prompt_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        few_shot_sha256=hashlib.sha256(few_shot_text.encode("utf-8")).hexdigest(),
        prompt_active_version=prompt_active_version,
        few_shot_active_version=few_shot_active_version,
    )


class PromptStore:
    """Active system prompt and few-shot versions, seeded from files at startup."""

    def __init__(self, persona: Persona, db: Database) -> None:
        self._persona = persona
        self._db = db
        self._prompt_version = 0
        self._prompt_body = ""
        self._few_shot_version = 0
        self._few_shot_body_yaml = ""
        # Serializes load/rollback_prompt/add_example/remove_example: without the lock
        # two parallel /ex add calls could both read the same self.examples() and one
        # added example would be lost when writing the new few-shot version.
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Seed a version from the file only if it matches no saved version.

        The file body used to be compared only with the active DB version: after
        `/rollback 1` the file on disk (the v2 body) differs from the new active one (v1),
        and the next `load()` (e.g. on restart) would seed v3 from the file and thereby
        undo the rollback. Comparing against ALL saved versions (after strip) fixes this:
        if the file body equals any existing version, no new one is created and the one
        active in the DB stays active.
        """
        async with self._lock:
            await self._load_locked()

    async def plan_seed(self) -> SeedPlan:
        """Seed decision without writing: what load() would do (read-only for the DB)."""
        prompt_active = await self._db.active_prompt()
        few_shot_active = await self._db.active_few_shot()
        return build_seed_plan(
            prompt_text=self._persona.prompts["system"],
            few_shot_text=self._few_shot_file_text(),
            prompt_bodies=await self._db.prompt_version_bodies(),
            few_shot_bodies=await self._db.few_shot_version_bodies(),
            prompt_active_version=prompt_active[0] if prompt_active else None,
            few_shot_active_version=few_shot_active[0] if few_shot_active else None,
        )

    def _few_shot_file_text(self) -> str:
        return self._persona.few_shot_path.read_text(encoding="utf-8")

    async def _load_locked(self) -> None:
        now = int(time.time())
        plan = await self.plan_seed()

        file_prompt = self._persona.prompts["system"]
        if plan.prompt == "seed":
            version = await self._db.add_prompt_version(file_prompt, "seed from file", now)
            self._prompt_version = version
            self._prompt_body = file_prompt
        else:
            active_prompt = await self._db.active_prompt()
            assert active_prompt is not None  # versions exist -> an active one exists
            self._prompt_version, self._prompt_body = active_prompt

        file_few_shot = self._few_shot_file_text()
        if plan.few_shot == "seed":
            version = await self._db.add_few_shot_version(file_few_shot, "seed from file", now)
            self._few_shot_version = version
            self._few_shot_body_yaml = file_few_shot
        else:
            active_few_shot = await self._db.active_few_shot()
            assert active_few_shot is not None  # versions exist -> an active one exists
            self._few_shot_version, self._few_shot_body_yaml = active_few_shot

    def system_prompt(self) -> str:
        return self._prompt_body

    def prompt_version(self) -> int:
        return self._prompt_version

    def few_shot_text(self) -> str:
        return render_few_shot(self.examples())

    def few_shot_version(self) -> int:
        return self._few_shot_version

    async def rollback_prompt(self, version: int) -> bool:
        async with self._lock:
            ok = await self._db.activate_prompt(version)
            if ok:
                # Re-read the active version straight from the DB rather than through a
                # full load(): load() already creates no version on a rollback (it
                # compares the file with ALL saved versions, not only the active one),
                # but there is no need to read the file from disk just to switch to an
                # already known version body.
                active_prompt = await self._db.active_prompt()
                if active_prompt is not None:
                    self._prompt_version, self._prompt_body = active_prompt
            return ok

    async def add_example(self, name: str, user: str, text: str, now: int) -> int:
        async with self._lock:
            examples = self.examples()
            examples.append(FewShot(name=name, user=user, speak=True, text=text))
            body_yaml = yaml.safe_dump(
                [item.model_dump(mode="json") for item in examples],
                allow_unicode=True,
                sort_keys=False,
            )
            version = await self._db.add_few_shot_version(body_yaml, "/ex add by owner", now)
            self._few_shot_version = version
            self._few_shot_body_yaml = body_yaml
            return version

    async def remove_example(self, index: int, now: int) -> int:
        async with self._lock:
            return await self._remove_example_locked(index, now)

    async def _remove_example_locked(self, index: int, now: int) -> int:
        examples = self.examples()
        if index < 1 or index > len(examples):
            raise OwnerError("ex_rm_range", index=index, total=len(examples))
        del examples[-index]
        body_yaml = yaml.safe_dump(
            [item.model_dump(mode="json") for item in examples],
            allow_unicode=True,
            sort_keys=False,
        )
        version = await self._db.add_few_shot_version(body_yaml, "/ex rm by owner", now)
        self._few_shot_version = version
        self._few_shot_body_yaml = body_yaml
        return version

    def examples(self) -> list[FewShot]:
        return _parse_few_shot_yaml(self._few_shot_body_yaml)
