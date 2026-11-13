# sotto: working contract for agents

A Telegram bot that plays one character in one group chat. The engine (this repository, English)
is language- and persona-neutral; everything that makes it a particular character lives in a
**persona pack** (a directory, see "The persona pack"). This file is the contract of the code:
structure, interfaces, invariants, conventions. It outranks habit. Rationale for the design is in
`docs/history/original-plan.md` (the original staged plan) and `docs/plans/`; how to write a
persona is in `docs/creating-a-persona.md`; operations are in `RUNBOOK.md`.

Rules that came out of owner decisions are stated here as plain rules. The history of how they
came about is in `docs/history/pull-requests.md` and is not repeated.

## Stack and tools

- Python 3.12, `uv` for dependencies and running (`uv sync`, `uv run ...`).
- aiogram 3.x, aiosqlite, httpx, pydantic v2, pydantic-settings, PyYAML. No frameworks on top of
  aiogram, no ORM: SQL is written by hand in `db.py`.
- Dev: pytest, pytest-asyncio (`asyncio_mode = auto`), pytest-cov, ruff (lint + format), mypy (strict).
- Before adding a dependency check the current version on PyPI and pin `>=major.minor`.
- Green before handing anything in: `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy src tools`, `uv run pytest -q`, the coverage gate and the golden compares (see "Testing").

## Repository layout

```
pyproject.toml, uv.lock, Dockerfile, docker-compose.yml, docker-compose.demo.yml
.env.example           # every key, empty values
README.md, RUNBOOK.md, CHANGELOG.md (engine, English), CLAUDE.md
docs/creating-a-persona.md   docs/history/   docs/plans/
locales/en/*.yaml      # owner-UI catalog (English), shipped in the image
personas/demo/         # the English demo pack ("Gus Marlow"), public, default for tests
personas/_legacy_ru/   # synthetic Russian pack for the migration proofs; removed in PR-9
src/sotto/             # the engine (below)
  lang/                # Language modules: base.py (protocol), en.py, ru.py
tools/                 # dev tools: golden/, coverage_gate.py, check_language.py, extract_*.py,
                       # inventory.py, split_changelog.py
tests/                 # generic tests on the demo pack, English data
  lang/                # Russian-language behaviour, on tests/fixtures/ru_pack/ (synthetic)
  legacy/              # migration proofs against personas/_legacy_ru and the golden baseline
  responder/           # the Responder tests, one file per area
  fixtures/ru_pack/    # synthetic Russian persona pack for tests/lang
deploy/                # deploy.sh, deploy-only.sh, setup-host.sh (server side, see RUNBOOK)
data/                  # bot.db at run time; git-ignored, volume in compose
exports/               # Telegram Desktop exports; git-ignored
```

Engine modules (`src/sotto/`), one task each:

| Area | Modules |
|---|---|
| Process | `app.py` (assembly, `main()`), `__main__.py`, `settings.py`, `wiring.py` (shared by `app` and `packcheck --boot`) |
| Pack | `persona.py`, `fragments.py`, `i18n.py`, `lang/`, `packcheck.py`, `stores.py` (`ConfigStore`, `PromptStore`) |
| Config | `config_models.py` (pydantic models, ranges, descriptions), `config.py` (`load_config`, `describe_key`) |
| Telegram | `bot.py` (message handler), `commands.py` (owner/chat commands), `reactions.py`, `vision.py` |
| Decision | `gate.py`, `gate_state.py`, `gate_types.py`, `patterns.py`, `timeutil.py`, `delays.py`, `motifs.py` |
| Generation | `responder.py`, `prompt.py`, `llm.py`, `postprocess.py`, `filters.py`, `judge.py`, `followup.py`, `stickers.py`, `places.py`, `weather.py`, `vocab.py`, `jokes.py`, `diary.py`, `chat_memory.py` |
| Storage | `db.py`, `schema.sql`, `retention.py`, `sanitize.py` |
| Operations | `alerts.py`, `changelog.py`, `errors.py` |
| Offline CLIs | `replay.py`, `export_parser.py`, `places_fill.py`, `stickers_fill.py`, `packcheck.py` |

## The persona pack

The pack is chosen at run time by `PERSONA_DIR` (a path). There is no module-level "active pack":
a loaded `Persona` is passed explicitly to everything that needs it. Without `PERSONA_DIR` the
process refuses to start (fail-closed: a lost mount must not start another persona in the real
chat); `ALLOW_DEMO_PERSONA=true` opts into `personas/demo`. The image sets neither;
`docker-compose.demo.yml` sets `PERSONA_DIR=/app/personas/demo`.

```
pack.yaml          # manifest: format (1), name, language, locale (default = language), ship (allowlist)
config.yaml        # persona + every behaviour/filter/llm/places number; one file
few_shot.yaml      # examples, rendered into the {few_shot} slot
fragments.yaml     # the wording the engine puts into prompts / persists (strict schema)
prompts/*.txt      # 13 prompts, fixed names (below)
stickers.yaml, places_manual.yaml, vocab.example.yaml   # data
locale/<code>/*.yaml   # optional owner-UI texts; missing keys fall back to locales/en
chat_notes.md      # release notes for the chat, `## [X.Y.Z] - DATE` then text (for /changelog)
.revision          # written by the deploy sync; "dev" when absent
golden/            # scenarios + baseline + ALLOWED_DIFFS.md (dev only, never shipped)
```

```python
# persona.py
class Manifest(BaseModel):                 # extra="forbid": format:int, name, language, locale:str|None, ship:list[str]
class Persona:                              # frozen dataclass
    root, manifest, config_path, few_shot_path, stickers_path, places_manual_path: Path
    prompts: Mapping[str, str]              # the 13 bodies by file stem, read once (utf-8)
    chat_notes: str; legacy_changelog: str  # legacy_changelog: transition source when chat_notes.md is absent
    frag: Fragments; lang: Language; catalog: Catalog; revision: str
    def config_layers(self) -> list[Path | Mapping]   # [lang default layer, config.yaml]
    def prompt_errors(self) / fragment_errors(self) -> list[str]
def load_persona(path: Path) -> Persona     # raises PackError (missing files, bad manifest, unknown language)
def resolve_persona_dir(persona_dir, allow_demo) -> Path
def persona_from_env() -> Persona           # for CLIs that need no BOT_TOKEN (PersonaSelector)
```

Wiring: `ConfigStore(persona, db)`, `PromptStore(persona, db)`, `Responder(persona=..., frag=...)`,
`Deps(persona=...)`; every checker/extractor (`Judge`, `FollowupChecker`, `StickerChooser`,
`ReactionChooser`, `VisionDescriber`, `WeatherPlaceExtractor`, `DiaryExtractor`, `ChatMemorizer`) gets
its template from `persona.prompts[...]`. Everything that needs an API key is built in
`wiring.build_llm_stack` only, so `packcheck --boot` exercises exactly what the process does.

**Config layering** (lowest first): engine pydantic defaults < `lang.default_filters()` < pack
`config.yaml` < DB overrides (`config_overrides`, written by `/set`). `load_config(layers, overrides)`
deep-merges layers (lists and scalars of the upper layer replace the lower), applies the flat
`overrides` (`"behaviour.daily_cap" -> "3"`, parsed by pydantic) and validates.
Engine defaults for persona data are neutral (name `"Persona"`, empty trigger lists, `UTC`); numeric
tuning defaults and their `ge/le` ranges stay in the engine because `/set` validates against them.
`/set` may change only keys under `behaviour.`, `llm.`, `places.`, `filters.`; `persona.*` changes in
the pack's yaml.

**Fragments** (`fragments.yaml`, `fragments.py`): groups `placeholders`, `situations`, `prompt`,
`render`, `weather`, `vocab`, `tools`. `FRAGMENT_SPEC` maps each dotted leaf key to the exact set of
`{placeholders}` it must contain; `LITERAL_BRACES` leaves (JSON answer formats) may contain other
braces; `FRAGMENT_LISTS`/`FRAGMENT_MAPS` hold the weekday list and the `accusative` / `weather.wmo`
maps. Strings that are persisted (`placeholders.*`: photo, sticker, participant, sticker tag, ...) must
not change once a database exists: every consumer derives its prefix/regex from them. Substitution is
`str.replace` or one `re.sub` pass, never `str.format` (security property, see "Invariants").

**Prompt slots** (`fragments.PROMPT_SLOTS`: prompt stem -> (required, allowed)); a slot outside
`allowed` stays literal in the rendered text, a missing required slot is an error; both are
`packcheck` errors:

| Prompt | Required | Allowed |
|---|---|---|
| `system` | `age few_shot context situation` | `age few_shot life diary vocab chat_memory jokes weather context recent_replies places avoid situation` |
| `judge` | `trigger candidate` | same |
| `sticker` | `menu trigger reply` | same |
| `followup`, `name_check` | `context recent_replies name text` | same |
| `checkin_prefilter` | `messages recent_replies names` | same |
| `reaction` | `context name text emoji reply_to` | same |
| `memory`, `memory_threads` | `period chat` | `period chat limit` |
| `memory_jokes` | `period chat limit` | same |
| `diary` | `reply` | same |
| `vision` | `caption` | same |
| `weather_place` | `text` | same |

In `system`, `{context}`, `{recent_replies}` and `{places}` are replaced by a "see below" marker and
the data goes in the user message; the others (`age few_shot life diary vocab chat_memory jokes
weather avoid situation`) are owner-curated or engine-built text and stay in `system`.

**Language modules** (`lang/base.py` protocol, `lang/en.py`, `lang/ru.py`, `get_language(code)`): only
what is linguistic: `stopwords`, `foreign_letters` (None switches the foreign-script checks off),
`reflexive_suffixes`, `plural(n)`, `fold(text)`, `name_pattern(word)` / `name_forms(word)`,
`vocab_rules(word)` / `vocab_prefix_tails(token)` / `vocab_name_stem(word)`, `fact_re()`,
`default_filters()`. Persona data (motif declension, district aliases, the character's foreign words)
is not language, it is pack. `en.py` is deliberately small (an `'s` name suffix, plain `-s/-ed/-ing`
vocabulary rules, no foreign-alphabet venue filter). Cyrillic is allowed only in `lang/ru.py`,
`tests/lang/`, `tests/fixtures/ru_pack/` and `personas/_legacy_ru/`.

**Owner-UI catalog** (`i18n.py`): `locales/en/{help,status,commands,alerts,errors,validation,config_keys}.yaml`,
top-level key = file stem; values are strings or plural maps (`one/few/many/other`); placeholders
`{name}` are validated at load. `Catalog.t(key, **params)`, `Catalog.plural(key, n, **params)`; a key
missing in the pack's locale falls back to `en` with one WARNING and never crashes a command.
`config_keys.yaml` gives `/get <key>` descriptions per locale (else `Field.description`). Owner-facing
errors are `OwnerError(code, **params)` (a `ValueError`), rendered as `catalog.t("errors."+code)`.

**`packcheck`** (`python -m sotto.packcheck DIR [--boot] [--locale-parity] [--db PATH --seed-dry-run]
[--allow-seed prompt|few_shot]`): prints file names and counts, never prompt or config text. Checks the
pack loads, every `ship` entry exists and covers the required files, prompt slots, fragments schema,
`load_config(layers)`, few-shot/stickers/places parse, catalog has no missing keys; `--boot` builds the
whole wiring against a temp DB and a dummy key without network; `--seed-dry-run` opens the DB
read-only and fails (exit 3) if starting would create a new prompt or few-shot version. Exit codes:
0 fine, 1 invalid, 2 usage, 3 seed would change.

**Prompt/few-shot versions** (`PromptStore.load`): the file body is compared (after `strip`) with the
active version in the DB; no versions -> v1 from the file; different -> new version with note
`seed from file`; otherwise the DB wins. After that the DB is the source of truth (`/rollback`,
`/ex add`). Never add a slot to `system.txt` casually: it creates a new version at every deploy.

## Settings (`settings.py`)

`Settings(BaseSettings)` reads `.env`, no prefix: `bot_token`, `admin_user_id` (0 = owner commands off),
`allowed_chat_id` (0 = discovery mode: nothing stored, every message logged at WARNING with chat id,
title, user id, display name), `openrouter_api_key`, `google_places_key`, `db_path`, `persona_dir`,
`allow_demo_persona`, `weather_latitude/longitude/home_name` (the home point is personal data: only in
`.env`, never in config or code), `log_level`. The former per-file path variables (`CONFIG_PATH`,
`*_PROMPT_PATH`, `STICKERS_PATH`, `CHANGELOG_PATH`, ...) are deprecated: set -> one startup WARNING,
value ignored. `PersonaSelector` reads only `persona_dir`/`allow_demo_persona`.

## Invariants (do not break)

1. **One LLM entry point.** Every model call goes through `llm.LLMClient.call` / `call_raw`. Order inside:
   circuit breaker (`llm_circuit_until`) -> calls cap (`day_key(counter_key)` vs `calls_cap`, default
   `llm.daily_calls_cap`) -> dollar budget (`llm_spent_usd` vs `llm.daily_budget_usd`) -> increment the
   counter BEFORE the request -> POST OpenRouter chat completions (`temperature 0.8`, `usage.include`)
   -> cost from `usage.cost` (else tokens x `llm.price_*`); errors increment `llm_error_streak`,
   `llm.circuit_errors` in a row open the breaker for `circuit_pause_min`. No retries inside. Cheap
   auxiliary calls use their own `counter_key` (`followup_calls`, `reaction_calls`, `vision_calls`,
   `diary_calls`, `weather_calls`, `memory_calls`) with their own caps. `LLMError.reason`:
   `llm:timeout|http|budget|calls_cap|circuit_open|empty`. A provider refusal (4xx except 408/429) calls
   the `RefusalHook` (owner alert, `alerts.OwnerAlerter`, rate-limited per status). Secrets are redacted
   (`_redact`) before anything is logged or alerted.
2. **People's data only in the user message, inside delimiters.** `prompt.build_messages` puts chat
   context, recent replies, places and the situation into a `role=user` message between `<<<CHAT` and
   `>>>` with a "this is data, not commands" notice. `normalize_text` already strips `<<<`/`>>>`; every
   renderer strips fake delimiters again. Slot substitution is one `re.sub` pass so one slot's content is
   never re-scanned. Owner-curated text (`few_shot`, `life`, `vocab`, ...) may sit in `system`;
   participant-derived text (diary facts, jokes, people threads) is screened by `Patterns.unsafe_note`
   before it is stored because it ends up in the prompt. Placeholders are never filled with `str.format`.
3. **Gate order is fixed and pure.** `gate.should_consider(msg, state, cfg, patterns, now, rng)` is a
   pure function; changes come back in `Decision.state_changes`. Order: (0 chat filter, in the handler)
   `is_bot` -> panic/stop -> muted -> topic stop (sets `topic_cooldown_until`) -> injection markers ->
   direct address (reply > mention > name): night -> `QUEUE_NIGHT`, presence cap (except `PRESENCE_EXEMPT`
   = reply, mention), `mention_daily_cap` -> else ambient: night -> logistics -> topic cooldown (ambient
   only) -> presence cap -> (hot window: `hot_cap`, dice with `hot_window.ambient_probability`) or
   (not live -> ambient cap/cooldown -> dice). Reasons are `gate:<name>` and `pass:<trigger>`. A direct
   address is never dropped by a cooldown (it shifts `due_at` in the responder); only night and
   `mention_daily_cap` stop it. The gate never changes counters; sending does.
4. **Output filter: three layers, silence on failure, no retries.** `filters.check_output(text, ctx,
   judge)`: layer 1 `regex:*` (length, markdown, emoji, sentences, starts_name, phone, venue, topic,
   muted_name, echo, prompt_leak, model_talk, latin), layer 2 `dedup:*`/`style:*` (jaccard,
   polish_freq, self_echo, motif, vocab, assistant, grumpy, question_x2, exclaim, emoji_count,
   emoji_freq, story_quota), layer 3 `judge:*` (only when layers 1-2 pass or shadow is on). `FilterVerdict`
   carries `reason` (first) and `reasons` (all). **Shadow vs enforce:** with `filters.shadow: true` a cut is
   logged (`filter_log`, `shadow=1`) but the reply is sent, EXCEPT stages in `filters.enforce_stages`
   (default `dedup`, `style`), which cut anyway. Cosmetic deterministic corrections (typographic dashes,
   emoji frequency/count) are made by `postprocess.soften` BEFORE the filter (`fix:*` rows in
   `filter_log`), not cut by it. A cut means silence; the main model is never called again.
5. **Responder is the only sender; one respond lock.** `Responder` sends everything to the chat.
   Generation + send + counters run under one `asyncio.Lock`. Direct addresses become `pending_replies`
   rows with a timer (collapse: a new trigger within a pending one updates its `due_at` and appends to
   its item list, never a second task); `_fire_pending` re-checks the deterministic gate steps (not dice,
   not live) and moves the reply to `night_queue` inside the quiet window. `mark_pending_done` happens
   only after a final outcome (a send or any `filter_log` verdict), never before the model call, so a
   SIGTERM mid-call does not lose the reply (`restore_pending` re-arms on start; older than
   `late_reply_threshold_sec` -> closed with `send:restart`). Background jobs (`morning_job`,
   `spontaneous_job`, `checkin_job`, `callback_job`, retention, chat memory) wrap each iteration:
   `CancelledError` propagates, other exceptions are logged and the loop lives on.
6. **Retention and privacy.** Messages live `behaviour.message_retention_days` (30); `purge_older_than`
   also trims `night_queue`, `pending_replies`, nulls `filter_log.candidate_text`, and deletes dated state
   keys (`:YYYY-MM-DD`, `:YYYY-Www`, day boundaries in `persona.timezone`). Long-lived memory
   (`chat_memory`, `chat_jokes`, `people_threads`, `self_facts`) has its own keep periods and is purged in
   `run_retention`; muted participants never enter summaries. Photos are described by a vision model and
   not stored (bytes and base64 are never logged). No chat content, names or places of real people
   appear in this repository: tests and docs use the synthetic demo.
7. **Commands in the chat do not enter the gate or `messages`.** The commands router is included first and
   stops propagation. Chat commands (all participants): `/stop` (24 h silence, silent, audited), `/mute`,
   `/unmute` (self; as a reply to another person owner only), `/ex add` (owner). Everything else is
   owner-only in a private chat (below). Replies are plain text (no parse mode), at most 3500 chars.
   Anyone else's command in a private chat is ignored silently.
8. **Time:** all times are unix seconds (`int`); a time zone appears only when displaying and when
   computing "days" through `persona.timezone` (`timeutil.local_date/day_key/week_key/in_window/
   seconds_until/day_start`).
9. **Triggers are strings in logs and the DB:** `Trigger` = `mention reply name followup ambient`; the
   responder adds `places` (a renamed address), `morning`, `spontaneous`, `checkin`, `callback`, `life`,
   `say`. `life`, `morning`, `checkin`, `callback` do not touch the mention/ambient counters.

## Interfaces by module

Signatures an agent needs; read the code for the rest.

**Storage.** `Database(path)`: `connect()` (WAL, `foreign_keys=ON`, `migrate()`), `close()`,
`migrate()` (`schema.sql` when `user_version == 0`, numbered `MIGRATIONS[n]` after; current
`user_version` 6). One aiosqlite connection per process; every writing method takes an internal
`asyncio.Lock`, multi-statement operations run entirely under it (callers never think about
serialization). Tables: `messages`, `bot_replies`, `state`, `night_queue`, `pending_replies`,
`muted_users`, `config_overrides`, `config_audit`, `places`, `filter_log`, `prompt_versions`,
`few_shot_versions`, `life_events`, `chat_memory`, `self_facts`, `chat_jokes`, `people_threads`,
`vocab_dicts`. Row dataclasses (`MessageRow`, `BotReplyRow`, `PendingRow`, `PlaceRow`, ...) live in
`db.py`. Key methods: `insert_message`, `recent_messages(chat_id, limit)` (returns asc),
`get_state/set_state/delete_state`, `increment_state`, `add_state_float`, `apply_state_changes`,
`insert_filter_log(trigger_tg_message_id, candidate_text, verdict, stage, reason, shadow, created_at)`,
`filter_log_summary(since)`, `set_override/delete_override` (audited), pending/night/mute/version/memory
CRUD. A new migration is a new `MIGRATIONS[n]` entry plus the same DDL in `schema.sql` and a bumped
`user_version`.

**Sanitizing** (`sanitize.py`, pure): `sanitize_display_name(raw, user_id, reserved)` (letters of any
script, hyphen, spaces; max 24 by word boundary; empty, equal to, or containing a reserved name as a
whole word -> the `placeholders.participant` fragment with `stable_n(user_id)`; part of a word does not
count); `normalize_text`; `media_placeholder(message, frag=)`. A display name matching
`Patterns.injection` is replaced the same way before it is stored.

**Patterns** (`patterns.py`): `Patterns(filters, name_triggers, bot_username, lang)`, compiled once,
pure methods returning the matched pattern string or None/bool: `topic_stop`, `injection`, `logistics`,
`name_trigger` (via `lang.name_pattern`), `mentions_bot`, `urgent`, `places_request`, `weather_request`,
`model_talk`, `assistant_marker`, `unsafe_note`, plus `motifs`, `story_markers`, `self_markers`. All
regexes are compiled `IGNORECASE`; an empty list means "never matches". `ConfigStore` rebuilds `Patterns`
on every load; `Deps.patterns_getter` is the getter, never a frozen object.

**Handler** (`bot.py`, `build_router(deps)`): chat filter -> discovery mode -> sanitize/normalize (photo:
`vision.should_describe` -> describe) -> `insert_message` -> `GateMessage` + `load_gate_state` ->
`should_consider` -> `apply_state_changes` -> DROP: `FOLLOWUP_REASONS` in the hot window go to the cheap
`FollowupChecker.check` (yes -> `Trigger.FOLLOWUP`), else `filter_log` + a delayed reaction
(`REACT_REASONS`); QUEUE_NIGHT -> `enqueue_night`; PASS -> for `Trigger.NAME` the cheap `check_name`
(no -> `followup:name_no`, reaction) then `responder.on_gate_pass`. Vocabulary explain requests
(`explain_requested`) turn a DROP/ambient PASS into FOLLOWUP.

**Responder** (`responder.py`): `Responder(bot, db, cfg_getter, llm, judge, sticker_chooser, followup,
weather, weather_places, diary, vocab_getter, patterns_getter, prompt_store, rng, chat_id, bot_user_id,
clock, sleep, persona, frag)`. Entry points: `on_gate_pass`, `restore_pending`, `morning_job`,
`spontaneous_job`, `checkin_job`, `callback_job`, `announce_life(event)`, `say(text)`, `preview(text,
display_name)` (the `/ask` dry run: same prompt build, model, `soften` and filter, writes and sends
nothing), `shutdown()`. `_generate_and_send` returns a `SendOutcome(sent, text, reason, tg_message_id)`.
Per reply: fresh context -> recent replies -> `build_messages` -> `llm.call(main_model)` -> `parse_reply`
(`{"speak": bool, "text": str, "reply_to"?: int|null}`; failure -> `llm:invalid_json`; `speak: false` ->
`llm:silent`) -> `soften` -> `check_output` -> sticker choice -> typing loop (`sendChatAction` every 4 s)
-> `send_message` -> `insert_bot_reply` -> counters -> optional hot window -> background diary extraction.
Rules in force: presence cap (`behaviour.presence`: allowance = `floor(human_messages_today *
max_share) + free_replies`; reply and mention exempt); `behaviour.min_gap_sec` between bot messages
(addresses are delayed, ambient/checkin/morning dropped, `life`/`say` exempt); the hot window after
`/life`/`/say` (and after any reply if `hot_window.open_on_any_reply`) raises ambient odds and caps
address delay; the check-in ("came back to look") runs only in quiet, answers only explicit addresses,
and a cheap prefilter precedes the main model; `callback` asks a person about an open `people_threads`
story once; provider failures (`llm:http`/`timeout`) re-arm an address pending once (`send:llm_retry`).
The places menu goes into the prompt for direct addresses only; the weather background for every
trigger, plus a second block when a question names another place (cheap extractor + geocoder).

**Prompt** (`prompt.py`): `build_messages(template, *, age, few_shot, context, recent_replies, places,
situation, avoid, life, diary, chat_memory, jokes, weather, vocab, json_reminder, frag)`,
`render_context`, the `situation_*` builders, `parse_reply`, `Reply(speak, text, reply_to)`. Wording
comes from `frag`, never from literals.

**Other generation helpers** (each a class with a prompt from the pack and a cheap model, failing closed
to "no"/None on `LLMError` or bad JSON): `Judge.check` (layer 3), `FollowupChecker` (`check`,
`check_name`, `check_batch`), `StickerChooser.choose`, `ReactionChooser`/`ReactionScheduler`
(`pick_reaction` is the deterministic pre-filter; delay `reactions.delay_sec`; palette without repeats;
only emoji from `TELEGRAM_REACTIONS`), `VisionDescriber`, `DiaryExtractor`, `ChatMemorizer`
(`summarize_period`, `run_due`; jokes and people threads come from two extra calls), `WeatherClient`
(Open-Meteo, cached, falls back to the stale cache) and `WeatherPlaceExtractor`, `Vocab`/`VocabStore`
(dictionaries in the DB, matching by `lang.vocab_rules`, offer/request/explain logic),
`places.render_places_menu`, `motifs` (tracking of recurring props and stories, the `{avoid}` slot).

**Config** (`config.py`): `load_config`, `flatten_config`, `describe_key(current, base, overrides, key,
catalog)` -> `KeyInfo(key, value, default, overridden, type_name, bounds, description, settable)`.
`config_models.py`: every field has `Field(description=...)` (English, one line) and numeric fields have
`ge/le`; sections: `persona`, `behaviour` (with `live_talk`, `spontaneous`, `reactions`, `stickers`,
`hot_window`, `followup`, `checkin`, `presence`, `chat_memory`, `diary`, `jokes`, `vocab`, `callback`,
`vision`, `weather`), `llm`, `places`, `filters`. A `Config` validator checks that reaction emoji are a
subset of `TELEGRAM_REACTIONS`.

**Stores** (`stores.py`): `ConfigStore(persona, db)`: `load`, `get`, `patterns`, `describe`, `flat`,
`set(key, raw, changed_by, now) -> (old, new)`, `unset`; `PromptStore(persona, db)`: `load`,
`system_prompt`, `prompt_version`, `few_shot_text`, `few_shot_version`, `rollback_prompt`,
`add_example`, `remove_example`, `examples`, `build_seed_plan` (shared with `packcheck`).

**Owner commands** (`commands.py`, `build_commands_router(deps)`; texts come from the catalog): `/panic`,
`/resume`, `/status` (first line `sotto v<version>`, then pack name and revision, counters), `/last [n]`,
`/why [hours]`, `/get [key|prefix]`, `/set k v`, `/unset k`, `/prompt [full]`, `/rollback N`,
`/ex last|rm|add`, `/life <text>|list|rm N|post N` (remember and announce news; held only by panic and
stop), `/say <text>` (verbatim, no model, not remembered), `/ask <text>` (preview), `/memory
[list|rm N|run]`, `/diary [rm N]`, `/jokes [rm N]`, `/people [rm N]`, `/words [show|on|off|rm name]`
(upload = a YAML document captioned `/words`), `/changelog [owner|<version>]`, `/help`. State-changing
commands write `config_audit`.

**Changelog** (`changelog.py`): `/changelog` shows the latest entry of the pack's `chat_notes.md`;
`/changelog owner` and `/changelog <ver>` read the engine `CHANGELOG.md` (Keep a Changelog, English,
one notes block per version).

**Offline CLIs:** `python -m sotto.replay EXPORT.json [--persona DIR] [--config FILE] [--seed N]
[--bot-username X] [--bot-user-id N] [--verbose] [--generate [--judge] [--max-calls N]]` runs the gate
over a Telegram Desktop export with in-memory state (a PASS counts as sent) and prints a per-day table;
`python -m sotto.places_fill [--dry-run] [--no-llm] [--manual-only] [--queries ...]` fills `places` from
Google Places (reviews are compressed to a validated `fact` and never stored; places not seen are marked
`operational=0` except `manual:*`); `python -m sotto.stickers_fill SET [--out] [--dry-run] [--no-llm]
[--model]` builds `stickers.yaml` with a vision model, merging by `file_id`. Check every generated
`fact`/sticker text by hand: it is somebody else's text from the internet.

## Testing

- Tests never touch the network or need tokens. The DB in tests is a temp file or `:memory:`.
- Default pack for tests is `personas/demo` (`tests/conftest.py` sets `PERSONA_DIR`; fixtures
  `demo_pack`, `ru_pack`, `legacy_pack`; helpers in `tests/demo.py`, `tests/packs.py`). English data
  only, except `tests/lang/` (Russian behaviour on `tests/fixtures/ru_pack/`) and `tests/legacy/`
  (migration proofs, which die with `personas/_legacy_ru`).
- One test file per module (`test_<module>.py`); `tests/responder/` holds the responder tests by area.
- **Golden harness** (`tools/golden/`, see its README): records what the engine sends to models, how it
  parses and persists, the built config and the owner UI, deterministically (virtual clock, seeded rng,
  mocked HTTP, fake bot, real `Database`); `compare` diffs against a baseline. Two sets, each must
  print `golden compare OK: 101 scenarios identical`:
  ```
  uv run python -m tools.golden compare --pack personas/_legacy_ru --scenarios tools/golden/scenarios_ru.yaml \
      --baseline tools/golden/baseline_synthetic --allow tools/golden/ALLOWED_DIFFS.md
  uv run python -m tools.golden compare --pack personas/demo --scenarios personas/demo/golden/scenarios.yaml \
      --baseline personas/demo/golden/baseline --allow personas/demo/golden/ALLOWED_DIFFS.md
  ```
  Never re-record `tools/golden/baseline_synthetic`. The demo baseline may be re-recorded only for an
  intended change, listed in the pack's `ALLOWED_DIFFS.md` with the reason. `tools/golden/adapter.py` is
  the only module that knows how to build the engine; a refactoring updates it, not the scenarios.
- **Coverage gate:** `uv run pytest -q --cov --cov-report=json && uv run python -m tools.coverage_gate`
  (total >= 93 % over `src/`, >= 95 % per module in `NEW_MODULES`). Delete `coverage.json` and
  `.coverage` afterwards.
- **Language gate:** `uv run python -m tools.check_language` reports Cyrillic outside the allowed paths
  (report only until PR-9).
- Tests assert on behaviour, not on log text, except where the log line is the contract.

## Conventions

- Logs go to stdout, format `%(asctime)s %(levelname)s %(name)s %(message)s`, level from
  `Settings.log_level`. Secrets never reach a log.
- Async wherever there is I/O. Pure functions are sync and free of aiogram, so tests stay simple.
- One module, one task. Do not touch another module unless the contract here requires it.
- No `print`, no bare `except`, no `# type: ignore` without a reason in a comment.
- Everything public is English: code, comments, docstrings, log and exception texts, docs, commit
  messages. Owner-facing texts of the engine live in `locales/en`; wording that goes into prompts lives in
  the pack's `fragments.yaml`, never in code (`tests/test_no_hardcoded_wording.py` guards it).
- Prompts and few-shot are loaded from the pack, never embedded in code.
- Secrets only through `.env`; `.env.example` lists every key with an empty value.
- `ruff` excludes `*.md`, so Python blocks in documentation are not formatted.

## Releases and the changelog

SemVer with major 0 (minor for a feature, patch for a behaviour fix or bug fix); the single source is
`pyproject.toml`, and `sotto.__version__` reads it via `importlib.metadata` (fallback `0.0.0`). Every PR
that changes behaviour extends `## [Unreleased]` in `CHANGELOG.md` (engine, owner/technical notes,
English) and, when something is visible in the chat, the pack's `chat_notes.md`. Release: bump `version`,
`uv lock`, move Unreleased to `## [X.Y.Z] - YYYY-MM-DD`, update the compare links at the bottom
(`https://github.com/tslava/sotto/compare/...`), merge, tag `vX.Y.Z` on the squash commit. The bot never
posts release notes to the chat (it would break the character); the owner does. Docker images are tagged
`sha-...`; the version is not tied to the deploy.

## What not to do

- Do not send to the chat from anywhere but `Responder`; do not call a model from anywhere but `LLMClient`.
- Do not add a web server, webhooks or open ports (long polling only).
- Do not keep messages longer than `message_retention_days`.
- Do not embed a prompt, few-shot or persona wording in code; do not add persona-specific defaults to
  `config_models.py`.
- Do not put real chat data, real people, real cities or a real persona pack into this repository
  (tests and docs use the invented demo); keep the home coordinates in `.env` only.
- Do not re-record a golden baseline to hide a diff; do not change `PROMPT_SLOTS`/`FRAGMENT_SPEC` without
  updating every pack and the `packcheck` tests.
- Do not bake secrets into the image or log them; do not edit the production DB while the process runs.
