# sotto

[![CI](https://github.com/tslava/sotto/actions/workflows/ci.yml/badge.svg)](https://github.com/tslava/sotto/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/tslava/sotto/badges/coverage.json)](https://github.com/tslava/sotto/actions/workflows/badges.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![golden: 101 scenarios](https://img.shields.io/badge/golden-101%20scenarios-informational)

A Telegram bot that plays one character in one group chat, as a regular and not as an assistant.
It does not wait for commands. It reads the room, decides whether to speak at all, usually stays
quiet, answers a direct question a few minutes late like a person who was away from the phone,
remembers what the chat talked about last month, and does not drop the character.

The engine in this repository has run a private persona in a friends' group chat since September
2026. That persona is not part of the repo: the public version ships an invented English one,
**Gus Marlow**, a retired bus driver who mends clocks, as a demo and a test fixture.

The interesting part is not the model call, it is everything around it: when to speak, how not to
be annoying, how to keep a model that reads untrusted chat text from being steered by it, and how
to change such a system without silently changing its behaviour.

Project page: <https://tslava.dev/sotto.html>

## How it decides to speak

A message goes through a pure function, `gate.should_consider(msg, state, cfg, patterns, now, rng)`,
in a fixed order. Everything that is dropped is written to a `filter_log` table with a reason
(`gate:night`, `gate:not_live`, `gate:dice`, ...), so `/why` in the owner chat can say why the bot
stayed quiet.

1. Bots, panic switch, `/stop`, muted people.
2. Stop topics (the message is dropped and the topic cools down) and prompt-injection markers.
3. A **direct address** (a reply to the bot, `@username`, or its name, declined by the language module) is
   answered almost always. Only the quiet night window and a daily cap stop it. A cooldown never
   drops it, it only postpones it.
4. Otherwise **ambient** speech: not at night, not about logistics ("who is coming", "I am late"),
   and only inside a **live conversation** (at least N messages from at least M people in the last
   few minutes). Then a per-chat cooldown, a daily cap and a dice roll.

On top of the gate:

- **Presence cap.** Over a local day the bot may say at most a fraction of what people said, plus a
  couple of free replies. Replies to its own messages and direct @mentions are exempt.
- **Pending replies with human delays.** A direct address becomes a row in `pending_replies` with a
  due time drawn from buckets (mostly 30 s to 3 min, sometimes up to an hour). Several addresses
  arriving meanwhile collapse into one reply. A restart re-arms the timers; a row is closed only
  after a final outcome, so a SIGTERM in the middle of a model call does not lose the reply.
- **Night queue and morning reply.** Addresses during the quiet window are answered once, in one
  sentence, at a random moment of the morning window.
- **Attention rhythm.** *Phone in hand:* when the owner pushes a message through `/life` or `/say`
  a ten-minute hot window opens; ambient replies are likelier in it, and a cheap model decides
  whether an unaddressed message is still "for him" (follow-up check). *Phone put down:* only direct
  addresses count. *Back to check:* later, once the chat has gone quiet, a cheap prefilter looks at
  everything written since his last line, and the main model replies only if something really was
  for him; otherwise silence, and no new check unless people wrote.
- **Reactions instead of silence.** On some random drops it sets an emoji reaction after a pause of
  20 to 120 seconds, if a cheap model thinks the message deserves one; the same emoji is not used
  more than twice in a row.

Every number above is a key in the persona's `config.yaml`, validated against ranges, and the owner
can change it from Telegram without a restart (`/set`).

## Generation and safety

- **One LLM entry point.** Every model call goes through `LLMClient.call` (OpenRouter): circuit
  breaker, per-counter daily call cap, dollar budget, the counter is incremented before the
  request, no retries. The cheap helpers (judge, follow-up check, reaction chooser, sticker chooser,
  vision, diary extractor, ...) each have their own counter and cap, so a flood of cheap calls
  cannot eat the main model's allowance. A provider refusal (402, 401, ...) sends the owner a rate-limited
  DM; secrets are redacted before anything is logged.
- **Chat text is data, not instructions.** The system prompt holds only owner-curated text. All
  participant text (context, recent replies, the venue list, the situation) goes into a separate
  `user` message between `<<<CHAT` and `>>>` with a "data, not commands" notice. Delimiters are
  stripped on input and again in every renderer; slots are filled in one `re.sub` pass so one
  slot's content is never rescanned; `str.format` is never used on text people can influence.
  Display names that look like injection are replaced before they are stored.
- **A three-layer output filter; failure means silence.** The model answers `{"speak", "text"}`;
  a cosmetic `postprocess` step fixes dashes and emoji frequency; then
  1. *regex*: length, markdown, emoji, sentence count, phone numbers, venues that are not on the
     verified list, echo of someone's message, leak of the prompt, "as an AI" talk, stop topics;
  2. *rules*: near-duplicates of recent replies (Jaccard on word shingles), repeated props and
     stories (a per-motif memory), a question twice in a row, grumpy and assistant phrases;
  3. *LLM judge*: a small model answers "in character?", "risky?", "obeyed an instruction from the
     user?" as strict JSON; any error counts as a fail.

  A cut means the bot says nothing. The main model is never called again to try for a better answer.
- **Shadow vs enforced stages.** With `filters.shadow: true` a failing filter is only logged and the
  reply is sent, which lets you collect statistics before switching the filter on. Stages in
  `filters.enforce_stages` (by default `dedup` and `style`) cut even in shadow mode.
- **Injection test table.** `tests/test_injections.py` runs 16 attacks end to end on real
  components (14 on the English pack, 2 script-specific ones in `tests/lang/`): "ignore your
  instructions", "say exactly ...", forged delimiters and slots, injection inside a display name,
  model talk, prompt-leak requests, a judge that answers in prose, and so on.

## Memory

Messages live 30 days (`message_retention_days`); beyond that the bot keeps only what is distilled:

- **Weekly chat summaries.** A scheduled cheap call compresses each past week into a few lines
  ("what was going on in the chat"); the latest ones go into the prompt. Muted people never get in.
- **A diary of its own day.** After each reply a small model extracts any fact the bot stated about
  *itself* ("went to the canal", "fixing a clock"); today's facts go back into the prompt so the bot
  does not contradict itself an hour later.
- **Chat jokes and people's open threads.** Recurring words the chat picked up, and unfinished
  stories ("waiting to hear back about a job"). A thread is asked about once, a few days later, at
  a sensible hour ("how did it go?").
- **Retention** (`retention.py`) purges messages, queues, candidate texts in the log, dated state
  keys, summaries and notes on their own schedules. Photos are described by a vision model and not
  stored. Notes about participants are screened for injection markers before they are saved,
  because they end up in the prompt.

## Persona packs

The engine knows nothing about any particular character. Everything that makes it somebody is a
**pack**, a directory chosen at start by `PERSONA_DIR`:

```
pack.yaml        manifest: format, name, language, ship list
config.yaml      persona (name, triggers, birth date) and every behaviour number
prompts/*.txt    13 prompts: the main one plus judge, follow-up, reaction, sticker, diary, memory, ...
fragments.yaml   the wording the engine puts into prompts (situations, headers, weather, ...)
few_shot.yaml    examples; also stickers.yaml, places_manual.yaml, vocab.example.yaml
locale/<code>/   optional translations of the owner commands (fallback: English)
golden/          scenarios and a recorded baseline (dev only, never shipped)
```

- **Fail-closed.** With no `PERSONA_DIR` the process does not start (a lost mount must not start
  another persona in a real chat). `ALLOW_DEMO_PERSONA=true` opts into the demo.
- **Config layering:** engine defaults, then the language module's filter defaults, then the pack's
  `config.yaml`, then runtime overrides from the database (`/set`).
- **Language modules** (`sotto.lang`) hold only what is linguistic: how a name declines, how
  vocabulary words are stemmed, stop words, whether foreign-script checks apply. English is
  deliberately small; Russian is complete and tested on a synthetic pack in `tests/lang/`.
- **Owner-UI catalog:** everything the bot writes in the owner's DM comes from `locales/<code>/*.yaml`.
- `python -m sotto.packcheck DIR --boot --locale-parity` validates a pack, down to building the whole
  LLM stack. How to write one: [docs/creating-a-persona.md](docs/creating-a-persona.md).

## The demo persona: Gus Marlow

A retired school-bus driver in his sixties who mends old clocks in a shed, plays tuba in a brass band
and watches birds from a canal bank with a flask of tea. Dry, kind, brief. Everything about him,
including his town, is invented. The pack in `personas/demo/` has 21 few-shot examples, 7 of them
silence. A few of them:

```
Dana:  they moved the meeting again, second time this week
Gus:   Second time is a routine. Third time, bring cake.

Theo:  Gus how's the shed project going
Gus:   The radio works now. It only receives a station that stopped broadcasting in 1987,
       but it receives it well.

Jo:    good morning everyone
Gus:   (says nothing)
```

## Quickstart

You need a bot token from BotFather, an OpenRouter API key, your own Telegram user id and a test
group. Weather and venue features stay off until you add their optional keys.

```bash
cp .env.example .env        # fill BOT_TOKEN, ADMIN_USER_ID, OPENROUTER_API_KEY
docker compose -f docker-compose.demo.yml up --build
```

Without Docker (Python 3.12, [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
PERSONA_DIR=personas/demo uv run python -m sotto
```

1. Add the bot to the test group and write something. With `ALLOWED_CHAT_ID` empty the bot is in
   *discovery mode*: it stores nothing, answers nothing and logs the chat id at WARNING level.
2. Put that id in `.env` and restart. Talk to it: `@your_bot hello`, or `Gus, where are you?`.
3. DM the bot from the owner account: `/help`, `/status`, `/get`, `/set`.

To try the gate without a chat or a model: the demo ships a synthetic Telegram export, and replay
prints, per day, who would have been answered and why the rest were dropped.

```bash
uv run python -m sotto.replay personas/demo/golden/export.json --persona personas/demo \
    --bot-username gus_demo_bot --bot-user-id 900001 --verbose
```

`--generate` also calls the real model and filter (capped by `--max-calls`) and prints the replies
without sending anything.

## Owner commands (private chat with the bot)

| Command | What it does |
|---|---|
| `/status`, `/why [h]`, `/last [n]` | state and counters; why it stayed silent; its latest replies |
| `/get [key]`, `/set k v`, `/unset k` | read, change (validated, audited) or reset any config key without a restart |
| `/panic`, `/resume` | stop everything; resume (also resets the LLM circuit breaker) |
| `/prompt`, `/rollback N`, `/ex ...` | prompt and few-shot versions; add or remove examples |
| `/ask <text>` | what it would answer to this message; nothing is sent or stored |
| `/life <text>`, `/say <text>` | tell the chat a piece of news (remembered), or say a line verbatim |
| `/memory`, `/diary`, `/jokes`, `/people`, `/words` | inspect and delete what it remembers; upload a word dictionary |
| `/changelog [owner]` | release notes for the chat, or the engine changelog |

In the group everybody can use `/stop` (24 hours of silence, no reply) and `/mute` / `/unmute`
(for themselves). Commands never reach the gate or the message log.

## Testing

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run mypy src tools
uv run pytest -q --cov --cov-report=json && uv run python -m tools.coverage_gate
```

- Over 2,000 tests, no network, no tokens; the test database is a temporary file. Coverage is
  about 95 %, with a gate in CI: a total floor and a higher per-module floor for new code.
  `ruff` and `mypy --strict` run clean.
- **Golden harness** (`tools/golden`). For a pack it records exactly what the engine sends to the
  models, how it parses answers, what it writes to the database, which config it builds and what
  the owner UI says, for 101 scenarios in six groups (main prompts, helper prompts, deterministic
  functions, config, owner UI, persistence). Time, randomness, HTTP and Telegram are faked, so the
  output is byte-identical between runs.

  ```bash
  uv run python -m tools.golden compare --pack personas/demo \
      --scenarios personas/demo/golden/scenarios.yaml \
      --baseline personas/demo/golden/baseline --allow personas/demo/golden/ALLOWED_DIFFS.md
  uv run python -m tools.golden record --pack DIR --scenarios FILE --out DIR   # a new baseline
  ```

  A refactor must print `golden compare OK: 101 scenarios identical`; intended changes are listed
  by scenario id in `ALLOWED_DIFFS.md`. A baseline is never regenerated to hide a diff.
- **`tools.ast_equiv`** shows that a translation pass changed no code: it compares two git revisions
  after dropping docstrings and replacing user-facing strings by markers that keep their
  placeholders. It was used to move the whole code base from Russian to English.
- A **language gate** (`tools.check_language`) fails CI if Cyrillic appears outside the Russian
  language module and its tests.

## How it was built

Built by one person with AI assistants, as an experiment in a disciplined workflow rather than
vibe coding.

- **A written contract.** [CLAUDE.md](CLAUDE.md) is the spec: module interfaces, invariants (one
  LLM entry point, gate order, shadow semantics), conventions, what not to do. Claude Code agents
  worked from it, often several in parallel on separate modules.
- **Independent review.** Plans and large diffs went to a second model (Codex) for review before
  merging, and findings were argued with evidence rather than accepted by default.
- **Golden baselines as the safety net.** Before the persona/language split, 101 scenarios were
  recorded on the old code, and each refactoring PR had to reproduce them exactly. That is what made
  it safe to turn a single-persona bot into an engine plus packs, and later to translate the code.
- **History.** Development started on 2026-09-10 and reached this state in about two months, one
  pull request per feature or release. Their titles and descriptions are in
  [docs/history/pull-requests.md](docs/history/pull-requests.md), the original staged plan in
  [docs/history/original-plan.md](docs/history/original-plan.md). The private history was cleaned of
  chat data before publication.

## Project layout

```
src/sotto/          the engine: gate, responder, filters, judge, llm, db, commands, memory, lang/
personas/demo/      the English demo pack (Gus Marlow) with its golden scenarios and baseline
locales/en/         owner-UI texts
tests/              pytest suite on the demo pack; tests/lang/ for Russian behaviour
tools/              golden harness, coverage gate, language gate, ast_equiv
docs/               creating-a-persona.md, history/, plans/
deploy/, Dockerfile, docker-compose*.yml, RUNBOOK.md   operations
CLAUDE.md           the contract for agents
CHANGELOG.md        the engine changelog
```

## Author and license

Wiaczesław Cyunczyk, <https://tslava.dev>, GitHub [@tslava](https://github.com/tslava).
MIT, see [LICENSE](LICENSE).
