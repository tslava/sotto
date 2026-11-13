# Demo persona: Gus Marlow

The persona pack that ships with the engine. Everything in it is invented: the town of
Brindlemoor, its pubs, the sticker set, the people in the golden scenarios.

**Gus Marlow** is a retired school bus driver in his sixties who mends old clocks in a shed,
plays the tuba in a brass band and watches birds from a canal bank with a flask of tea. He is a
regular of a small friends' chat, not an assistant: dry, kind, brief, and he tells a short story
only now and then. He answers when someone talks to him, chips in once or twice a day on his own,
and stays silent a lot.

## Quickstart (skeleton)

Requirements: a Telegram bot token from BotFather, an OpenRouter API key, your own Telegram
user id (the owner), and a group chat to test in.

```bash
cp .env.example .env
# BOT_TOKEN, ADMIN_USER_ID, OPENROUTER_API_KEY; leave ALLOWED_CHAT_ID empty for now
docker compose -f docker-compose.demo.yml up --build
```

1. Add the bot to your test group and write a message. The log (WARNING lines) prints the chat id.
2. Put it in `.env` as `ALLOWED_CHAT_ID`, restart. Talk to the bot: `@your_bot hello`, or just `Gus,
   where are you?` (his name triggers are `gus` and `marlow`).
3. Message the bot privately, from the owner account: `/help`, `/status`, `/get`, `/set`.

Without Docker: `uv sync`, then `PERSONA_DIR=personas/demo uv run python -m sotto`.

## What is in the pack

| File | Purpose |
|---|---|
| `pack.yaml` | manifest: name, language (`en`), the list of shipped files |
| `config.yaml` | the persona (name, triggers, age) and every behaviour number |
| `prompts/*.txt` | the 13 model prompts: the main one and the cheap judges and extractors |
| `fragments.yaml` | the wording the engine puts into prompts (situations, headers, weather) |
| `few_shot.yaml` | 21 examples, 7 of them silence |
| `stickers.yaml` | a synthetic 4-sticker catalog; the file ids are fake, see `stickers_fill` |
| `places_manual.yaml` | six invented venues; fill the cache with `places_fill --manual-only` |
| `vocab.example.yaml` | an example dictionary of invented words; send it to the bot with the caption `/words` |
| `chat_notes.md` | release notes for the chat (`/changelog`) |
| `golden/` | synthetic scenarios, a small Telegram export, the recorded baseline |

The owner texts (`/help`, `/status`, errors) are the engine's English catalog, so the pack has no
`locale/` directory.

## Checking the pack

```bash
uv run python -m sotto.packcheck personas/demo --boot --locale-parity
uv run python -m sotto.replay personas/demo/golden/export.json --persona personas/demo \
    --bot-username gus_demo_bot --bot-user-id 900001 --verbose
uv run python -m tools.golden compare --pack personas/demo \
    --scenarios personas/demo/golden/scenarios.yaml --baseline personas/demo/golden/baseline \
    --allow personas/demo/golden/ALLOWED_DIFFS.md
```

`replay` runs the gate over the export and prints, per day, who would have been answered and why
the rest were dropped; with `--generate` it calls the real model (needs the API key).

## Making your own persona

Copy this directory, change `config.yaml`, `prompts/system.txt`, `few_shot.yaml`, the stickers and
the places, and keep `python -m sotto.packcheck` green. The golden scenarios re-record with
`tools.golden record`.

## English is a demo language

English support is deliberately small (`src/sotto/lang/en.py`): names take an optional `'s`, the
vocabulary rules know plain `-s/-ed/-ing` endings, and there is no foreign-alphabet venue filter,
so invented venues are caught by the places block and the judge, not by `regex:venue`.
