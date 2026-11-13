# RUNBOOK: operating sotto

The owner's operations book: what to do by hand, and when, including from a phone. How the code
is built is in `CLAUDE.md`, how to write a persona in `docs/creating-a-persona.md`, installation
and the first deploy in `README.md`. This file covers only what happens after the bot is already
running (stage 7 of the original plan, `docs/history/original-plan.md`).

All owner commands go in a private chat with the bot, and the answer comes back there. `/why [hours]`
accepts at most 168 (a week).

The persona (config, prompts, few-shot, fragments, stickers, venues) is a **persona pack**, a
directory the process reads from `PERSONA_DIR` (see `docs/creating-a-persona.md`). Wherever this
runbook says "edit `prompts/system.txt`" or "`config.yaml`", it means the files of the pack that
runs on the host. Validate a pack before it goes live:

```bash
uv run python -m sotto.packcheck <pack-dir> --boot --locale-parity
# against the live database (read-only): would starting create a new prompt / few-shot version?
uv run python -m sotto.packcheck <pack-dir> --db data/bot.db --seed-dry-run
```

---

## Daily

Nothing. The bot runs by itself; debounce, retention and backups are background tasks and cron.

## Weekly

In the private chat with the bot:

- `/status`: panic, stop, prompt/few-shot versions, the pack name and revision, models, shadow,
  the day's counters (ambient/mention/llm_calls/llm_spent/reactions/stickers), pending, night queue.
- `/why 168`: a summary of the reasons for silence over the week (`stage:reason count`), sorted by
  frequency. The main signal: one reason suddenly dominates, which is a cause to investigate before
  the monthly cycle. `send:sticker` means the reply went out as a sticker rather than text; frequent
  `send:sticker` next to few replies is a cause to lower `behaviour.stickers.daily_cap` or raise
  `min_replies_between` with `/set`. **`fix:*` (`fix:dash`, `fix:emoji_freq`, `fix:emoji_count`) is
  not a cut but a correction**: the reply went to the chat the same, only without a typographic dash
  or with a corrected emoji (`postprocess.py`). Do not count them together with `regex:*`/`judge:*`
  for the cut share (section "Turning shadow mode off"). Many `fix:emoji_freq` means the model still
  sticks an emoji onto almost every reply more often than intended: adjust the emoji rule in the
  pack's `prompts/system.txt`, not `postprocess.py`.
- Check the OpenRouter balance ([openrouter.ai/credits](https://openrouter.ai/credits)). Without a
  top-up, requests are cut by length long before the account reaches zero (see "Incidents from the
  phone" below). The bot also messages the owner when the provider refuses a call (402, 401, ...;
  `llm.alert_owner`).

On the host (your own user and key, not the deploy key; see README, "First rollout", item 6):

```bash
ssh owner@host "sudo -u bot docker compose -f /opt/trolobot/docker-compose.yml \
  --env-file /opt/trolobot/deploy.env logs --since 7d | grep -c WARNING"
```

A noticeable growth in WARNING lines week over week: read the lines themselves (`logs --since 7d`
without `grep -c`), do not ignore them. Log rotation (`docker-compose.yml`, `max-size: 10m`,
`max-file: 5`) keeps only the last ~50 MB; older lines vanish on their own.

```bash
ssh owner@host "sudo -u bot du -h /opt/trolobot/data/bot.db"
```

Fast growth of the size between weeks (not explained by normal chat activity) is a cause to check
that the retention job (`retention.py`, hourly, `message_retention_days` from the pack's
`config.yaml`, 30 days by default) really runs: look for `retention purge: messages=...` lines in
the log.

## Monthly

**1. Review `filter_log` and edit the prompt.**

```bash
ssh owner@host
cd /opt/trolobot
sqlite3 data/bot.db "
  SELECT stage, reason, COUNT(*) AS n
  FROM filter_log
  WHERE verdict = 'cut' AND created_at >= strftime('%s','now','-30 days')
  GROUP BY stage, reason
  ORDER BY n DESC;
"
```

(No need to stop the container: SQLite runs in `WAL` mode, so reading alongside the running process
is safe. This is a `SELECT` only, not a hand edit; see "Never do".) While `filters.shadow: true`
(the default), this includes what really went to the chat; the table itself is the basis for the
shadow decision (section below).

Based on the result, edit the pack's `prompts/system.txt` (answer structure, excess cuts) or
`few_shot.yaml` (voice examples) and roll the pack out the way the host takes it (the pack is a
directory on the host that `PERSONA_DIR` points to). Never touch versions by hand: at container
start `PromptStore.load()` compares the file body with the active version in the database and
creates a new one (the "seed") itself if the file differs, for `system.txt` and `few_shot.yaml`
alike. After the rollout, in the private chat:

```
/prompt
```

and check that the version grew and the body is the new one. For a bad edit, `/rollback <version>`
(see "Incidents" below), not a second change made on a hunch.

**2. Refresh the venue cache.**

`places_manual.yaml` is part of the pack, so edits to `quiet`/`category`/`district` arrive with the
pack. The fill run happens inside the container:

```bash
ssh owner@host
cd /opt/trolobot
sudo -u bot docker compose -f docker-compose.yml --env-file deploy.env \
  exec bot python -m sotto.places_fill --dry-run   # first look at the fact column, writes nothing
sudo -u bot docker compose -f docker-compose.yml --env-file deploy.env \
  exec bot python -m sotto.places_fill              # the real run, writes to data/bot.db
```

You must review the `fact` column in the printed table (a review is somebody else's text from the
internet, an indirect-injection channel; `validate_fact` is the last automatic barrier, not the only
one), and fix by hand whatever was compressed badly:

```bash
sqlite3 data/bot.db "SELECT name, fact FROM places;"
```

A venue that Google did not return in this run is quietly marked `operational=0` and the bot stops
naming it; nothing more to do. Without a Google key use `--manual-only` (README, "Venues", item 3).

**3. Refresh the sticker catalog.**

Only needed after the sticker set in Telegram changed (stickers added or removed). `stickers.yaml`
is part of the pack, so hand edits of `text`/`when`/`enabled` arrive with it. To rebuild the catalog
from the set, inside the container:

```bash
ssh owner@host
cd /opt/trolobot
sudo -u bot docker compose -f docker-compose.yml --env-file deploy.env \
  exec bot python -m sotto.stickers_fill <set_name> --dry-run   # first look at the table
sudo -u bot docker compose -f docker-compose.yml --env-file deploy.env \
  exec bot python -m sotto.stickers_fill <set_name>             # the real run, writes stickers.yaml
```

The script merges with the existing `stickers.yaml`: the owner's edits (`text`, `when`, `enabled`)
to already known stickers (by `file_id`) are kept, new ones get the next ids, ones that vanished from
the set stay in the file but are turned off (`enabled: false`). The result **must be checked by
hand** (`text`/`when` of new stickers are a draft by a vision model, not what is really written) and
committed into the pack's repository, otherwise the next pack rollout reverts the edits to what the
repository had. A restart is not required: `stickers.yaml` is read at start, but this run changed
something on disk inside the container, not in the repository.

To turn stickers off altogether without touching the catalog: `/set behaviour.stickers.enabled
false` (back: `/set behaviour.stickers.enabled true`), or turn individual ones off by hand
(`enabled: false` in `stickers.yaml`). In `/why`, `send:sticker` is a reply that went out as a
sticker; `react:error` and other `stage:reason` have nothing to do with stickers, they are emoji
reactions (another module, see above).

**4. Check the backups.**

```bash
ssh owner@host "ls -lh /opt/trolobot/backups"
```

There should be up to 8 files `bot-YYYY-MM-DD.db` (cron `0 3 * * 0`, `deploy/setup-host.sh` keeps
the last 8, about 2 months), the newest not older than a week. Empty or long stale means cron did
not run: `crontab -u bot -l` on the host and `/opt/trolobot/backups/backup.log`. By default a backup
lies on the same disk as `data/`; it is on a separate volume only if that was set up by hand (README,
"First rollout", item 8). If not set up, remind yourself separately that losing the disk loses the
backup together with the database.

---

## Incidents from the phone

Everything goes from the private chat with the bot, except `/stop` and `/mute` (they go in the group
chat and any participant can use them without a reply).

| Symptom | Action |
|---|---|
| The bot started writing the wrong things | `/stop` in the chat (any participant, 24 hours of silence, no questions) or `/panic` in the private chat (indefinitely, until `/resume`) |
| The bot is silent but should talk | `/why 1` (the reason in the last hour), `/status` (panic? stop_until? day budget?), `/get llm.` (limits and current models). The reason `llm:circuit_open` in `/why` is the fuse after a series of API errors; it releases itself after `llm.circuit_pause_min` minutes and cannot be released by hand |
| The bot is silent to every address, `/why 1` shows `llm:http` or `llm:circuit_open` | (a) OpenRouter with no topped-up balance cuts the request length at about 3400 tokens, error 402 "Prompt tokens limit exceeded" (seen as `llm:http`), fixed by topping up the balance, see [openrouter.ai/credits](https://openrouter.ai/credits); (b) five errors in a row (`llm.circuit_errors`) open the fuse for 30 minutes (`llm.circuit_pause_min`); it is released by `/resume` or by the timer; (c) after release, the addresses unanswered meanwhile are not repeated: whoever got no answer has to write to the bot again |
| The bot sends garbage in the format (markdown, emoji, too long) | `/set filters.shadow false` turns on real cutting (by default `shadow: true` only logs, does not cut) |
| Replies too often | `/set behaviour.ambient_probability 0.05` (or lower) |
| A bad prompt edit | `/rollback <version>`; look the number up in `/prompt` |
| Someone asked it to leave them alone | They `/mute` themselves in the chat, without a reply. Only the owner can mute another person, as a reply to their message |
| Emoji reactions annoy or are too many | `/set behaviour.reactions.enabled false` turns them off completely; `/set behaviour.reactions.probability 0.05` just makes them rarer. `/why` shows `react:error` if reactions are forbidden in the chat by a group administrator (Telegram, chat settings, allow reactions); it is not a bot bug and is fixed in the chat settings, not with `/set` |

---

## Recovery

**The container does not start after a deploy.**

```bash
ssh owner@host "sudo -u bot docker compose -f /opt/trolobot/docker-compose.yml \
  --env-file /opt/trolobot/deploy.env logs --tail 200"
```

Typical start failures: `PERSONA_DIR not set` (the pack mount or the variable is missing; the
process fails closed on purpose and never starts the demo persona unless `ALLOW_DEMO_PERSONA=true`),
`pack ... is missing files` or an invalid pack (run `packcheck` on the pack directory).

CI rolls back to the previous tag itself if the new one did not become healthy within 60 seconds
(the healthcheck is `getMe` against the Telegram API). To roll back by hand, when you need to go back
further or CI is unavailable, see README, "Manual rollback": edit `IMAGE_TAG` in
`/opt/trolobot/deploy.env` and run `docker compose --env-file deploy.env up -d`, or run
`deploy/deploy.sh` with an explicit `sha-xxxxxxx` over SSH from your own checked copy of the
repository. The deploy key (`SSH_KEY` in GitHub Secrets) is not suitable for manual operations: it
is a forced command that only passes the form `bash -s -- sha-xxx`.

**`.env` lost on the host.**

It lives only in `/opt/trolobot/.env` (mode 600) and has never been in git or in CI. Its contents
are in README, "First rollout", item 2:

| Key | Where to reissue |
|---|---|
| `BOT_TOKEN` | [@BotFather](https://t.me/BotFather) `/token` (or `/revoke` plus a new bot if the token is compromised) |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys): a new key, set the `credit limit` again |
| `GOOGLE_PLACES_KEY` | Google Cloud Console, Credentials: a new key, restrict it to the API again (Places API (New)) |
| `ADMIN_USER_ID`, `ALLOWED_CHAT_ID` | If also lost: clear `ALLOWED_CHAT_ID` (discovery mode, README, "Discovery mode") and re-read `user_id`/`chat_id` from the `WARNING` log lines |
| `PERSONA_DIR`, `WEATHER_*` | Set them again; see `.env.example` |

The file is copied to the host once by hand (`setup-host.sh` creates it empty from `.env.example`);
CI does not touch it and cannot restore it.

**Restore the database from a backup.**

```bash
ssh owner@host
cd /opt/trolobot
sudo -u bot docker compose --env-file deploy.env down     # stop the container
sudo cp backups/bot-2026-09-03.db data/bot.db             # pick the date from ls backups/
sudo chown 10001:10001 data/bot.db                        # the uid of the process in the container
sudo -u bot docker compose --env-file deploy.env up -d    # start again
```

Stopping the container before copying is mandatory, otherwise the live process immediately writes
over the backup again.

---

## Turning shadow mode off

The default is `filters.shadow: true`: the filter layers (regex/dedup/judge) only count and write to
`filter_log`, they cut nothing, except stages listed in `filters.enforce_stages` (by default `dedup`
and `style`), which cut even in shadow. The criterion for turning it off:

1. The bot has run in shadow for at least a week.
2. `/why 168` in the private chat: look at the share of the reasons `regex:*`, `dedup:*`, `judge:*`
   among all `verdict='cut'` over the week (the same `filter_log` as in the monthly review, only on
   a weekly window).
3. If in total **more than 30%** of replies would be cut, first fix the rules (frequent false
   `regex:venue`/`regex:latin`: extend `filters.places_whitelist` / `filters.polish_words` in the
   pack's `config.yaml`; frequent `judge:*`: check whether the judge prompt is too strict), then look
   at `/why 168` again after a week. Turning `shadow` off at a high false-positive rate means cutting
   live replies, not garbage.
4. Only when the share is below 30% and the reasons look like real cuts (echo, prompt leak, real
   garbage), `/set filters.shadow false`.

While `shadow: true` and `llm.judge_model` is set, the LLM spending doubles (the judge is called on
almost every generated reply, including those already cut by layers 1-2). Account for this when
planning the `credit limit` on the OpenRouter key (README, "Budget in shadow mode").

---

## Never do

- Open ports to the outside. The bot works on long polling and needs no incoming ports; when it
  comes to a web panel (stage 8 of the original plan), strictly `127.0.0.1` and an SSH tunnel, not a
  public port.
- Copy `data/` (or `bot.db` backups) to the cloud. It is an archive of real people's conversation;
  a backup lives on the host (or on a separate mounted disk of the same server), no further. The same
  goes for real chat exports and for a real persona pack in a public repository.
- Bake secrets into the image. The `Dockerfile` contains none and must not; all keys go only through
  `/opt/trolobot/.env` on the host, `env_file` in `docker-compose.yml`.
- Edit the database by hand in production without stopping the container. The live process and a
  manual `sqlite3 ... UPDATE/DELETE` on the same `bot.db` race and risk a corrupt write; a `SELECT`
  for analysis (see "Monthly", "Weekly") is fine without stopping, any write only after
  `docker compose down`.
