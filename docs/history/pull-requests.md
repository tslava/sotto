# Pull requests, September to November 2026

The project was developed in a private repository with one pull request per feature or release.
That repository's history was cleaned of private chat data before publication, and this repository
was created from the cleaned history, so the original pull request pages are not part of it.
Their (English) titles and descriptions are kept here. `PR N` in commit messages refers to this list.

## PR 1: Persona: działka -> plot

Merged 2026-09-11 as commit `e65e521`; 6 files, +71 / -10.

The Polish word in the persona's texts is replaced with "plot": card, prompt, few-shot, README. The filter vocabulary keeps działka so an accidental Polish word is not cut.

## PR 2: Image: run without uv at runtime

Merged 2026-09-11 as commit `9915ff5`; 3 files, +21 / -12.

On the host the container crashed in a loop: uv run under a user with no home directory could not create its cache. CMD now runs python from the venv directly. deploy.sh captures the crashed container's logs before recreating it. setup-host.sh leaves existing ufw rules alone.

## PR 3: Character: more good-natured, no brush-offs and no "I told you so"

Merged 2026-09-11 as commit `f9d04ae`; 10 files, +122 / -11.

From a review of 20 live replies: the persona brushed questions off, got irritated at repeats and complained about being discussed. Removed the wording about a phrase it would not be happy to hear and about "trolling", added good nature and an answer on the substance before the tale. New filter rule style:grumpy. Beer: a cheap brand. Few-shot: two dry examples rewritten warmer, two new ones.

## PR 4: /resume resets the LLM circuit breaker; RUNBOOK on the OpenRouter limit

Merged 2026-09-11 as commit `8bde618`; 5 files, +215 / -3.

After the incident with a 402 from OpenRouter with no balance: /resume now also clears the LLM circuit breaker, /status shows its state, RUNBOOK describes the scenario.

## PR 5: Character: humor and playing along; few-shot rewritten

Merged 2026-09-12 as commit `0fa627e`; 8 files, +85 / -71.

The persona answered like a questionnaire. Added to the prompt: almost every reply has something to smile about; it plays along with teasing instead of correcting; it backs jokes about the server. Few-shot rewritten: 13 examples with humor in its manner, 4 silences.

https://claude.ai/code/session_01CNAELGZraWe4AXbwgfuzSF

## PR 6: Emoji reactions instead of silence on a dice cut

Merged 2026-09-12 as commit `ce43094`; 13 files, +803 / -2.

#### What
When the gate cuts a message by `gate:dice` or `gate:ambient_cooldown`, the bot sometimes sets a thumbs-up or poop reaction on that message via `setMessageReaction` instead of staying fully silent. No model call, no new message.

Rules (`behaviour.reactions` in config.yaml): probability 0.2, at most once an hour, not to the same person twice in a row, cap of 8 a day, emoji only from `filters.allowed_emoji`. Switch: `/set behaviour.reactions.enabled false`.

No reaction is set for any other cut reason (night, panic, stop topic, logistics, not a live conversation...) and never on PASS. `gate.py` is unchanged; the decision lives in the new `reactions.py`.

#### Also
- `/status` shows `reactions=N/cap`; `/why` shows `react:sent` / `react:error`.
- Telegram errors (reactions disabled in the chat, message deleted) are logged as WARNING and `react:error`; the handler does not crash.
- Docs: CHARACTER.md section 6, PLAN.md, README, RUNBOOK, CLAUDE.md.
- Tests: 20 new (a pure function per rule, state, sending with a fake Bot, handler integration, config validation).

https://claude.ai/code/session_01CNAELGZraWe4AXbwgfuzSF

## PR 7: Stickers instead of text: a second cheap-model call, a sticker catalog

Merged 2026-09-14 as commit `aeb612e`; 24 files, +2394 / -16.

#### What
The bot sometimes answers with a sticker from one sticker set instead of text.

The "second call to a cheap model" scheme (owner's decision): the main model writes the text as before, the text passes the output filter as before, and the finished text together with the interlocutor's message and a list of captions goes to the nano model (the same one as the judge). It answers with a sticker number or null. The sticker menu is never mixed into the main prompt.

Limiters (`behaviour.stickers`): **at most 1 per day**, at least 4 text replies between stickers, the last 10 used are not repeated, only on an address or ambient (not morning, not spontaneous). Switch `/set behaviour.stickers.enabled false`.

#### Catalog
`stickers.yaml`: 56 stickers, captions recognized by a vision model via `python -m trolobot.stickers_fill <set_name>`, 5 unrecognized ones filled in by hand, 1 (a picture without a caption) disabled. A rerun of the script merges by file_id and does not overwrite the owner's edits. The `text`/`when` fields are worth reviewing by eye.

#### Also
- `LLMClient.call_raw` for messages with images; `call` became a wrapper over it.
- `/status` shows `stickers=N/1 (55 in the catalog)`, `/why` shows `send:sticker`.
- In `bot_replies` a sticker is stored as `[sticker #N] caption` so it gets into the model context and `/last`.
- Docs: CHARACTER.md section 6, PLAN.md, README, RUNBOOK, CLAUDE.md.
- Tests: 55 new, 809 in total.

https://claude.ai/code/session_01CNAELGZraWe4AXbwgfuzSF

## PR 8: Emoji where they fit, no dashes: post-processing, prompt v7

Merged 2026-09-14 as commit `a737bd4`; 11 files, +552 / -70.

#### What
The owner's observation: an emoji in half of the replies and always at the end, many typographic dashes.

**Prompt (v7).** The emoji rule: place it where it relates to a word or sentence, sometimes mid-phrase, do not glue it to the end out of habit. A new rule: do not use dashes, a hyphen only if there is no other way. The prompt itself and few-shot are rewritten with no dashes at all (comma, period) so the model has nothing to copy: there were 17 dashes, now 0.

**Post-processing before the filter** (`postprocess.py`, new): all kinds of dashes become "-"; if one of the last 4 replies already had an emoji, the new one is cut from the text (instead of cutting the whole reply in the filter, which in shadow mode cuts nothing anyway); more than one emoji leaves the first. In `/why` these are `fix:dash`, `fix:emoji_freq`, `fix:emoji_count`. A reply that is a single emoji after the fix means silence `fix:empty`.

**Filter.** The `style:emoji_position` rule is removed (it required the emoji strictly at the end). `style:emoji_freq`/`style:emoji_count` stay as statistics.

#### Also
- Docs: CHARACTER.md (sections 3, 4, 6), PLAN.md stage 4, CLAUDE.md, RUNBOOK.md.
- Tests: 31 new, 840 in total.

https://claude.ai/code/session_01CNAELGZraWe4AXbwgfuzSF

## PR 9: Config key reference: /get <key> with descriptions and /help

Merged 2026-09-14 as commit `46a9b4f`; 11 files, +809 / -116.

#### Why

The owner configures the bot from Telegram, with no repository at hand. `/set key value` exists, but there was no way to learn which keys exist and what they mean.

#### What was done

- `config_models.py`: every field has `Field(description=...)` in Russian (75 keys, up to 90 characters). The descriptions come from the model; there is no separate dictionary.
- `config.py`: `KeyInfo` and `describe_key()` give the value, type, range from `ge/le`, description, the yaml default on override, and `settable` (persona.* only in yaml). `ConfigStore.describe()`.
- `/get <exact key>` shows a card; `/get [prefix]` shows the list as before plus a hint line. Example:
  ```
  behaviour.daily_cap: 3
  type: int, 0..50
  Daily limit for ambient and "just because" combined
  ```
  (the real card text is in Russian)
- An explicit `/help`; a block of chat commands (`/stop`, `/mute`, `/unmute`, `/ex add`) is added to the help.
- Lists in `/get` are no longer wrapped by the YAML dump at 80 characters.
- Contract in CLAUDE.md ("Interfaces: config key reference"), README "Commands" section.
- A new explicit dependency `annotated-types>=0.8` (it was transitive through pydantic).

#### Verification

`ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 858 tests. Cards were run on the real `config.yaml`.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 10: Life events: /life posts and remembers news, /say verbatim

Merged 2026-09-14 as commit `d8be462`; 28 files, +4749 / -98.

#### Why

The owner wants to decide when the bot announces news about itself (for example, selling an old car and buying another), and for the bot to remember it afterwards without editing the prompt by hand.

#### What was done

- **`/life <text>`** in DM: the event is written to the new `life_events` table (migration `user_version` 1 -> 2, `CREATE TABLE IF NOT EXISTS`), the bot immediately generates a reply for the "you have news" situation and sends it to the chat. The night window, limits and dice are not checked; only `panic`/`stop` hold it. The owner gets "Recorded #N. Sent: ..." in DM, or the cut reason and the candidate text.
- **Memory**: a `{life}` slot in `system.txt` after the biography, marked "this is fresher and more important than what is written above". It does not age and is cleaned with `/life rm N`. `/life list`, `/life post N` (repeat after a cut or `/resume`). Slot lines have no "- " so `regex:prompt_leak` does not treat them as instructions.
- **`/say <text>`**: verbatim to the chat, no model and no filter, stored in `bot_replies` as `trigger="say"`, not written to memory.
- `Responder._generate_and_send` returns `SendOutcome(sent, text, reason, tg_message_id)`; `announce_life`/`say` run under the same lock; `"life"` does not touch the mention/ambient/spontaneous counters, stickers or venues.
- The note text goes as `trigger_text`: Latin-script words from it (a car model name) land in the `regex:venue`/`regex:latin` whitelist. Checked by a test and by hand.
- `audit_stop(..., new_value)`: the audit of `/life` and `/say` stores the text.
- `/status`: `life events: N (M not sent)`. `/help`, README, contract in CLAUDE.md.

#### After deploy

`PromptStore` creates a new prompt version from the file by itself ("seed from file"). The migration applies on start.

#### Verification

`ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 910 tests (52 new).
#### Hot window (second commit)

After a successful `/life` or `/say` a window opens for `behaviour.hot_window.minutes` (30): the delay of a reply to an address is capped by `mention_max_delay_sec` (120), ambient needs no live conversation, no cooldown and no daily limit, with probability `ambient_probability` (0.5) and a cap `ambient_cap` (4) per window. Outside the window behaviour does not change at all. `/status` shows `hot window: until HH:MM (n/cap)`. 934 tests.
#### Attention like a living person (third commit)

The chat is slow and people write to the bot without a reply or its name. A human attention rhythm instead of timer rules:

- **Phone in hand**: the hot window opens after any bot reply. In the window a message the gate rejected as a non-address goes through a cheap check by the judge model (`followup.py`, `prompts/followup.txt`, its own `followup_calls` counter, cap `behaviour.followup.daily_cap`). "Yes" -> `followup` trigger, answered like an address, with the window's delay cap; the model gets the caveat "if it is not for you, stay silent".
- **Put aside**: the window closed, only reply, name and @ get through.
- **Back to check** (`checkin_job`): `behaviour.checkin.after_min` (120-240 min) after the window closes, in waking hours, one main-model call over everything written after its last reply. The answer goes as a reply to the message the model chose (`reply_to` in JSON). It answered -> the window is open again. It stayed silent -> the next check in 2-4 hours, if there was writing. `topic_max_hours` (48) without replies -> the topic is dead.
- `LLMClient.call` got `counter_key`/`calls_cap`; `hot_window.ambient_cap` 4 -> 8; `/status` shows `followup calls` and `checkin: due`.

994 tests.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 11: Long-term chat memory and vision on photos

Merged 2026-09-15 as commit `edb4387`; 28 files, +3564 / -35.

#### Why

Two features by the owner's decision: the bot should remember what was talked about months ago even though messages live 30 days, and it should see photos instead of "[photo]".

#### Long-term chat memory

- A `chat_memory` table (migration `user_version` 3). Once a week, in the 04:00-06:00 window, a model compresses the past period into 3-7 "what happened in the chat" lines (`chat_memory.py`, `prompts/memory.txt`). On the first run it catches up on up to 4 past weeks from the available messages.
- A `{chat_memory}` slot in the system prompt before life events: the last 8 summaries, by week, as the persona's memories. Lines have no "- " so they do not count as a prompt leak.
- Messages of muted participants do not get into the summary. Kept for 365 days, cleaned by retention. Period boundaries follow local days, accounting for the DST change.
- `/memory list | rm N | run`, a line in `/status`, audit.

#### Photo vision

- `vision.py`: a picture is described by a vision model in one or two sentences; the description goes into `messages.text` as `[photo: ...] caption` before the gate, and from there the photo lives as plain text: name triggers in the caption, followup, reply.
- Only when there is a reason: an address, a hot window, or a 30% dice, under a cap of 20 descriptions a day with its own `vision_calls` counter. Photo bytes and base64 are not logged or stored. A download or model error does not break message handling.
- A line in `/status`, `vision:described / skipped / failed` in `/why`.

#### Verification

Two branches from Opus agents are merged, conflicts in shared files resolved by hand. `ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 1090 tests (96 new). `config.yaml` loads, both config classes validate `model`.

#### After deploy

The migration and the new prompt version apply by themselves. The first memory run happens at night in the 04:00-06:00 window or via `/memory run` in DM.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 12: Fewer and more varied replies: presence cap, gap, quiet, props and tales

Merged 2026-09-16 as commit `b68a979`; 30 files, +2384 / -85.

#### Why

On 15.09 the chat stopped the bot for a day: it was said to be too present and too repetitive. On 14.09 the bot gave 19 replies against 49 human messages (39%), 6 of them unprompted; across 22 replies the wife appeared 7 times, the greenhouse 5, the garage 5, a reference to the nineties 5, and almost every reply followed one formula. The filter saw the repeats, but `shadow: true` let everything through. Six measures by the owner's decision, all thresholds in config.

#### Fewer

1. **Presence cap** `behaviour.presence`: the bot's replies per day at most 15% of people's messages plus 2 free ones. Above the cap only replies to the bot's messages, `/life` and `/say` pass. Checked in the gate (`gate:presence_cap`) and before sending (`send:presence_cap`, `send:recheck_presence`). `/status` shows `presence: n/allowance`.
2. **Gap between replies** `behaviour.min_gap_sec` (300): an address is postponed, unprompted replies are cut with `send:min_gap`. The **hot window** again opens only after `/life` and `/say` (`hot_window.open_on_any_reply: false`).
3. **"Back to check"** only in quiet (`checkin.quiet_min` 10) and only on something explicitly addressed: the message chosen by the model is rechecked by a cheap model, and without an addressee nothing is sent. The situation text is tightened.

#### More varied

4. **Shadow is off for dedup and style** (`filters.enforce_stages: [dedup, style]`): repeats and extra emoji are now actually cut, no retry, the bot stays silent.
5. **Props and tales** `filters.motifs`: props used in the last 10 replies are collected (wife, greenhouse, garage, the nineties...), the `{avoid}` slot in the prompt tells the model "now without them"; a motif from the last 3 replies in a candidate is cut with `dedup:motif`. Tale quota: more than one in 5 replies is cut with `style:story_quota`, and the slot demands a short on-point answer.
6. **Character text**: "a tale is not in every answer", "often something to smile about" instead of "almost every", a ban on fitting into other people's plans and trips; `/life` no longer suggests going together; 3 of 17 few-shot examples are replaced with short answers without a tale. CHARACTER.md is synchronized.

An example of the slot on yesterday's real replies (the real text is in Russian):
```
In your recent replies you already mentioned: the greenhouse, the garage, the nineties, the son, the car, "but then".
Now without them: another detail or no tale at all. Do not tell a tale now: a short on-point answer, one phrase.
```

#### Verification

Two branches from Opus agents are merged, a conflict in tests resolved by hand. `ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 1168 tests (78 new). `config.yaml` loads with the new defaults.

#### After deploy

The bot is currently under `/stop` until 16.09 14:00. The new prompt version is created by itself. Success metric: the share of the bot's replies against people's messages per day, visible in `/status`.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 13: /status: show an expired stop as "none"

Merged 2026-09-16 as commit `23a4319`; 2 files, +31 / -3.

The `stop_until` key is not deleted after it expires; the gate just compares it with the current time. `/status` printed the date from the key without that check and showed a stop that no longer applied (16.09, after a stop from 15.09 14:00). Now the date is shown only while the stop is in force. A test covers both cases.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 14: Versions and changelog: 0.5.0, /changelog in DM, version in /status

Merged 2026-09-16 as commit `eef5a6c`; 14 files, +535 / -4.

#### Why

The chat members ask for release notes. So far the version `0.1.0` had not changed, there were no tags, and 13 PRs were undocumented.

#### What was done

- **Version** `0.5.0` in `pyproject.toml`, the single source; `__version__` is read from the package metadata. After the merge I tag `v0.5.0` on the squash commit.
- **CHANGELOG.md** in Keep a Changelog format with a retrospective 0.1.0-0.5.0. Each version has two blocks: "For the chat" (3-5 lines in Russian without technical words, the owner publishes it) and "For the owner" (config, commands, prompt, PR links).
- **`/changelog`** in DM returns the chat block of the latest version, ready to copy; `/changelog owner` the owner block; `/changelog 0.4.0` a specific version. The bot does not post release notes into the chat: that breaks the character.
- **`/status`** starts with the version line, `v0.5.0`.
- **Process** in CLAUDE.md and README: every PR with a behavior change appends to Unreleased; a release = a version bump, moving Unreleased, a tag.

#### Verification

`ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 1187 tests. `/changelog` checked on the real file.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 15: Hot window 10 minutes instead of 30

Opened 2026-09-16. Closed without merging: included in PR 16 together with the reactions, to avoid a CHANGELOG conflict. 5 files, +8 / -4.

Owner's decision of 16.09: after `/say`, half an hour of fast replies looks like "there is too much of it", and ten minutes is enough for the first responses to the news. `behaviour.hot_window.minutes` defaults to 10; config, contract, README and CHANGELOG (Unreleased) are updated.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 16: Delayed, meaningful reactions; 10-minute hot window

Merged 2026-09-18 as commit `6c5daf8`; 15 files, +1270 / -29.

#### Why

On the evening of 16.09 the bot set a reaction 0.3 seconds after a message, and a member noticed. The reaction was chosen by dice, without reading the message: a poop emoji could land on anything. Plus the owner's decision to shorten the hot window after `/life` and `/say` from 30 to 10 minutes: that change was a separate PR 15 and is included here to avoid changelog conflicts.

#### Reactions

- **Pause** of 20-120 seconds before a reaction, `behaviour.reactions.delay_sec`. After the pause conditions are rechecked: one-hour cooldown, cap of 8 a day, not the same author twice in a row. A failure is logged as `react:recheck`.
- **Meaning**: the cheap judge model gets the last 6 messages and the new one and answers whether the persona would set a reaction and which of the thumbs-up, poop or laughing emoji. In most cases the answer is "no", logged as `react:declined`. The dice is not rolled with `semantic: true`; the model is the filter. Its own call counter `reaction_calls`, cap 40 a day.
- `semantic: false` brings back the old dice. Without an LLM key it is also the dice, with a warning at startup.
- `/status` shows `semantic calls N/40`.

#### Hot window

- `behaviour.hot_window.minutes` defaults to 10, was 30. Already applied on the server via `/set`; the PR fixes the default.

#### Verification

`ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 1215 tests, 30 new. CHANGELOG Unreleased is extended with both blocks.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 17: Cheap prefilter for the "back to check" run

Merged 2026-09-18 as commit `c644249`; 14 files, +599 / -30.

#### Why

Each "back to check" run costs about 3 cents on the main model, and on 17.09 all seven ended in silence: about 20 cents a day for nothing.

#### What was done

- Before the main model, the batch of messages after the bot's last reply is looked at by the judge model in one call: which of them are addressed to the persona or continue its topic. An empty list means the main model is not called, `send:checkin_prefilter_no` is written to the log, and the next check is rescheduled as after silence. A non-empty list means the usual path with the whole batch and `send:checkin_prefilter_yes`.
- If the cheap check is impossible (no model or template, a network error or a broken answer), the batch passes to the main model: we could not filter it out, let it decide.
- Keys `behaviour.checkin.prefilter` and `prefilter_max_tokens`, the prompt `prompts/checkin_prefilter.txt`, the `followup_calls` counter with the same cap.

#### Verification

`ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 1202 tests, 15 new. CHANGELOG Unreleased is extended.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 18: Weather: background in the prompt and by place from the question

Merged 2026-09-19 as commit `09942bd`; 23 files, +2460 / -2.

#### Why

Asked about tomorrow's weather, the persona answered that it would guess: it had no data.

#### What was done

- **Background.** The home weather goes into the system prompt as one line for any trigger, marked "this is background, mention only when relevant". Source: Open-Meteo, free and keyless. Half-hour cache; on a network error the previous snapshot is used; a snapshot from another day is not used, so "tomorrow" does not mean "today".
- **Another city.** If a weather regex matches in an address, a cheap model extracts the place name in the nominative case from the text: inflections cannot be parsed with regexes. The Open-Meteo geocoder returns coordinates; city coordinates are remembered for a month, including a negative answer. A second line is added to the prompt. For messages that are not addresses no place lookup is done.
- **Privacy.** There are no coordinates in the code or in `config.yaml`: the repository is public. They live in `.env` together with the home city name, which is not substituted into the prompt text and is only needed to avoid duplicating the block when asked about the home city itself. Without coordinates the weather is simply off.
- **Texts without inflections**: "Weather outside ...", "Asked about a place: <name>. Now ...". The previous templates produced ungrammatical forms.
- In the log: `weather:place`, `weather:no_place`, `weather:not_found`. In `/status` a line with the current weather and the update time.
- `ruff format` no longer rewrites code blocks inside `CLAUDE.md`.

#### Verification

`ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 1318 tests, 88 new. There is no network in tests, the coordinates in them are arbitrary.

#### After merge

Three lines with coordinates and the city name need to be appended to the server `.env`, otherwise weather in production stays off. I will do it at rollout.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 19: Name in grammatical cases; cheap-model address check

Merged 2026-09-22 as commit `871d5f5`; 19 files, +1077 / -77.

#### Why

Two mirror cases from the chat. A message mentioning the persona in the third person got an answer as if addressed to it. And a message mentioning the persona in an oblique case was not heard at all and died as a non-live conversation, because the name regex does not know cases. One root: the name is searched by a crude match that does not tell an address from a mention.

#### What was done

- **Cases.** A new module builds the name forms from a closed list of endings, so inflected forms of the name and its aliases match. There is no open stem, so unrelated words that merely share a stem do not match. Checked by a table of tests.
- **The decision is given to a model.** A name trigger no longer answers instantly: the cheap judge model gets the context and decides whether it is being addressed or just mentioned. In the log this is `followup:name_yes` and `followup:name_no`. If the check could not be done, we answer: missing an address is worse than one extra reply. A reaction can be set on a refusal; the reason is added to the list of occasions.
- **A reply and an @ mention** pass instantly, without a model, as before.
- Keys `behaviour.followup.name_check` and `name_check_max_tokens`, the prompt `prompts/name_check.txt`, a counter shared with the other cheap checks.

#### Verification

`ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 1387 tests, 69 new. Checked by hand on real phrases from the chat.

I fixed a wording in the changelog after the agent: a message mentioning the persona in an oblique case is exactly the case where it should answer, while a passing third-person remark is the example of "does not butt in".

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 20: Loosen silence thresholds: live conversation, dice, reactions, stickers

Opened 2026-09-22. Closed without merging: landed in PR 21 (first commit), shipped in 0.6.0. 11 files, +54 / -32.

#### Why

Over two days the bot said one phrase against 68 messages, and there were no reactions or stickers at all. Funnel review: 54 of 68 messages were dropped by the live-conversation threshold, of the remaining live ones the dice ate 85%, and the most frequent refusal reason did not even give the right to a reaction.

#### What was loosened

| Parameter | Was | Now |
|---|---|---|
| `live_talk.min_messages` | 3 | 2 |
| `live_talk.window_min` | 10 | 20 |
| `ambient_probability` | 0.15 | 0.25 |
| `stickers.min_replies_between` | 4 | 2 |

Over seven days of chat history the live-conversation threshold goes from 55% to 73% of messages.

- `gate:not_live` is added to the list of occasions for a reaction. It is the most frequent refusal reason, and without it no reactions appeared in a slow chat at all.
- The reactions prompt no longer says "in most cases no": the cheap model took it literally and refused 4 times out of 4. Now the wording is softer, but "a reaction to every message looks intrusive" remains.

#### What I did not touch

The presence cap stays as before: at most 15% of people's messages plus two replies a day. It is the safeguard against a return to "there is too much of it", so loosening the dice and thresholds is safe. The reaction cooldown (an hour), their daily cap (8) and the gap between replies (5 minutes) were not changed either.

#### Verification

`ruff check`, `ruff format --check`, `mypy src`, `pytest` are green, 1387 tests. Tests pinned to the old values are fixed: the "three messages from two people" scenarios pin the thresholds themselves rather than depend on defaults.

https://claude.ai/code/session_01Vg8ut8p9VHmHVSmAT5hCJm

## PR 21: 0.6.0: two-week review: reactions, addressing, daily diary, chat jokes

Merged 2026-10-09 as commit `99b3b8b`; 47 files, +5269 / -130.

Review of behavior for 24.09-09.10 and version 0.6.0. Plan: `docs/plans/2026-10-09-review-fixes.md`; contract: three new sections in CLAUDE.md. Absorbs PR 20 (the `loosen` branch goes as the first commit).

#### What was wrong (from the production DB over 15 days)

- 0 reactions in three weeks: 163 of 262 model answers were truncated (`max_tokens: 40`), and a laughing emoji is not accepted by Telegram as a reaction (`REACTION_INVALID`).
- Direct addresses were lost: `@` under the presence cap, the stop-topic cooldown muted addresses for 45 minutes, an OpenRouter failure buried the reply.
- "Back to check": 66 runs, 0 sends: the prefilter passed what the main model does not answer.
- A new prop, the plant (about 8 replies of 52), and a manner of steering the conversation to itself.

#### What was done

**Fixes**
- Reactions: `max_tokens` 120, a different laughing emoji, check against the set of Telegram reactions, U+FE0F does not count.
- Under the presence cap a reply and `@` pass; `topic_cooldown` only for ambient; an address on `llm:http`/`llm:timeout` is postponed once by 2-5 minutes.
- The plant motif, a "do not steer the conversation to yourself" line in the prompt; the checkin prefilter has the same criteria and all names of the bot.
- `loosen` (PR 20): live_talk 2/2/20, ambient 0.25, more frequent stickers, `gate:not_live` gives the right to a reaction.

**More interesting**
- Daily diary (`{diary}`, `/diary`): facts about itself from its own replies so it does not contradict itself.
- Chat jokes (`{jokes}`, `/jokes`) and "how did it go?" (`callback`, `/people`) from the weekly summary.

The stop-list of topics was not changed (owner's decision). Migrations 4 and 5 are `CREATE TABLE IF NOT EXISTS`.

#### Review

Each task went through Codex (adversarial) review and a separate reviewer; fixed: a double reply on collapsing into a pending whose generation is running (the bug existed before), binding a thread to a namesake, duplicate threads, instructions in long-term memory (`Patterns.unsafe_note`), the time of a diary fact, closing a thread on cancellation, restoring pending after a restart. Deliberately not done: a persistent flag for the A6 retry, isolating the LLM circuit breaker from cheap calls (as with judge/followup/reactions).

#### Verification

`ruff check`, `ruff format --check`, `mypy src` are clean; `pytest`: 1542 passed.

https://claude.ai/code/session_01LqBp73PApq2Q4CUwWpnZER

## PR 22: 0.6.1: talk about itself every other time

Merged 2026-10-12 as commit `5c6cd58`; 25 files, +461 / -52.

After 0.6.0 the bot still talked about itself in almost every reply: from 09.10 to 12.10, 7 of 7 unprompted replies were about itself. The owner is fine with the frequency of interventions: the gate and probabilities are untouched, only the content changes.

- **Ambient:** instead of an empty situation, `SITUATION_AMBIENT`: respond to the last message, to the topic, or turn it; do not start its own story (otherwise `speak=false`). Replay `--generate` builds ambient the same way.
- **Few-shot:** about itself only in answers to questions about it; for something not said to it, a reaction about the other person. 21 examples, 5 silences. CHARACTER.md is synchronized.
- **Diary:** a neutral header of the `{diary}` slot, the prompt rules out memories, "usually happens" and exaggeration.
- **`/status` metric:** "about self over 3 days: ambient X/Y, addressed X/Y, other X/Y" (`filters.self_markers`), cuts nothing.

A quota filter `style:self_quota` is deliberately not made: the `style` stage in `enforce_stages` would cut at once. First we watch the metric for 3-4 days.

Each task went through Codex adversarial review; findings are fixed in separate commits.

https://claude.ai/code/session_01LqBp73PApq2Q4CUwWpnZER

## PR 23: 0.6.2: reactions that fit the message

Merged 2026-10-15 as commit `a9a77ca`; 11 files, +287 / -23.

A third of the reactions were off: the laughing emoji on a complaint, on a reply to a casual question, on a remark right after someone mentioned an injury, a thumbs-up on a photo without a description.

- `prompts/reaction.txt` rewritten: almost always null; the laughing emoji only if the author is joking; always null on a complaint, failure, misfortune, illness, kids taken seriously and anything unclear.
- The `{reply_to}` slot: the message the author replied to (even if it is older than the context window). Three states: not a reply, target found, reply to something invisible (purged or not yet recorded), and then the model refuses.
- The reactions model is `openai/gpt-6-luna` (in config.yaml; $0.1/$0.5, half the price of nano). `max_tokens` 120 -> 600: the model thinks first.
- `"emoji": "null"` as a string counts as a refusal.

Measured on 52 past cases: the obvious misses did not repeat on any model; Luna with reasoning chooses most accurately, Haiku 5.5 with reasoning refuses always.

Codex review: the finding about an "invisible reply target" is fixed; the ones about chat data in the system message and the daily-cap race are deferred (old, low risk).

https://claude.ai/code/session_01LqBp73PApq2Q4CUwWpnZER

## PR 24: 0.7.0: slang words from an uploaded glossary

Merged 2026-10-17 as commit `f5f62a1`; 27 files, +2159 / -29.

The persona occasionally and appropriately uses words from an uploaded glossary and knows their meaning if asked. The first glossary is a set of slang words; the persona's backstory is updated accordingly.

- **Loading with no hardcode.** A YAML file sent to the bot in DM with the caption `/words` (format: `vocab.example.yaml`), `/words`, `/words show|on|off|rm <name>`. Glossaries live in the DB (`vocab_dicts`, migration 6); only an example is in the repository.
- **Rarely.** The `{vocab}` slot: 3 words to choose from only on the `behaviour.vocab.probability` dice (0.15), if fewer than `daily_cap` (2) a day, and none of the last `recent_window` (10) replies had a glossary word. The `dedup:vocab` output filter cuts an unprompted word beyond the limit (except a word from the interlocutor's question).
- **Meaning on a question.** A word from an address or the last chat messages goes into the prompt with its meaning; the persona's own recent word only on an address.
- **Word forms** by stem and a closed list of Russian endings (prefixed and reflexive verbs, fleeting vowel, adjectives). No false matches on 1942 real chat texts.
- `/status`: a vocabulary line. No new model calls.

Codex review: three findings (the limit was held only by a hint, a collision between a verb and its reflexive form, crude cases) fixed with tests.

https://claude.ai/code/session_01LqBp73PApq2Q4CUwWpnZER

## PR 25: 0.8.0: /ask, what the persona would answer

Merged 2026-10-18 as commit `26987f5`; 10 files, +651 / -138.

`/ask <text>` in the owner's DM: what the persona would answer if the owner addressed it in the chat with this text. The same prompt with the fresh chat context, the same model, post-processing and output filter. Nothing goes to the chat, nothing is written to the DB (no `filter_log`, replies, counters, diary, jokes or glossary words). The model call is real and counts against the daily budget.

- Prompt assembly is moved out of `Responder._generate_and_send` into `_build_prompt`, plus `_filter_context` and `_cut_reason`: the live reply and the preview share one code path. Behavior in the chat does not change.
- The owner's message is appended to the context in memory only, so the model sees it in full and the filter works as in the chat.

Codex review: the finding about the message missing from the context is fixed with tests. The shared LLM circuit breaker and the daily-cap race (old, common to all parallel calls) are deliberately left.

https://claude.ai/code/session_01LqBp73PApq2Q4CUwWpnZER

## PR 26: 0.8.1: glossary words on request and "didn't understand"

Merged 2026-10-18 as commit `8f6be64`; 15 files, +591 / -35.

A request to say something in the glossary's dialect was silently ignored: the glossary reached the prompt only by the dice (15%) or if the word was already in the message.

- **Request.** An address with the glossary name or `vocab.request_markers` (words like "glossary", "dialect") gives `request_size` (5) words in the prompt with no dice, daily cap or pause. `dedup:vocab` lets them through; if there are not enough unsaid words, it tops up from recent ones.
- **"Didn't understand".** A phrase like "didn't understand", "what does it mean", "meaning?" (`vocab.explain_markers`) within `explain_window_min` (30) minutes after its reply containing a glossary word passes the gate as a followup (`followup:vocab_explain`), on both DROP and ambient PASS. In the prompt the meanings of its words go first, and the filter does not cut the explanation. It does not count as a question to it: a reply to another person, and the case when people wrote more than `explain_max_between` (2) messages after its reply.

Codex review: three findings (a reply to someone else, ambient PASS with no exception, an empty list on request) fixed with tests.

https://claude.ai/code/session_01LqBp73PApq2Q4CUwWpnZER

## PR 27: 0.9.0: owner alert on provider refusal and more varied reactions

Merged 2026-11-05 as commit `a8cf1a5`; 20 files, +946 / -65.

Two things in one release.

**Owner alert on provider refusal.** On 22-26.10 OpenRouter answered 402 (credits ran out), the bot was silent for four days, and `/status` showed "errors in a row: 0": the cheap models passed and reset the streak. Now on a refusal (4xx except 408 and 429) the bot writes to the owner in DM, at most once per `llm.alert_interval_hours` per code. Sending is in the background, the slot is taken before sending, a failure or a shutdown means a retry in an hour. The key and tokens are cut from the provider's answer in the alert and in logs. `/status` shows the last provider refusal within a day (temporary 429/5xx do not overwrite it).

**Reactions.** Over 26.10-05.11, of 29 reactions 21 were the laughing emoji and 8 were thumbs-up. `behaviour.reactions.emoji` became a dictionary "emoji: when appropriate" (a list is still accepted), defaulting to seven emoji; `max_repeat` (2) removes an emoji from the choice after two identical ones in a row. Measured on 150 past cases: 22 reactions, 14 of one smiling emoji, 4 laughing, 2 fire, 2 thumbs-up.

Two rounds of Codex review, all findings fixed. ruff, mypy, 1741 tests are green.

https://claude.ai/code/session_01LqBp73PApq2Q4CUwWpnZER

## PR 28: 0.9.1: slang words actually get used

Merged 2026-11-05 as commit `12b7841`; 10 files, +73 / -18.

Since 17.10 the bot has not used a glossary word even once. The mechanics worked: the glossary is loaded, the slot is in the prompt, the matcher and counters are fine. But the wording "you may, only if it comes naturally" was never followed by the model.

Measured on past cases (a copy of the DB, the live one untouched):
- old wording: 0 of 8;
- "use one", a choice of 3 words: 8 of 8, but forced;
- "use one", a choice of 8 words: 8 of 8, fitting;
- the final wording with "do not change your thought for the word", 2 seeds: 12 of 16.

Changes:
- `vocab.render_vocab`: "In this reply use one of these words: pick the one that fits your thought best... Do not change what you are talking about for the word".
- `behaviour.vocab.offer_size` 3 -> 8 (default and config.yaml).
- Matcher: nouns with the irregular "-mya" declension use the "-en" stem, so an unrelated verb form no longer counts as a glossary word.
- Rarity untouched: probability 0.15, daily_cap 2, recent_window 10.
- CLAUDE.md, CHANGELOG 0.9.1, version 0.9.1.

Codex review: one finding, a length threshold let two short nouns through, fixed in a separate commit with a test.

https://claude.ai/code/session_01LqBp73PApq2Q4CUwWpnZER
