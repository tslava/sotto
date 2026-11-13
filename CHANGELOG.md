# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).
While the major version is 0: minor for a feature, patch for behaviour tweaks and bug fixes.
This file is the engine changelog (config keys, commands, prompts, internals) and is what
`/changelog owner` shows. Chat-facing release notes live in each persona pack's `chat_notes.md`.

## [Unreleased]

### Notes

- Persona packs: everything persona-specific (`config.yaml`, `few_shot.yaml`, `stickers.yaml`, `places_manual.yaml`, `prompts/`, `pack.yaml`) lives in the directory named by `PERSONA_DIR`. Config layering: engine defaults, then the pack's `config.yaml`, then runtime overrides. `python -m sotto.packcheck <dir> [--boot] [--db bot.db --seed-dry-run] [--locale-parity]` validates a pack.
- Fragments: engine wording that goes into model prompts or the DB (situations, block headers, placeholders such as `[photo]`/`[sticker #N]`/"Participant N", weather line pieces, vocab text, `places_fill`/`stickers_fill` prompts) moved from code to the pack's `fragments.yaml` (schema `fragments.FRAGMENT_SPEC`; a pack with a missing or broken file does not start).
- Language modules: `sotto.lang` (ru/en) holds the language-dependent matching (name declension, vocab stemming, text rules); a pack selects its language in `pack.yaml`.
- Owner-UI catalog: everything the bot writes to the owner in DMs (`/help`, `/status`, command replies, provider alerts, errors) comes from `locales/<code>/*.yaml`; a pack may ship its own `locale/<code>/`, missing keys fall back to English. Config key descriptions (`/get <key>`) are English in the engine.
- Neutral engine defaults: persona-bound defaults (birth year and district, places search queries and geocoder language, vocab markers, reaction descriptions) are no longer in the engine; packs set them explicitly in `config.yaml`.
- English demo persona in `personas/demo` (enabled with `ALLOW_DEMO_PERSONA=true` when no `PERSONA_DIR` is set); the process refuses to start without either.
- Golden harness in `tools/golden`: compares assembled prompts and outputs for a pack before and after engine changes.
- The `CONFIG_PATH`, `FEW_SHOT_PATH`, `PROMPT_PATH`, `*_PROMPT_PATH`, `STICKERS_PATH` and `CHANGELOG_PATH` variables are no longer read (one startup WARNING if set); `replay` gained `--persona`, `places_fill` and `stickers_fill` take paths from the pack.
- Tests, docs and code comments are now English; the package was renamed `trolobot` -> `sotto`.
- `/changelog` reads chat notes from the pack's `chat_notes.md` (`tools/split_changelog.py` builds one from an old-style changelog); `/changelog owner` shows this file.

## [0.9.1] - 2026-11-05

### Notes

- Vocab words: over the previous three weeks the chat saw none. The mechanism worked, but the "use one if it fits" hint was never followed by the model (0 of 8 on a replay of past cases). Now, when the dice roll hits (`behaviour.vocab.probability` 0.15, daily cap `daily_cap` 2 and `recent_window` 10 replies pause unchanged), the prompt says "use one of these words in this reply", pick the fitting one and do not change the thought for it. `behaviour.vocab.offer_size` 3 -> 8: with three random words the model forced an unfitting one. Replay of the same 8 cases, two runs each: a word used in 12 of 16, in the rest the model found nothing fitting (#28).
- Vocab matching: nouns ending in "-mya" decline from the "-en" stem, so they no longer collide with similar shorter words (#28).

## [0.9.0] - 2026-11-05

### Notes

- Reactions: `behaviour.reactions.emoji` is now a dict "emoji: when appropriate"; the description goes to the reaction-choosing model. Default palette: 🤣 😁 👍 💩 🔥 👏 🍾. The old list form (`/set behaviour.reactions.emoji [👍, 🤣]`) is still accepted, without descriptions. One emoji at most `behaviour.reactions.max_repeat` (2) times in a row, after that it is removed from the choice (asking in the prompt was ignored by the model). `prompts/reaction.txt` rewritten for the palette, an ordinary on-topic message gets null. Motivation: in a 10-day window 21 of 29 reactions were 🤣 and 8 were 👍. Replay on the same 150 cases: 22 reactions, 😁 14, 🤣 4, 🔥 2, 👍 2 (#27).
- Provider refusal alerts: an OpenRouter refusal (4xx except 408 and 429: out of credits, key rejected, model not found) sends the owner a DM with the code, model, provider message and what to do. At most once per `llm.alert_interval_hours` (24) per status code; disable with `llm.alert_owner: false`. Motivation: a 402 went unnoticed for four days because cheap models kept succeeding and resetting the error streak. Sending is a background task, chat replies do not wait for it; a failed send retries in an hour. The API key and auth tokens are redacted from the alert and from logs (#27).
- `/status`: the circuit-breaker line shows "last provider refusal: 402 at HH:MM dd.mm" or "none in 24h" (#27).

## [0.8.1] - 2026-10-18

### Notes

- A request inside an addressed message (the dictionary name, or `behaviour.vocab.request_markers`) puts `request_size` (5) words into the prompt without the dice, daily cap or pause; the filter lets them through (#26).
- "Didn't get it", "what does it mean" (`vocab.explain_markers`) within `explain_window_min` (30) minutes after a reply containing a vocab word passes the gate as a followup (`followup:vocab_explain` in `/why`); the meanings of that reply's words go first in the prompt, and the filter does not cut an explanation containing them. A reply to another person, or "didn't get it" after more than `explain_max_between` (2) other messages, is not treated as a question to the bot (#26).

## [0.8.0] - 2026-10-18

### Notes

- `/ask <text>` in the owner DM: what the persona would answer if addressed in the chat. Same prompt with fresh chat context, same model, post-processing and output filter; nothing is sent to the chat and nothing is written to the DB (no `filter_log`, replies, counters, diary or vocab). The model call is real and counts toward the daily budget. The reply shows the text, or why it would stay silent, and what the filter flagged (#25).
- Prompt assembly moved out of `Responder._generate_and_send` into `_build_prompt`, shared by live replies and the preview (#25).

## [0.7.0] - 2026-10-17

### Notes

- Vocab dictionaries: upload a YAML file to the bot's DM with the caption `/words` (format in `vocab.example.yaml`); `/words`, `/words show|on|off|rm <name>`. Stored in the DB (table `vocab_dicts`, migration 6), not in the repo. The `{vocab}` slot in the system prompt carries the meanings of words that came up in conversation and, occasionally, 3 words to choose from. No new model calls.
- The limit is enforced beyond the prompt hint: the output filter `dedup:vocab` cuts a reply with a vocab word if the daily cap is already reached or the word appeared in the last `recent_window` replies (except a word from the interlocutor's question). The model sees meanings of its own words only on direct address.
- `behaviour.vocab.*`: `probability` 0.15, `daily_cap` 2, `recent_window` 10, `offer_size` 3, `meaning_context` 5, `max_meanings` 3. `/status`: a vocab line.
- Prompt: the persona's biography gained a hometown detail; PromptStore creates a new prompt version by itself.

## [0.6.2] - 2026-10-15

### Notes

- Reactions are chosen by `openai/gpt-6-luna` instead of `llm.judge_model` (`gpt-5.4-nano`): on 52 past cases nano put a laughing reaction on complaints, on a joking status and on a photo without a description. `reactions.max_tokens` 120 -> 600 (cap 200 -> 1000): Luna and Haiku think first, and at 120 tokens the JSON did not finish, at 400 only occasionally. Haiku 5.5 with reasoning produced no reaction in 52, without reasoning 5 (#23).
- `prompts/reaction.txt`: "almost always null"; the laughing reaction only when the author is joking; always null on a complaint, failure, trouble, kids in earnest and the unclear (a photo without description, a link, a video). Removed the "cheap way to show presence" line (#23).
- New slot `{reply_to}`: the message the author replied to, even if it is older than the `context_messages` window. If the reply target is not visible (purged, not yet recorded), the model gets "a reply to a message that is not visible" and declines instead of reading the reply as a standalone phrase (#23).
- `"emoji": "null"` as a string is parsed as a refusal, not as an emoji outside the list (#23).

## [0.6.1] - 2026-10-12

### Notes

- Ambient replies no longer get an empty situation: `prompt.SITUATION_AMBIENT` says respond to the last message, to the topic, or turn it, and do not start your own story (otherwise `speak=false`). Gate and probabilities unchanged (#22).
- `few_shot.yaml`: talk about self only in answers to questions about the persona; answers to things said to someone else are rewritten as a reaction about the interlocutor; three such examples and one silence example added (21 instead of 17). After deploy PromptStore creates a new few-shot version (#22).
- Diary: the `{diary}` slot header is neutral ("Your affairs today ... do not bring it up yourself"), `prompts/diary.txt` rejects memories, "usually happens" and exaggerations. Wrong old facts can be removed with `/diary rm N` (#22).
- `/status` metric: "about self for 3 days: ambient X/Y, addressed X/Y, other X/Y"; markers are `filters.self_markers`. Cuts nothing (#22).

## [0.6.0] - 2026-10-09

### Notes

- Reactions: `reactions.max_tokens` 40 -> 120 (the model answer was truncated, 163 of 262 answers did not parse); default `emoji` `["👍", "💩", "🤣"]` because 😂 is not in Telegram's reaction set (`REACTION_INVALID`). `reactions.emoji` is now validated against `TELEGRAM_REACTIONS`, not `filters.allowed_emoji`; a broken model answer is visible in a WARNING (#21).
- Replies and `@` mentions pass under the presence cap (`gate.PRESENCE_EXEMPT`); name address does not (#21).
- `topic_cooldown` is checked only for ambient: direct addresses see it neither in the gate nor in recheck. A stop-topic message is still `gate:topic`; the stop list is unchanged (#21).
- A direct address hitting `llm:http`/`llm:timeout` is postponed once by 2-5 minutes (`send:llm_retry`) instead of going silent (#21).
- Motif `factory` added to `filters.motifs`; the prompt gained a "do not steer the talk to yourself" line; the checkin prefilter uses the same criteria as the main model (16 of 16 "yes" over two weeks ended in silence from the main model) (#21).
- Daily diary: `behaviour.diary.*`, table `self_facts` (migration 4), a cheap model extracts a fact about the persona from every text reply (counter `diary_calls`), slot `{diary}` after `{life}`, `/diary [rm N]`, a `/status` line, `DIARY_PROMPT_PATH` (#21).
- Chat jokes and callbacks: `behaviour.jokes.*`, `behaviour.callback.*`, tables `chat_jokes`/`people_threads` (migration 5), two extra calls in the weekly summary (`prompts/memory_jokes.txt`, `prompts/memory_threads.txt`), slot `{jokes}` after `{chat_memory}`, trigger `callback` (`send:callback`), `/jokes [rm N]`, `/people [rm N]`, a `/status` line. Notes about people are kept up to `callback.max_age_days`; muted users are excluded (#21).
- Long-memory safety: diary facts, jokes and threads that trip `injection`/`model_talk`/assistant markers are not saved (`Patterns.unsafe_note`); a thread is not attached to a namesake; a new address is not collapsed into a pending item whose generation is already running (previously this could double-reply) (#21).
- Silence thresholds relaxed after reviewing two quiet days (1 reply per 68 messages): `live_talk.min_messages` 3 -> 2, `live_talk.window_min` 10 -> 20, `ambient_probability` 0.15 -> 0.25, `stickers.min_replies_between` 4 -> 2; `gate:not_live` added to `REACT_REASONS` (the most common refusal reason, 54 of 68, previously gave not even a reaction); the reaction prompt no longer says "in most cases no" (the model refused 4 of 4). The presence cap (15% + 2) is unchanged and stays as a safety net (#21).
- Delayed, meaningful reactions: `behaviour.reactions.delay_sec` (20-120 s), `semantic: true` (a cheap model picks the emoji instead of the dice), `model` (empty -> `llm.judge_model`), `max_tokens`, `semantic_daily_cap` (40, counter `reaction_calls`), `context_messages` (6); default `emoji` `["👍", "💩", "😂"]` (#16).
- `probability` only applies with `semantic: false` (the old behaviour); prompt `prompts/reaction.txt`, path `REACTION_PROMPT_PATH` (#16).
- `filter_log` gained `react:recheck` (conditions diverged during the pause) and `react:declined` (the model said no); `/status` shows `semantic calls <N>/<cap>` (#16).
- `behaviour.hot_window.minutes` defaults to 10 (was 30) (#15).
- Weather as background in the system prompt: slot `{weather}` (after `{chat_memory}`, before `{life}`), source Open-Meteo (public API, no key), settings `behaviour.weather.*` (`enabled`, `ttl_min`, `timeout_sec`). Home coordinates live only in `.env`: `WEATHER_LATITUDE`, `WEATHER_LONGITUDE`, `WEATHER_HOME_NAME` (the repo is public, `config.yaml` has none); if unset there is no weather block and a startup WARNING.
- Weather for a place named in the question: `behaviour.weather.place_lookup`/`lookup_model` (empty -> `llm.judge_model`)/`lookup_max_tokens`/`lookup_daily_cap` (counter `weather_calls`)/`geocode_ttl_days`, trigger regexes `filters.weather_request`, prompt `prompts/weather_place.txt` (path `WEATHER_PLACE_PROMPT_PATH`), Open-Meteo geocoder. Called only on a direct address; `/why` shows `weather:place`, `weather:no_place`, `weather:not_found`. `/status` shows a weather line with the update time.
- Name in cases: new module `morph.py` (`name_pattern`/`name_forms`); `patterns.name_trigger` compiles each trigger from `persona.name_triggers` into all singular cases using a closed list of endings, while false positives (words that merely start with the same letters) still do not match. The letter "yo" is not normalised in code: forms are built for both spellings separately (#19).
- A name address is confirmed by a cheap model: `behaviour.followup.name_check` (true), `behaviour.followup.name_check_max_tokens` (60), prompt `prompts/name_check.txt` (path `NAME_CHECK_PROMPT_PATH`), `FollowupChecker.check_name`. Uses the same `followup_calls` counter and the same `followup.daily_cap`; if the check cannot run (no model, `LLMError`, not JSON) the message counts as an address and gets a reply. A reply to the bot and `@username` are never checked by the model (#19).
- `/why` shows `followup:name_yes` and `followup:name_no`; `followup:name_no` is in `reactions.REACT_REASONS`, so a reaction is possible instead of a reply (#19).
- Cheap prefilter before "come back to check": `behaviour.checkin.prefilter`/`prefilter_max_tokens`, `FollowupChecker.check_batch`, prompt `prompts/checkin_prefilter.txt`; `send:checkin_prefilter_no`/`send:checkin_prefilter_yes` in `/why`. Saves main-model calls when nobody wrote anything addressed since the last reply (#17).

## [0.5.0] - 2026-09-16

### Notes

- Presence cap `behaviour.presence` (15% of human messages + 2 free), `gate:presence_cap`, `send:presence_cap`; `/status` shows `presence` (#12).
- Pause `behaviour.min_gap_sec` (300); the hot window opens only after `/life` and `/say` (`hot_window.open_on_any_reply: false`) (#12).
- "Come back to check" waits for quiet (`checkin.quiet_min` 10) and re-verifies addressing with a cheap model (#12).
- `filters.enforce_stages: [dedup, style]`: these stages cut even with `shadow: true` (#12).
- Props and stories: `filters.motifs`, slot `{avoid}` in the prompt, `dedup:motif`, `style:story_quota`; prompt v10, few-shot v6 (#12).
- `/status`: an expired stop is shown as "none" (#13).
- `/changelog [owner|<version>]` in the DM, version in `/status`, `CHANGELOG.md` and SemVer tags (#14).

## [0.4.0] - 2026-09-15

### Notes

- `/life <text>`: news goes to the chat at once and into memory for good, slot `{life}`; `/life list | rm | post`; `/say <text>` verbatim (#10).
- Attention rhythm: hot window `behaviour.hot_window`, cheap "is this for me?" check `behaviour.followup`, return visit `behaviour.checkin`; `Trigger.FOLLOWUP`, triggers `life`, `say`, `checkin` (#10).
- Long chat memory `behaviour.chat_memory`: weekly summaries, slot `{chat_memory}`, `/memory list | rm | run` (#11).
- Photo vision `behaviour.vision`: the description goes into `messages.text` as `[photo: ...]`, own cap `vision_calls` (#11).
- `LLMClient.call` gained `counter_key`/`calls_cap`; DB migrations up to `user_version` 3 (#10, #11).

## [0.3.0] - 2026-09-14

### Notes

- Stickers: a second call to a cheap model, catalog `stickers.yaml`, `stickers_fill.py`, `behaviour.stickers` (#7).
- Post-processing `postprocess.py`: typographic dashes -> hyphen, frequent emoji removed before the filter; `style:emoji_position` removed; prompt v7 (#8).
- `/get <key>` shows a card with description, type and range from `Field(description)`; `/help`; chat commands in the help text (#9).

## [0.2.0] - 2026-09-12

### Notes

- Persona: a biography canon, humour and playing along, few-shot rewritten (#1, #3, #5).
- Emoji reactions on a dice-based filter cut, `behaviour.reactions` (#6).
- `/resume` resets the LLM circuit breaker; a runbook for the OpenRouter limit (#4). The image has no uv at runtime (#2).

## [0.1.0] - 2026-09-10

### Notes

- Plan stages 1-7: skeleton and DB, gate and replay, generation and sending, a three-layer output filter, venues, control from Telegram, operations. Deployed to a VPS through GitHub Actions.

[Unreleased]: https://github.com/tslava/sotto/compare/v0.9.1...HEAD
[0.9.1]: https://github.com/tslava/sotto/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/tslava/sotto/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/tslava/sotto/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/tslava/sotto/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/tslava/sotto/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/tslava/sotto/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/tslava/sotto/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/tslava/sotto/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/tslava/sotto/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/tslava/sotto/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/tslava/sotto/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/tslava/sotto/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tslava/sotto/releases/tag/v0.1.0
