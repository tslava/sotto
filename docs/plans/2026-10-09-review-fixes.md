# Plan: review of 24.09-09.10, fixes and "more interesting"

**Owner decisions from the review (09.10):** the stop-list is left alone (A3 item 2, option "c"); the ambient situation
"answer the last message" is not added; the "talking about himself" quota is prompt-only; A2 covers REPLY + MENTION; everything goes in one PR
as separate commits, merge and deploy are done by the lead agent. The contract is CLAUDE.md, three new sections.

The basis is a review of the production database over 15 days (1029 human messages, 52 bot replies, 0-11% a day,
38% of replies got an answer, about $0.19 a day). Presence and variety after the 15.09 measures work;
reactions are broken, direct addresses get lost, "came back to check" yields nothing, and new
stage props appeared (a factory). Tasks follow. **[decision]** marks places where the owner must choose.

"Before" metrics for comparison after the rollout:

| Metric | Now |
|---|---|
| Reactions sent | 0 in 21 days (262 model calls) |
| Addresses lost, not at night | 6 (presence_cap 1, topic_cooldown 2+, llm:http 3) |
| checkin: prefilter_yes -> sent | 16 -> 0 |
| Replies with "factory/line/workshop/shift" | ~8 of 52 + 3 cut candidates |
| Replies starting with "I / I have / We have" | 14 of 52 |
| Silence reason no. 1 | `gate:not_live`: 376 |

---

## Part A: fixes

### A0. Roll out the `loosen` branch

Commit 4c60210 from 22.09 (live_talk 3/2/10 -> 2/2/20, ambient_probability 0.15 -> 0.25,
stickers.min_replies_between 4 -> 2, `gate:not_live` in REACT_REASONS, a softer reactions prompt)
never reached main. `gate:not_live` is still the main reason for silence.

- Rebase onto main, run ruff/mypy/pytest, PR, merge.
- The 15% + 2 presence ceiling is left alone: it protects against "too much of the bot".

### A1. Reactions: bring them back to life

Three independent causes, fix all three:

1. **A truncated answer.** 163 of 262 answers were `invalid answer`. `reactions.max_tokens: 40`, while the model
   writes `reason` in Russian at 100+ characters and the JSON is cut off.
   - `max_tokens` 40 -> 120 (default in `config_models.py` and `config.yaml`).
   - In `prompts/reaction.txt`: "reason is at most five words".
   - `_parse_choice` returned None -> log a WARNING with the first 80 characters of the model's answer (without participant
     data, since the model's answer paraphrases them anyway; truncate hard), so that the next failure
     is visible at once.
2. **The laughing-with-tears emoji cannot be set as a reaction.** Both times the model said "yes" it was that emoji ->
   `REACTION_INVALID`. The Telegram reaction list has the rolling-on-the-floor-laughing emoji, and not that one.
   - `reactions.emoji`: `["👍", "💩", "🤣"]`.
   - Reaction validation is decoupled from `filters.allowed_emoji` (that list is for TEXT):
     `reactions.py` gets a constant `TELEGRAM_REACTIONS: frozenset[str]` (the standard Bot API
     set), and a `model_validator` checks `reactions.emoji ⊆ TELEGRAM_REACTIONS`.
     In the prompt: "🤣 means funny".
3. **The call ceiling is spent on junk.** On busy days `semantic_daily_cap: 40` ran out by
   evening. After item 1 there will be no junk; leave the ceiling at 40 and watch `/status`.

Tests: `test_reactions.py`: parsing a long reason; the laughing-with-tears emoji outside TELEGRAM_REACTIONS is rejected by the
validator; the rolling-on-the-floor one passes; a WARNING with the truncated answer. `test_config.py`: the validator.

### A2. An `@` address does not go deaf under the presence ceiling

On 26.09 a question to the bot via `@username` -> `gate:presence_cap`, silently.
Currently only `Trigger.REPLY` passes under the ceiling.

- **[decision]** Let through under the ceiling: (a) REPLY + MENTION (recommended: `@` is an unambiguous
  address), (b) REPLY + MENTION + NAME (the name is often about him in the third person, which is riskier).
- `gate.py` step 6 and `responder._recheck` (`send:recheck_presence`) use the same set
  of triggers; extract it into a `PRESENCE_EXEMPT` constant.
- Tests: `test_gate.py`, `test_responder.py`.

### A3. A stop topic does not silence addresses; a stop-list without everyday matches

`gate:topic_cooldown` (45 minutes after a stop topic) ate 80 messages over 15 days, including
addresses (a photo with a caption addressed to the bot, a reply on 08.10 cut by `send:recheck_topic`). Half of
the stop-list matches were everyday mentions of the country (a phone number, a cinema release, a news item
about a robot). On 06.10 the bot was silent all day.

1. Addresses (reply/mention/name/followup) are not checked against `topic_cooldown`, neither in the gate nor in
   recheck. A message containing a stop topic itself is still DROP `gate:topic`: he never answers
   about politics, even to a direct address. The output filter `regex:topic` stays.
   Step order in `gate.py`: step 5 (`topic_cooldown`) moves into the ambient branch.
2. **[decision]** The stop-list:
   - (a) remove the country-abbreviation regex and the country-adjective regex from `topic_stop` (recommended: the other markers,
     war, front, the head of state, mobilisation, sanctions..., catch politics without them);
   - (b) keep them but without a cooldown: a new list `filters.topic_soft`: DROP of this message only,
     `topic_cooldown_until` is not set;
   - (c) change nothing.
3. Tests: `test_gate.py` (an address in cooldown passes; a message with a stop topic and an address is
   DROP; ambient in cooldown is DROP), `test_responder.py` (recheck does not cut an address by cooldown),
   `test_patterns.py` for option (b).

### A4. New stage props and "me too"

The wife and the greenhouse are gone, and a factory has taken their place: the line, the workshop, the shift
foreman, the fitter, the welders. And a general manner: almost every reply steers the conversation to himself.
Ambient often answers the topic as a whole rather than the last message.

- `filters.motifs`: + a `factory` key with word-boundary regexes (Russian stems for factory, line, workshop,
  shift, fitter).
  In `motifs.py` the declension dictionary: factory -> "factory".
- `prompts/system.txt`, the "HOW YOU WRITE" block: "Do not steer the conversation to yourself. More often
  answer what the person said than tell your own story. 'Me too...' is no more than once every few
  replies." CHARACTER.md section 3 gets the same line.
- ~~Ambient situation "answer the last message"~~: rejected by the owner: butting in on the topic in general is fine too.
- **[decision]** A "talking about himself" quota in the filter (`style:self_quota`: the reply starts with
  "I/I have/We have/To me/Me" and the same start is in 2 of the last 3 replies -> cut), or
  the prompt alone is enough. Recommendation: the prompt only at first, the filter if a week later it did not help.
- Tests: `test_motifs.py`, `test_prompt.py` (SITUATION_AMBIENT), `test_responder.py` (ambient
  gets a situation with the trigger text).

### A5. "Came back to check": align the prefilter with the main model

66 runs, 0 sends; 16 times the prefilter said "there is something" and Opus stayed silent (14 times `llm:silent`,
1 `checkin_not_addressed`). The cause is different criteria: the prefilter lets through "directly continue
the topic he raised", while after 15.09 the main model answers only to what is explicit.

- `prompts/checkin_prefilter.txt`: remove "or directly continue the topic he raised"; the same
  three signs as in `SITUATION_CHECKIN_FOOTER` (by name, a question to him, a reply to his message),
  plus "if in doubt, an empty list".
- After a week, via `/why`: `checkin_prefilter_yes` ≈ `send:checkin`. If sends are still zero,
  turn `behaviour.checkin.enabled` off by default (as a separate decision).
- Tests: the prompt template is not tested; `test_followup.py` does not change.

### A6. An address is not lost to an OpenRouter failure

On the evening of 24.09 there were 10 `llm:http` errors in a row, and three addresses were lost.
The LLMClient contract "no retries" is left alone; the responder does the deferral.

- `_generate_and_send` / `_fire_pending`: for addresses on `llm:http` or `llm:timeout`, a single deferral of the pending to `now + randint(120, 300)`
  (a `retried` flag in `_pending_info`, not needed in the DB:
  after a restart the deferral simply will not repeat). A second failure means silence as now.
  ambient/checkin/morning get no deferral.
- filter_log: `send:llm_retry`.
- Tests: `test_responder.py` (the first `llm:http` -> deferral; the second -> silence; ambient is not
  deferred; circuit_open is not deferred).

### Not doing

- A false trigger on the name (a participant was talking about their own father): one case, the reply
  turned out funny and the chat picked it up. We keep watching.
- Post-processing stripped emoji 37 times: it works as designed.

---

## Part B: more interesting

All three are **[decision]** items about character and privacy; the recommended form is below.

### B1. A diary of his own day

On 26.09 the bot said via `/say` that he was going to the sauna, and 7 minutes later that he was sitting on a fishing
trip. The chat caught the contradiction in time. A reply about mushrooms was in "Your recent
replies", but the model does not hold it as a fact.

- After every successful send (the common tail of `_generate_and_send`, and `say`) a cheap model
  (`followup.model or judge_model`, its own counter `diary_calls`, a ceiling of 30/day) extracts from the
  reply a fact about his own day/affairs: `{"fact": "went to the sauna" | null}`.
- A `self_facts (id, text, created_at)` table, migration 4. Retention is 7 days.
- A `{diary}` slot in system.txt after `{life}`: "Today you have already said about yourself (stick to it,
  do not contradict): 10:46 went to the sauna; 12:39 tinkered with the boat". Only facts of the current
  local day; for the past 6 days a single line "This week: ..." (storylines: the fence,
  paint in the garage, continue on their own).
- `/status`: "diary: N facts today". `/diary` in the private chat is a list and `rm N`.
- Tests: the extractor (JSON/null/failure -> nothing), slot rendering, the slot in system.

### B2. Memory about people and the question "so how did it go?"

What makes a friend a friend is that a couple of days later he asks how it ended. A weekly memory
already exists, but it is about the chat as a whole.

- `prompts/memory.txt` is extended with a second answer block: "Unfinished stories of people: who is
  going to do something or is waiting for something (an interview, a renovation, a trip), one line per person, with no
  judgements". ChatMemorizer parses two blocks; a table `people_threads (id, user_id, display_name,
  text, created_at, asked_at, closed)`. Muted users are excluded (as now).
- A new trigger `callback`: no more often than once per `callback.min_days` (3) days, at waking hours, in
  quiet (`checkin.quiet_min`), under the presence ceiling, the thread's author wrote in the chat in the last
  7 days. Situation: "A couple of days ago {name} said: '{text}'. Ask how it ended, in one
  short phrase, in a friendly way. If asking is inappropriate, stay silent." The reply has no reply-to,
  with the name. After that `asked_at` is set and the thread is closed.
- A `behaviour.callback` section (enabled, min_days, max_age_days 14). `/people` in the private chat: a list,
  `rm N`, `off`.
- **Privacy**: a retelling about specific people is stored longer than the conversation: the same decision as for
  chat_memory, but more targeted. An explicit "ok" is needed.

### B3. The chat's inside jokes

The chat gives birth to its own memes and words. Bringing one back in context a week later says "he is one of us" more than anything.

- The same weekly ChatMemorizer call: a third block "Jokes and words the chat picked up (up to three,
  verbatim, with no explanations)". A table `chat_jokes (id, text, created_at, last_used_at,
  uses)`, retained for `keep_days`.
- A `{jokes}` slot in system.txt: "Your chat's jokes (you may bring one back in context, no more than once every
  few days): ..." Only those not used within `jokes.cooldown_days` (5).
- Use is detected after sending: 3+ shared normalised words with a joke -> `last_used_at`.
  No extra model call.
- `/jokes` in the private chat: a list, `rm N`.

---

## Order and PRs

Outcome of the review: everything in one PR, as separate commits, version 0.6.0. B2/B3 were done not as blocks in
`memory.txt` but as two separate model calls in the same weekly ChatMemorizer (prompts
`memory_jokes.txt`, `memory_threads.txt`), migration 5. The table below is the original plan.

| PR | Tasks | Version |
|---|---|---|
| 1 | A0 `loosen` | 0.5.1 |
| 2 | A1 reactions | 0.5.2 |
| 3 | A2 + A3 + A6: addresses are not lost | 0.6.0 |
| 4 | A4 + A5: voice and checkin | 0.6.1 |
| 5 | B1 diary of the day | 0.7.0 |
| 6 | B3 chat jokes | 0.8.0 |
| 7 | B2 people and "so how did it go?" | 0.9.0 |

Each PR: CHANGELOG Unreleased (both blocks), CLAUDE.md with the contract of the new interfaces, README where relevant. Sonnet
agents code to the contract, then a reviewer, then fixes.

Check a week after PRs 2-4 (the same query over `bot_replies` / `filter_log`): 1-4 reactions a
day; zero addresses lost in the daytime; `checkin_prefilter_yes` ≈ `send:checkin` or the feature is off;
"factory" ≤ 1 per 10 replies; "I / I have" at the start ≤ 1 of 5; the bot's share 5-12%.
