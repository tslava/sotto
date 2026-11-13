# Golden harness

Records, for a persona pack, exactly what the engine sends to the models, how it parses and
persists, which config it builds and what the owner UI says; `compare` shows any drift later.
It exists to prove that the stage 2/3 refactoring (persona split, English) is lossless for the
real pack. Plan: `04-stage2-3-plan.md`, section 3.2.

```
uv run python -m tools.golden record  --pack DIR --scenarios FILE --out DIR [--only ID|GROUP ...]
uv run python -m tools.golden compare --pack DIR --scenarios FILE --baseline DIR \
        [--allow ALLOWED_DIFFS.md] [--only ID|GROUP ...]
```

A *pack* directory is a persona pack as `sotto.persona.load_persona` reads it (PR-2): `pack.yaml`,
`config.yaml`, `few_shot.yaml`, `prompts/*.txt`, `CHANGELOG.md`, `stickers.yaml`, `places_manual.yaml`
(the repo's own pack is `personas/_legacy_ru`). Output: one canonical JSON per scenario
(`sort_keys`, `indent=1`, `ensure_ascii=False`). `compare` prints a unified diff per scenario and
exits 1; `--allow` takes a markdown file with lines `- <scenario id>: reason` (deliberate changes,
owner sign-off). `--only` accepts scenario ids or group letters and skips the per-group minimums.

## Determinism

Virtual clock (default `2026-11-04 12:00` local time of the pack, a Wednesday), `time.time` and
`asyncio.sleep` replaced for the run; `random.Random(1)`; LLM, weather and geocoder behind
`httpx.MockTransport` returning the replies listed in the scenario (one queue, consumed in call
order; an empty queue answers `{"speak": false, "text": ""}`); a temp SQLite file opened through
the real `Database`; a fake Telegram bot that records every call. Sleeps of 10 s or more from
background tasks (pending-reply timers, reaction pauses) wait for an explicit release, so a scenario
controls when they fire (`gate_pass` releases them at the end, `advance` releases the ones due).
Run in separate processes with different `PYTHONHASHSEED` the output is byte-identical.

## Files

- `adapter.py`: the ONLY module that knows how to build the engine (paths, constructors, the
  engine package name) and how every action calls it. A refactoring PR updates this file only.
- `runner.py`, `canonical.py`, `schema.py`, `__main__.py`: engine-independent.
- `scenarios_ru.yaml`: synthetic Russian chat text; for the real pack it is copied to
  `sotto-soul/golden/scenarios.yaml`. `baseline_synthetic/`: recorded from the repo's own root
  files (public-safe because they are synthetic). Never regenerate a baseline to hide a diff.
- Debug aids: `GOLDEN_DEBUG=1` prints swallowed exceptions, `GOLDEN_CAPTURE_ALL=1` captures
  requests, sent messages and the database in every scenario (never for baselines).

## Scenario groups (101 scenarios)

| Group | n | What is snapshotted |
|---|---|---|
| A main prompt | 33 | the full model request for each trigger and context slot (ambient, hot window, mention collapse, reply, name, followup, late reply, morning, spontaneous, checkin, callback, life, places, weather, diary, memory, jokes, vocab, motifs, placeholders, sticker choice) |
| B auxiliary prompts | 14 | judge, followup check / name check / batch prefilter, reaction chooser, sticker chooser, vision, weather place extractor, diary extractor, memory summary + jokes + threads, `places_fill` fact prompt, `stickers_fill` OCR prompt |
| C deterministic functions | 13 | `soften`, `check_output` (27 candidates), `Patterns` probes, `sanitize`, `morph`, `Vocab.find`, motifs, `validate_fact`, places menu, weather rendering, reply parser, situations, `/ask` preview |
| D config | 8 | `flatten_config`, `describe_key` cards, compiled pattern strings, with and without overrides |
| E owner UI | 20 | the real command handlers with a fake `Message.answer`: `/help /status /get /set /unset /why /last /prompt /ex /life /say /memory /diary /jokes /people /words /changelog /ask`, chat commands, alert texts |
| F persistence | 13 | rows and state the engine writes: `messages` (placeholders, name fallbacks, injection name), `bot_replies`, `filter_log` per stage, state keys at 12:00, 23:59:30 and 00:00:30 on a day that is also an ISO-week boundary, historical `[стикер #N]` / `[фото: ...]` rows driving sticker repeats, motifs, emoji counting and vocab |

Large prompt bodies in `prompt_versions` / `few_shot_versions` are stored as sha256 + length.

## Known limits (scenarios not driven, and why)

- `app.main()` is not run (it starts aiogram polling); the adapter mirrors its wiring. PR-2's
  `packcheck --boot` is the guard for that wiring.
- `Bot` calls are recorded at the `FakeBot` level, aiogram's HTTP layer is not exercised.
- HTTP headers of the LLM request (`X-Title`) are not recorded: only url-independent body fields.
- `/memory run` (weekly job through the command), the periodic `*_job` loops and `restore_pending`
  with stale rows are not driven; their per-iteration methods are (`morning`, `spontaneous`,
  `checkin`, `callback`, `restore_pending`, `ChatMemorizer.summarize_period`).
- `/words` upload covers only the invalid-file branch (the fake download returns non-YAML bytes);
  a valid dictionary is seeded through `VocabStore.put`.
- `places_fill`/`stickers_fill` network search and sticker-set download are not driven, only their
  model prompts.
