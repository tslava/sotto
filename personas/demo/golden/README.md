# Demo golden

- `scenarios.yaml`: 101 scenarios (groups A to F),
  written for Gus Marlow's chat. All text is invented.
- `baseline/`: what the engine produced for them, recorded with `tools.golden record`. Never
  re-record to hide a diff.
- `export.json`: a small synthetic Telegram Desktop export (three days, six people) for
  `python -m sotto.replay`.
- `ALLOWED_DIFFS.md`: deliberate diffs (none).

The comparison also runs in `tests/test_demo_pack.py`.
