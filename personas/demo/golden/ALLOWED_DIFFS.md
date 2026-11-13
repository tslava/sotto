# Allowed golden diffs (demo pack)

No deliberate diffs. A change of the pack that moves a snapshot is re-recorded together with the
change, with the diff reviewed:

    uv run python -m tools.golden record --pack personas/demo \
        --scenarios personas/demo/golden/scenarios.yaml --out personas/demo/golden/baseline

`/status` prints the package version; the golden adapter pins it (`PINNED_VERSION` in
`tools/golden/adapter.py`), so a release bump does not move the baseline.
