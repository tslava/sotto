# Allowed golden diffs (demo pack)

No deliberate diffs. A change of the pack that moves a snapshot is re-recorded together with the
change, with the diff reviewed:

    uv run python -m tools.golden record --pack personas/demo \
        --scenarios personas/demo/golden/scenarios.yaml --out personas/demo/golden/baseline

`/status` prints the package version, so `E_status` and `E_panic_resume` differ after a release
bump when compared from the command line. `tests/test_demo_pack.py` pins the version the baseline
was recorded with, so the test run is unaffected.
