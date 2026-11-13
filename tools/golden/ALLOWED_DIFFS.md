# Allowed golden diffs

- E_status: PR-1 rename, the first line of /status prints the package name (`trolobot v...` -> `sotto v...`).
- E_panic_resume: PR-1 rename, the embedded /status output prints the package name.
- D_flat_dump: PR-8a, the unused config field `persona.arrived_wolsan` was deleted (owner decision), so the flat dump lost that one line.
- D_flat_dump_overridden: PR-8a, same: `persona.arrived_wolsan` is gone from the dump.
- D_flat_dump_names: PR-8a, same: `persona.arrived_wolsan` is gone from the dump.
- E_get: PR-8a, `/get` no longer lists `persona.arrived_wolsan` (the 3500-character cut moves by that line).
