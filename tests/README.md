# Tests

Host-side validation scripts. They exercise a live, ready server and may write
reports under `evidence/`; they never start or stop anything.

- `smoke_test.py` — checks `/health`, picks the served model from `/v1/models`,
  sends one deterministic chat completion, and optionally saves an evidence
  report. Python standard library only.

```bash
python3 -m py_compile tests/*.py
python3 tests/smoke_test.py --base-url http://127.0.0.1:30000
python3 tests/smoke_test.py --save-dir evidence     # also write the report JSON
```

A benchmark script for the experiments listed in `docs/NEXT_EXPERIMENT.md` is
not written yet.
