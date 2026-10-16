# Tests

Host-side validation scripts. They exercise a live, ready server and may write
reports under `evidence/`; they never start or stop anything.

- `smoke_test.py` — checks `/health`, picks the served model from `/v1/models`,
  sends one deterministic chat completion, and optionally saves an evidence
  report. Python standard library only.
- `benchmark.py` — concurrency benchmark (TTFT, decode speed, aggregate
  throughput, spec acceptance). Never starts or stops the server.
- `long_context_retrieval.py` — needle-in-a-haystack at a near-limit chat
  prompt. Does not store the haystack in the evidence file.
- `prefix_reuse_probe.py` — checks radix/mamba prefix-cache hits on a shared
  prefix versus an unrelated one.
- `gen_manifest.py` — regenerates `docs/package-manifest.json`.

```bash
python3 -m py_compile tests/*.py
python3 tests/smoke_test.py --base-url http://127.0.0.1:30000
python3 tests/smoke_test.py --save-dir evidence     # also write the report JSON
```

A benchmark script for the experiments listed in `docs/NEXT_EXPERIMENT.md` is
not written yet.
