# Next experiments

Status: none scheduled. Nothing runs automatically; every run is a manual,
foreground start on an idle GPU 0 (see `docs/TESTING.md`).

## Candidate experiments (not yet run)

1. **Short-decode benchmark (baseline)** — fixed prompt, forced output at
   concurrency 1/4/8; record client-observed tokens/s. There is no throughput
   number recorded yet, so this comes first.
2. **DFLASH on/off A/B** — same benchmark with `--speculative-algorithm DFLASH`
   removed; compare acceptance behavior and tokens/s. Candidate file:
   `configs/docker/candidates/qwen27b-no-spec.sh`.
3. **KV dtype comparison** — `fp8_e4m3` vs `bf16` KV: capacity vs decode speed.
4. **`--mem-fraction-static` sweep** — 0.80 / 0.85 / 0.90: KV token capacity vs
   headroom.
5. **Mamba cache strategy** — `extra_buffer` vs `auto` / `no_buffer`.
6. **Long-context retrieval** — near-limit chat-formatted synthetic prompt
   (e.g. 200k tokens) with a test code in the middle; verify recall and that
   nothing is truncated.
7. **Concurrency behavior** — up to `max_running_requests` (45 recorded) with
   short prompts; watch queueing and retractions.

## Rules for running them

- One experiment at a time, on an idle engine; capture `/server_info` before
  and after.
- New configs go to `configs/docker/candidates/`; each validated result gets an
  evidence file in `evidence/` named by profile and UTC timestamp.
- Never overwrite evidence; failed probes are retained and labeled.
- No systemd, no restart policies, no retry loops — if a run takes the VM down,
  it stays down until a human retries.
