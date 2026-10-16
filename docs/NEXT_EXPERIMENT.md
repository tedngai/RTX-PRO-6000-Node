# Next experiments

Status: **approved experiments and the 2026-10-16 backlog are measured**
(see `results.md`). The live config remains the promoted launcher
(DFLASH D=4 + ratio 2.0, `--max-running-requests 12`, KV pool ~708k —
`evidence/recommended-c12-validation-20261016T040048Z.md`). None of the
backlog candidates were promoted. An intervening GPU 0 vGPU wedge is
documented in `evidence/gpu0-wedge-20261016.md`. Nothing runs
automatically; every run is a manual start on an idle GPU 0 (see
`docs/TESTING.md`).

## Capacity context (measured 2026-10-16, from source + startup log)

The KV pool (181,925 tokens) is small by configuration, not by hardware:
`--mamba-full-memory-ratio 6.08` gives the mamba state pools 6.08× the KV
memory (42.8 GB vs 6.0 GB), and DFLASH's `--speculative-num-draft-tokens 8`
creates a 25.9 GB intermediate-SSM pool (46 slots × 8 draft tokens × ~70 MB).
The 45-request cap is `max_mamba_cache_size 225 ÷ 5 slots/request`
(extra_buffer strategy). Lowering the ratio / draft count moves memory into
the shared KV pool.

## Approved next experiments, in order

1. **Speculative-decode A/B — DONE (2026-10-16).** Three arms measured with
   identical prompts. Winner: **DFLASH with `--speculative-num-draft-tokens 4`**
   (KV pool 181,925 → 307,769 tokens, cap 45 → 48, ≤6% aggregate loss at
   saturation). EAGLE/MTP 3/1/4 gives the largest pool (376,415) but is
   20–30% slower at mid load. See `results.md` for the full table and
   `configs/docker/candidates/qwen27b-dflash-d4.sh`,
   `configs/docker/candidates/qwen27b-mtp-eagle-314.sh`.
2. **Mamba/KV rebalance — DONE (2026-10-16).** `--mamba-full-memory-ratio`
   2.0 on top of DFLASH-4: KV pool 307,769 → **442,745 tokens**, request cap
   holds at 48 (mamba states 254; 254÷5 = 50 > 48), speed within run-to-run
   noise of DFLASH-4. Promoted to the main launch script
   `configs/docker/run_qwen27b_sglang.sh`. Evidence:
   `evidence/dflash-d4-r2-bench-20261016T024247Z.json`,
   `evidence/dflash-d4-r2-server-info-20261016T024552Z.json`.

Both approved experiments are complete. The backlog below was run on
2026-10-16 and left unpromoted, except the second GPU, which is still a
policy decision.

## Backlog (not scheduled)

- `--mem-fraction-static` 0.85 → 0.90 + CUDA-graph batch-size trims —
  **measured 2026-10-16, not promoted**. KV pool +10% (708,463 → 779,104),
  headroom after init 10.89 → 6.95 GB, benchmark 9/9 with no OOM. Short of
  the ~12–15% estimate. `evidence/mf90-graphtrim-20261016T042118Z.md`.
- `--mamba-radix-cache-strategy extra_buffer_lazy` (4 slots/req) —
  **measured 2026-10-16, not promoted**. Effective cap 57 (requested 80;
  `/server_info` still shows 80), prefix reuse intact (2,688/2,704), KV
  pool 425,330 because the DFLASH intermediate pool grew to 16.31 GB.
  `evidence/mamba-lazy-20261016T042627Z.md`.
- `no_buffer` (3 slots/req) — **measured 2026-10-16, not promoted**.
  Requires `--disable-overlap-schedule` (first launch failed arg
  validation; GPU stayed up). Effective cap 65 (requested 96), KV pool
  427,154, prefix reuse only up to 2048-token chunk boundaries
  (2,048/2,704 and 6,144/8,093). `evidence/mamba-nobuffer-20261016T043240Z.md`.
- Long-context retrieval quality — **DONE (2026-10-16)** on the promoted
  c12 server, no relaunch. Exact quote of a planted function at 4k and at
  238k tokens (91% of the 262,144 context), needle at 0.10 / 0.50 / 0.90.
  Prefix reuse on `extra_buffer` cached 2,688/2,704 tokens.
  `evidence/recommended-c12-longctx-20261016T041501Z.md`.
- Second GPU (GPU 1) as an independent replica — policy decision, 2× capacity.

## Rules for running them

- One experiment at a time, on an idle engine; capture `/server_info` before
  and after.
- New configs go to `configs/docker/candidates/`; each validated result gets an
  evidence file in `evidence/` named by profile and UTC timestamp.
- Never overwrite evidence; failed probes are retained and labeled.
- No systemd, no restart policies, no retry loops — if a run takes the VM down,
  it stays down until a human retries.
