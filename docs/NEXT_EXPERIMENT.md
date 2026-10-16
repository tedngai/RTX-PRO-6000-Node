# Next experiments

Status: **baseline benchmark done (2026-10-16, see `results.md`)**. The
approved order is: baseline → speculative-decode A/B → mamba/KV rebalance.
Nothing runs automatically; every run is a manual start on an idle GPU 0
(see `docs/TESTING.md`).

## Capacity context (measured 2026-10-16, from source + startup log)

The KV pool (181,925 tokens) is small by configuration, not by hardware:
`--mamba-full-memory-ratio 6.08` gives the mamba state pools 6.08× the KV
memory (42.8 GB vs 6.0 GB), and DFLASH's `--speculative-num-draft-tokens 8`
creates a 25.9 GB intermediate-SSM pool (46 slots × 8 draft tokens × ~70 MB).
The 45-request cap is `max_mamba_cache_size 225 ÷ 5 slots/request`
(extra_buffer strategy). Lowering the ratio / draft count moves memory into
the shared KV pool.

## Approved next experiments, in order

1. **Speculative-decode A/B** — current DFLASH (D=8, DFlash2 draft) vs
   **EAGLE/MTP with the in-checkpoint 1-layer MTP head**
   (`--speculative-algorithm EAGLE --speculative-num-steps 3
   --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, no draft
   download; vendor-recommended recipe). Optional third arm: NGRAM.
   Expected effect: the intermediate-SSM pool halves (25.9 → ~13 GB), the
   5-layer draft's ~4-5 GB (weights + KV + graphs) is freed, and the request
   cap rises; the big KV gain still needs experiment 2.
   Candidate file: `configs/docker/candidates/qwen27b-mtp-eagle-314.sh`.
   Compare with `tests/benchmark.py --profile <label>` (identical seeds).
2. **Mamba/KV rebalance** — `--mamba-full-memory-ratio` 6.08 → ~2 (or pin
   `--max-mamba-cache-size` explicitly). Expected: KV pool 182k → ~500k
   tokens; concurrency cap roughly holds (K grows as D shrinks). Run on top
   of the experiment-1 winner; validate stability under load.

## Backlog (not scheduled)

- `--mem-fraction-static` 0.85 → 0.90 + CUDA-graph batch-size trims (~+12-15%
  to every pool number; validate headroom under load).
- `--mamba-radix-cache-strategy extra_buffer_lazy` (4 slots/req) or
  `no_buffer` (3 slots/req) — raises the request cap; verify prefix-reuse
  behavior.
- Long-context retrieval quality (near-limit chat prompt, test code in the
  middle).
- Second GPU (GPU 1) as an independent replica — policy decision, 2× capacity.

## Rules for running them

- One experiment at a time, on an idle engine; capture `/server_info` before
  and after.
- New configs go to `configs/docker/candidates/`; each validated result gets an
  evidence file in `evidence/` named by profile and UTC timestamp.
- Never overwrite evidence; failed probes are retained and labeled.
- No systemd, no restart policies, no retry loops — if a run takes the VM down,
  it stays down until a human retries.
