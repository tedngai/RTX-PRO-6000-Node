# Results

Recorded results for the Qwen3.8-27B SGLang configuration
([`configs/docker/run_qwen27b_sglang.sh`](../configs/docker/run_qwen27b_sglang.sh)).
All numbers are from manual runs on 2026-10-15 on the RTX PRO 6000 Blackwell
vGPU node (GPU 0 only). These are single synthetic checks, not latency
distributions or quality evaluations.

## 2026-10-15 — startup (container `qwen27b-sglang`)

| Measurement | Value |
| --- | --- |
| Time to ready (tokenizer e2e) | **118.52 s** (≈2 min) |
| Weight load | 10.38 s |
| KV cache allocation | 2.17 s |
| CUDA graph capture | 52.77 s prefill, 9.76 s target-verify, 6.30 s draft-decode |
| KV capacity | **181,932 tokens** (fp8_e4m3) |
| Max running requests | 45 — capped by the mamba state cache (`max_mamba_cache_size=225`, 5 state slots per request) |
| Context length | 262,144 |
| Available GPU memory after init | 8.11 GB |
| GPU 0 memory in use while serving | ~85.9 GB / 96 GB |
| GPU 1 | idle (69 MiB) |

DFLASH was active: the draft runner initialized (`DFlash2DraftModel`, block
size 8), the selector decode was folded into the draft CUDA graph, and the draft
verify graph was captured for batch sizes up to 45. No acceptance-rate numbers
recorded yet.

## 2026-10-15 — smoke test

`python3 tests/smoke_test.py --save-dir evidence`:

- `/health` → HTTP 200
- `/v1/models` → `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`
- Chat completion (temperature 0): `"2 + 2 equals 4."`, `finish_reason=stop`,
  47 completion tokens (37 of them reasoning), client-observed **0.31 s**

Evidence: `evidence/qwen27b-server-info-20261015T221826Z.json`,
`evidence/qwen27b-smoke-chat-20261015T221828Z.json`.

## 2026-10-16 — baseline benchmark (current DFLASH config)

`python3 tests/benchmark.py --profile baseline-dflash8 --save-dir evidence`
against the running config (DFLASH, `--speculative-num-draft-tokens 8`,
`--mamba-full-memory-ratio 6.08`, `--mem-fraction-static 0.85`; KV pool
181,925 tokens, 45-request cap). Forced decode (`ignore_eos`, 256 output
tokens), deterministic unique prompts, **zero prefix-cache hits, zero
retractions**, all requests exact length.

| Scenario | Aggregate tok/s | decode tok/s per req (p50) | TTFT p50 (s) | accept len |
| --- | --- | --- | --- | --- |
| short c1 (≈512 in) | 204.7 | 216.6 | 0.07 | 5.33 |
| short c8 | 644.6 | 190.3 | 0.26 | 5.72 |
| short c16 | 1,475.7 | 144.4 | 0.50 | 5.85 |
| short c32 | 2,166.7 | 100.2 | 0.75 | 6.32 |
| short c45 | 2,012.3 | 69.1 | 0.88 | 6.19 |
| long c1 (≈16k in) | 121.8 | 245.9 | 1.06 | 6.10 |
| long c4 (≈16k in) | 186.8 | 106.1 | 2.87 | 6.15 |

Observations:

- Aggregate throughput peaks around c32 (~2,170 tok/s) and is slightly
  **lower** at c45 — the engine is compute-bound before the 45-request cap
  binds. Per-user decode at full load: ~69 tok/s (p50).
- 16k-token prefill: ~1.1 s TTFT at c1 (~15k tok/s prefill); 2.9 s at c4.
- Accept length on the synthetic pseudo-prose (5.3–6.3) is higher than the
  server's live average (3.15): predictable filler drafts well, real text
  will be lower. Profile A/B comparisons remain valid because every run
  sends byte-identical prompts.

Evidence: `evidence/baseline-dflash8-bench-20261016T022322Z.json`.

## Caveats recorded during bring-up

- A 16-token request returned empty `content` with only `reasoning_content` —
  the model spends tokens on reasoning first; give short Q&A ≥ ~100–256 tokens.
- `/health` returned 503 for a moment immediately after "Application startup
  complete", then 200 on the next check.

## Not measured yet

- Output quality (retrieval/tool-call correctness under load).
- See [`NEXT_EXPERIMENT.md`](NEXT_EXPERIMENT.md) for the remaining plan
  (speculative-decode A/B, mamba/KV rebalance).
