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

## Caveats recorded during bring-up

- A 16-token request returned empty `content` with only `reasoning_content` —
  the model spends tokens on reasoning first; give short Q&A ≥ ~100–256 tokens.
- `/health` returned 503 for a moment immediately after "Application startup
  complete", then 200 on the next check.

## Not measured yet

- Throughput/latency under load (any concurrency), DFLASH speedup, long-context
  behavior, output quality. See [`NEXT_EXPERIMENT.md`](NEXT_EXPERIMENT.md).
