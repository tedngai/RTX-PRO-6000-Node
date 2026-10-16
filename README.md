# RTX-PRO-6000-Node — Qwen3.8-27B inference optimization

An evidence-driven SGLang deployment and measurement project for serving
Qwen3.8-27B (NVFP4 weights, FP8 KV) with DFLASH speculative decoding on a
single MIG slice of an RTX PRO 6000 Blackwell vGPU node. It records how the
container, model, KV cache, attention, and speculative-decoding settings were
chosen and what has actually been measured — not just headline tokens per second.

This repository captures one tested environment. It is a reference for
experimentation, not a turnkey distribution. The status below reflects the
latest evidence in the repository, recorded on 2026-10-16; it is not a live
check of a running server.

## Latest recorded configuration

The latest validated service snapshot is
[`configs/docker/run_qwen27b_sglang.sh`](configs/docker/run_qwen27b_sglang.sh).

| Setting | Recorded value |
| --- | --- |
| Hardware | 2× RTX PRO 6000 Blackwell DC-4-96Q vGPU (MIG 4g.96gb slices); test uses **GPU 0 only** |
| Host / image | driver 580.159.03; `lmsysorg/sglang:latest` = SGLang **v0.5.21**, CUDA 13.0.3 |
| Runtime | Docker + NVIDIA Container Toolkit; container `qwen27b-sglang`, `--restart no`, manual starts only |
| GPU selection | MIG device UUID (resolved dynamically; index selection breaks CUDA on this vGPU) |
| Model | `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` (ModelOpt mixed precision, ~22 GiB) |
| Speculative decoding | **DFLASH**, draft `incoai/Qwen3.8-27B-DFlash2`, **4 draft tokens** (was 8; A/B in `docs/results.md`) |
| KV cache | `fp8_e4m3`; **708,463 token slots** in the latest startup (was 181,932; 442,745 at the pre-cap admission of 48) |
| Attention backend | `flashinfer` (linear-attention/mamba backend: triton) |
| Hybrid (mamba) settings | `--mamba-full-memory-ratio **2.0**` (was 6.08), `--mamba-radix-cache-strategy extra_buffer`, `--mamba-ssm-dtype bfloat16` |
| Static memory fraction | `0.85` (≈10.9 GB headroom after init) |
| Context | 262,144 tokens |
| Admission | `max_running_requests` **12** (user requirement; the engine validated 48). The explicit cap shrinks DFLASH's per-request intermediate pool ~10.3 GB, growing the KV pool 442,745 → 708,463 tokens (~59k/user) |
| API | OpenAI-compatible on port 30000 (`--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`) |

## What passed on the latest profile

Evidence: [c12 validation report](evidence/recommended-c12-validation-20261016T040048Z.md),
[server info snapshot (c12 profile)](evidence/recommended-c12-server-info-20261016T035923Z.json),
[benchmark report](evidence/recommended-c12-bench-20261016T040048Z.json),
[startup-era server info](evidence/qwen27b-server-info-20261015T221826Z.json),
[smoke report](evidence/qwen27b-smoke-chat-20261015T221828Z.json).

- **Startup:** server ready in ~150–165 s on warm caches (~570 s cold, incl.
  weight load, every profile tested); no post-ready OOM or runtime error was
  recorded during the scoped checks.
- **Interfaces:** `/health` 200, `/v1/models` lists the served model, and a chat
  completion returned `"2 + 2 equals 4."` (`finish_reason=stop`) with reasoning
  separated into `reasoning_content`.
- **Single-GPU isolation:** GPU 0 ~85.9 GB in use while serving; GPU 1 idle.
- **DFLASH active:** draft runner initialized (`DFlash2DraftModel`, block size
  4 in the current profile), selector folded into the draft CUDA graph.
- **Concurrency benchmark** (identical seeded prompts across profiles, forced
  256-token decodes, `tests/benchmark.py`): up to the 12-user admission cap the
  engine holds ~110 tok/s per user (aggregate ~707 tok/s at c12; ~183 tok/s on
  12 × 16k-token prompts); beyond the cap excess requests queue (c45 TTFT p50
  ~3.1 s) instead of slowing everyone to ~59–69 tok/s as at the old cap of 48;
  zero retractions and exact output lengths on every request. Full tables in
  `docs/results.md`.

## Failures, regressions, and caveats

- **Container GPU-selection trap:** `--gpus device=0` (index) exposes the parent
  vGPU — `nvidia-smi` works but `torch.cuda` reports "No CUDA GPUs are
  available". The container must be pinned to the MIG device UUID. See
  [`docs/INFERENCE_CONTEXT.md`](docs/INFERENCE_CONTEXT.md).
- **Bundled NCCL 2.28.3 cannot init on this vGPU if used**
  (`cudaMemPoolCreate ... operation not supported`). Single-GPU serving does not
  touch NCCL; the host-NCCL 2.32.3 fallback is documented in the launch script
  for any future multi-GPU work.
- **First-touch stalls during image pull:** `docker run` could stall 90–120 s
  while the 31.5 GB image was still downloading; after the pull, starts are fast.
- **`/health` startup race:** a 503 was observed for a moment right after
  "Application startup complete", then 200.
- **Reasoning-model output shape:** a 16-token request returned empty `content`
  with only `reasoning_content`; give short Q&A ≥ ~100–256 tokens.

## Not yet validated

- Tool-call correctness under load (retrieval at 238k tokens, 91% of
  context, did quote a planted function exactly — see
  `evidence/recommended-c12-longctx-20261016T041501Z.md` — but that is one
  needle, not a tool-call eval).
- Anything multi-GPU (GPU 1 is untouched; TP>1 would need the NCCL fallback).
  The 2026-10-16 backlog (mem-fraction 0.90, `extra_buffer_lazy`,
  `no_buffer`) was measured and left unpromoted; see `docs/results.md`.

## Repository contents

- `configs/docker/` — the current launch configuration and one-time host setup;
  `candidates/` for prepared or historical alternatives.
- `tests/` — host-side validation scripts (smoke test + concurrency benchmark).
- `evidence/` — captured server snapshots and experiment reports.
- `docs/TESTING.md` — the manual-runs-only operating policy (hard rules).
- `docs/INFERENCE_CONTEXT.md` — hardware/vGPU/Docker/model/flag background.
- `docs/NEXT_EXPERIMENT.md` — planned experiments and their rules.
- `docs/results.md` — detailed results.

## Validation

```bash
bash -n configs/docker/*.sh
python3 -m py_compile tests/*.py
python3 tests/smoke_test.py --base-url http://127.0.0.1:30000   # server must be running
```

## Adapting the configurations safely

The launch script is specific to this host and model. Before using it
elsewhere, review the container image tag, model paths, HF cache mount, MIG UUID
resolution, port, and memory settings. The server binds `0.0.0.0` with no
authentication — do not expose it to an untrusted network. Model weights are not
included; obtain them under the model provider's terms. Captured reports may
contain host-specific metadata.

## Operating policy

Manual runs only: no systemd, no cron, no restart policies, no retry loops.
One server at a time, started in the foreground by a human. See
[`docs/TESTING.md`](docs/TESTING.md).
