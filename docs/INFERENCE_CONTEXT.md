# Inference context — RTX PRO 6000 Blackwell vGPU node

Background for the configurations and evidence in this repository. Facts were
recorded on 2026-10-15 on the test host; re-verify before reusing on other hosts.

## Hardware / virtualization

- 2× NVIDIA RTX PRO 6000 Blackwell `DC-4-96Q` **vGPU**, each presented as a
  `MIG 4g.96gb` slice (98,304 MiB / 96 GiB per slice).
- No NVLink; `can_device_access_peer=False`; PCIe/SHM only.
- The vGPU has **no CUDA memory pools** (`cudaDevAttrMemoryPoolsSupported=0`)
  and no UVM. This breaks NCCL ≤ 2.30.7 at init (see Known issues).
- Host driver 580.159.03, CUDA 13.0; 62 GiB host RAM; `/dev/shm` 32 GiB.
- The Qwen3.8-27B test uses **GPU 0 only**; GPU 1 is left untouched.

## MIG device selection (important)

On this vGPU, MIG is active, and container GPU selection behaves as follows
(all four variants tested with the actual SGLang image):

| Container flag | Result inside the container |
| --- | --- |
| `--gpus device=0` (index) | `nvidia-smi` lists the parent GPU, but `torch.cuda` reports **"No CUDA GPUs are available"** |
| `--gpus device=MIG-1e57ece9-...` (UUID) | **1 device, CUDA available** — used by the launch script |
| `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=MIG-...` | 1 device, CUDA available (equivalent) |
| `--gpus all` | 2 devices (both slices) |

The launch script resolves GPU 0's MIG UUID dynamically from `nvidia-smi -L` on
every run, so it survives VM reboots (UUIDs are not hardcoded).

## Docker stack

- docker.io 29.1.3 (Ubuntu 22.04) + NVIDIA Container Toolkit 1.20.1, with the
  `nvidia` runtime wired in.
- Image: `lmsysorg/sglang:latest` = SGLang **v0.5.21**, CUDA **13.0.3**, bundled
  NCCL **2.28.3**; entrypoint `/opt/nvidia/nvidia_entrypoint.sh`, workdir
  `/sgl-workspace/sglang`.
- Containers run with `--restart no` and are launched by hand
  (see `docs/TESTING.md`). Container name: `qwen27b-sglang`.
- Mounts: `--shm-size 32g`, `--ipc=host`, port `30000:30000`, and the Hugging
  Face cache `/home/atadmin/.cache/huggingface -> /root/.cache/huggingface`
  (Xet layout; model directories are symlink trees into `blobs/`).

## Models

- `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` — main checkpoint (~22 GiB),
  ModelOpt mixed precision (NVFP4 weights + FP8 attention, FP8 KV config),
  architecture `Qwen3_5ForConditionalGeneration` (hybrid linear-attention
  "mamba" layers), 262,144 max positions.
- `incoai/Qwen3.8-27B-DFlash2` — DFLASH draft head (~3.6 GiB), 5 layers,
  `DFlash2DraftModel`.
- Both repositories are public; no HF token is needed (optional `HF_TOKEN`
  passthrough exists in the script).

## Why these flags

- `--kv-cache-dtype fp8_e4m3` — halves KV footprint.
- `--mem-fraction-static 0.85` — static weights + KV reservation; leaves about
  13.9 GB free after init with the current pool sizes.
- `--attention-backend flashinfer` — prebuilt kernels for sm_120 in this image.
- `--chunked-prefill-size 2048` — moderate prefill chunks.
- `--mamba-full-memory-ratio 2.0` (was 6.08 until 2026-10-16),
  `--mamba-radix-cache-strategy extra_buffer`, `--mamba-ssm-dtype bfloat16` —
  hybrid linear-attention (GDN) cache sizing and strategy; `extra_buffer`
  keeps extra state buffers for the radix cache. The ratio splits the cache
  budget between mamba states and the KV pool (mamba gets ratio/(1+ratio) of
  it); 6.08 starved the KV pool (182k tokens); 2.0 rebalances to 442,745 KV
  tokens with 254 mamba states — enough for the 48-request cap (5 slots/req).
- `--speculative-algorithm DFLASH`, `--speculative-draft-model-path
  incoai/Qwen3.8-27B-DFlash2`, `--speculative-num-draft-tokens 4` (was 8
  until 2026-10-16) — DFLASH speculative decoding (a v0.5.21 builtin); the
  token count is the verify window (alias of `--speculative-dflash-block-size`).
  D=8 reserved a 25.9 GB intermediate-SSM pool and capped requests at 45;
  D=4 halves that pool and lifts the cap to 48 for a small speed cost
  (A/B in `docs/results.md`; EAGLE/MTP 3/1/4 is the max-capacity alternative,
  kept in `configs/docker/candidates/`).
- `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder` — OpenAI
  reasoning/tool-call parsing.

## Known issues and quirks (recorded)

- **NCCL**: the image bundles NCCL 2.28.3, which cannot init on this vGPU
  (`cudaMemPoolCreate ... operation not supported`) **if NCCL is used**.
  Single-GPU serving does not initialize NCCL, so this has not blocked the
  current config. Fallback: mount and preload the host's NCCL 2.32.3 wheel
  (archived at
  `/home/atadmin/archive/vllmtest-20261015/infra/nccl/ext_cu13_2.32.3/nvidia/nccl/lib/libnccl.so.2`);
  the comment block in the launch script has the exact flags.
- **First container start during image pull**: while the 31.5 GB image was
  downloading, `docker run` calls could stall 90–120 s before the container
  started. After the pull completed, starts are fast. Treat one-off stalls
  during a pull as I/O contention, not a configuration defect.
- **`/health` race**: immediately after "Application startup complete",
  `/health` can return 503 for a moment; it returned 200 on the next check.
- **Reasoning model output shape**: chain-of-thought arrives in
  `reasoning_content`; with a small `max_tokens` the answer `content` can be
  empty and `finish_reason` is `length`. Give short Q&A at least ~100–256
  tokens.
- **Startup cost**: recorded ~2 min to ready (weights ~10 s warm, CUDA graph
  capture ~70 s).

## Archived predecessor work

Earlier vLLM projects on this host (DeepSeek-V4-Flash-NVFP4 TP2/DP2,
Qwen3.8-Flash-Next-FP8, OLMoE TP2+EP2, and the NCCL bring-up) are archived at
`/home/atadmin/archive/vllmtest-20261015/` with their own README.
