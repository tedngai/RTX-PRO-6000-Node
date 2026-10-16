# Runtime validation of the promoted config (DFLASH D=4, r=2.0, cap 12)

Date: 2026-10-16 ~03:45–04:01 UTC. Companion JSON evidence:
`recommended-c12-server-info-20261016T035923Z.json` (pre-bench),
`recommended-c12-bench-20261016T040048Z.json`,
`recommended-c12-server-info-post-20261016T040050Z.json` (post-bench).

## Context

This is the first successful runtime validation of the promoted main config.
All three post-promotion launch attempts on 2026-10-16 (03:02–03:22 UTC) died
at CUDA init in the GPU 0 vGPU wedge (`gpu0-wedge-20261016.md`). The VDI guest
rebooted at 2026-10-15 23:45 local time; recovery followed the wedge report's
next-steps checklist.

- Wedge recovery probe (post-reboot, before launch): inside the sglang image
  against GPU 0's MIG slice, `torch.cuda.is_available() True`,
  `device_count() 1`, `mem_get_info() (97522810880, 102590840832)` — the
  wedge is cleared. (MIG UUID observed post-reboot:
  `MIG-b12c2fdf-…`; the launch script re-resolves it dynamically.)
- Launch: `configs/docker/run_qwen27b_sglang.sh` (promoted config, unmodified)
  via the documented detached recipe; `/health` 200 after ~570 s
  (tokenizer_e2e 497.55 s, weight load incl. draft 386.43 s).

## Startup pools (from `logs/qwen27b_sglang.log`)

| Pool | cap 12 (this run) | cap 48 (dflash-d4-r2) |
| --- | --- | --- |
| Mamba states | 254 (ssm 17.93 + conv 0.70 GB) | 254 (same) |
| DFLASH intermediate | 3.66 + 0.07 GB | 13.78 + 0.27 GB |
| Shared KV pool | **708,463 tokens** | 442,745 tokens |
| max_running_requests | **12** (explicit) | 48 (engine-derived) |
| Free GPU mem after init | 10.89 GB | 13.9 GB |

The explicit cap 12 shrinks the per-request speculative intermediate SSM pool
(~10.3 GB freed), which is why the shared KV pool grew 60% over the cap-48
measurement: ~59k tokens of context per admitted user.

## Benchmark (`tests/benchmark.py --profile recommended-c12`)

9/9 scenarios, all requests OK, zero retractions, accept length 3.12–3.68
(of 4 draft tokens). Aggregate tok/s / decode-p50 tok/s per request:

| Scenario | cap 48 (2026-10-16 02:42) | cap 12 (this run) |
| --- | --- | --- |
| short c1 | 116.9 / 130.9 | 117.0 / 130.9 |
| short c8 | 580.8 / 107.7 | 567.2 / 104.8 |
| short c12 | — | 706.5 / 110.0 |
| short c16 | 982.4 / 97.6 | 733.0 / 109.3 |
| short c32 | 1,783.9 / 75.5 | 1,064.5 / 114.5 |
| short c45 | 1,917.6 / 58.7 | 1,034.0 / 106.6 |
| long c1 | 89.6 / 144.0 | 90.0 / 144.3 |
| long c4 | 163.5 / 79.4 | 163.9 / 82.5 |
| long c12 | — | 183.5 / 37.4 |

Reading: up to the 12-user cap the numbers are identical to the cap-48 run
(within the known ±10% wobble). Beyond the cap, admitted traffic keeps
~110 tok/s decode (vs 59–76 when the engine ran 16–45 at once) while excess
requests queue — c16/c32/c45 TTFT p50 rises to 0.37/2.81/3.12 s and aggregate
drops as designed. long-c12 (12 × 16k-token prompts concurrently) completes
cleanly: TTFT p50 7.0 s, aggregate 183.5 tok/s.

Post-bench `/server_info` unchanged (708,463 / 12); `/health` 200.

## Verdict

The promoted config in `configs/docker/run_qwen27b_sglang.sh` is now
**runtime-validated**: healthy launch, config confirmed in `/server_info`,
full benchmark suite green. The wedge-incident caveat on the promotion is
resolved.
