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

## 2026-10-16 — speculative-decode A/B (DFLASH D=8 vs DFLASH D=4 vs EAGLE/MTP 3/1/4)

Same harness, byte-identical prompts (`tests/benchmark.py`); only the
speculative-decoding flags differ. Capacities from startup logs:

| Profile | Spec config | KV pool (tokens) | Request cap | Intermediate SSM |
| --- | --- | --- | --- | --- |
| baseline-dflash8 | DFLASH D=8, DFlash2 draft | 181,925 | 45 | 25.88 GB |
| dflash-d4 | DFLASH D=4, DFlash2 draft | 307,769 | 48 | 13.78 GB |
| mtp-eagle-314 | EAGLE/MTP 3/1/4, in-checkpoint head | 376,415 | 48 | 13.78 GB |

Benchmark (aggregate tok/s / per-req decode p50):

| Scenario | DFLASH-8 | DFLASH-4 | MTP 3/1/4 |
| --- | --- | --- | --- |
| short c1 | 204.7 / 216.6 | 117.0 / 131.2 | 98.8 / 105.5 |
| short c8 | 644.6 / 190.3 | 569.8 / 112.7 | 526.0 / 95.6 |
| short c16 | 1,475.7 / 144.4 | 1,187.0 / 98.3 | 897.8 / 88.3 |
| short c32 | 2,166.7 / 100.2 | 1,861.9 / 78.6 | 1,430.2 / 70.4 |
| short c45 | 2,012.3 / 69.1 | 1,900.0 / 60.0 | 1,848.7 / 58.9 |
| long c1 | 121.8 / 245.9 | 89.9 / 144.6 | 82.7 / 129.2 |
| long c4 | 186.8 / 106.1 | 152.1 / 73.5 | 150.8 / 71.8 |

Reading:

- DFLASH-4 matches MTP's capacity mechanics (intermediate SSM pool halves,
  request cap 45 → 48) but is 20–30% faster at mid concurrency: DFlash2
  drafts all tokens in a single pass, while MTP chains 3 small steps.
  MTP's extra ~69k pool tokens come from its 1-layer draft KV (0.72 GB vs
  DFlash2's 2.94 GB).
- At saturation (c45) the three converge (1.85–2.01k tok/s): compute-bound.
- Caveat: the synthetic prose inflates D=8 acceptance (5.3–6.3). The server's
  live average on earlier short-answer traffic was 3.15 — essentially D=4's
  3.1–3.6. On real text, D=8 wastes drafts 5–8, so its real-world speed edge
  over D=4 should be far smaller than measured here.

Decision: **DFLASH D=4** is the preferred spec profile for the capacity-first
goal (+69% KV pool, cap 48, ≤6% aggregate loss at saturation). EAGLE/MTP is
retained as the max-capacity spec alternative. The mamba/KV rebalance
(experiment 2) runs on top of DFLASH-4.

Evidence: `evidence/dflash-d4-bench-20261016T023634Z.json`,
`evidence/mtp-eagle-314-bench-20261016T023049Z.json`,
`evidence/baseline-dflash8-bench-20261016T022322Z.json`.

## 2026-10-16 — mamba/KV rebalance (`--mamba-full-memory-ratio` 6.08 → 2.0, on top of DFLASH-4)

Candidate: [`configs/docker/candidates/qwen27b-dflash-d4-r2.sh`](../configs/docker/candidates/qwen27b-dflash-d4-r2.sh).
Startup: mamba states 254 (ssm 17.93 GB + conv 0.70 GB) + speculative
intermediate 13.78 GB + 0.27 GB; **KV pool 442,745 tokens** (13.51 GiB main +
4.22 GB DFlash2 draft KV); request cap **48** (254 states ÷ 5 slots = 50 > 48 —
the mamba cache no longer binds admission).
Server info: `evidence/dflash-d4-r2-server-info-20261016T024552Z.json`.

| Scenario | baseline (D=8, r=6.08) | dflash-d4-r2 | delta vs baseline |
| --- | --- | --- | --- |
| short c1 | 204.7 / 216.6 | 116.9 / 130.9 | −43% (spec D 8→4) |
| short c8 | 644.6 / 190.3 | 580.8 / 107.7 | −10% |
| short c16 | 1,475.7 / 144.4 | 982.4 / 97.6 | −33% |
| short c32 | 2,166.7 / 100.2 | 1,783.9 / 75.5 | −18% |
| short c45 | 2,012.3 / 69.1 | 1,917.6 / 58.7 | −5% |
| long c1 | 121.8 / 245.9 | 89.6 / 144.0 | −26% |
| long c4 | 186.8 / 106.1 | 163.5 / 79.4 | −12% |

(aggregate tok/s / per-req decode p50. A fresh-prompt variance probe
`python3 tests/benchmark.py --profile dflash-d4-r2-variance --run-seed 1
--only short` gave c16 1,074.8 / c32 1,727.4 / c45 1,852.9 — short 2–6 s
scenarios wobble ~±10% run-to-run, so the rebalance costs nothing measurable
in speed: it is identical to `dflash-d4` at c1/c8/long and within noise
elsewhere. The probe was not saved as evidence; the command reproduces it.)

**Headline vs the 2026-10-15 baseline config (D=8, r=6.08):**

- Shared KV pool: 181,925 → **442,745 tokens (2.43×)**
- Concurrent requests: 45 → **48**
- Per-user context at full load: ~4.0k → **~9.2k tokens each** (442,745 ÷ 48)
- Aggregate throughput at saturation: ~2.0k → ~1.9k tok/s (−5%)
- Headroom after init: 8.1 → 13.9 GB

This profile is promoted to
[`configs/docker/run_qwen27b_sglang.sh`](../configs/docker/run_qwen27b_sglang.sh)
(the pre-optimization D=8 / r=6.08 config remains in git history and the
`candidates/` files document each A/B arm).
Evidence: `evidence/dflash-d4-r2-bench-20261016T024247Z.json`.

## 2026-10-16 — runtime validation of the promoted config (`--max-running-requests 12`)

The three launch attempts made right after the promotion (03:02–03:22 UTC) all
failed at CUDA init in the GPU 0 vGPU wedge (`evidence/gpu0-wedge-20261016.md`);
the guest was rebooted and GPU 0 verified healthy (`mem_get_info` ≈ 97.5 GB
free on the MIG slice) before this run. Narrative + full table:
[`evidence/recommended-c12-validation-20261016T040048Z.md`](../evidence/recommended-c12-validation-20261016T040048Z.md).

Startup (promoted config, unmodified): mamba states 254 (17.93 + 0.70 GB,
unchanged); DFLASH intermediate **3.66 + 0.07 GB** (was 13.78 + 0.27 at the
engine-derived cap 48 — the explicit cap 12 shrinks the per-request pool);
**KV pool 708,463 tokens** (+60% vs 442,745); `max_running_requests=12`;
free mem after init 10.89 GB; `/health` 200; server info before and after the
benchmark identical.

Benchmark (`--profile recommended-c12`, aggregate / decode-p50 tok/s): up to
c12 the numbers match the cap-48 run within noise (c1 117.0/130.9, c8
567.2/104.8, c12 706.5/110.0; long c12 183.5, TTFT p50 7.0 s). Above the cap,
excess requests queue rather than slow the admitted 12: c16/c32/c45 keep
~109–115 tok/s decode p50 (vs 98/76/59 at cap 48) while TTFT p50 grows to
0.37/2.81/3.12 s. 9/9 scenarios green, zero retractions, accept length
3.12–3.68 of 4 draft tokens.

**The promoted config is now runtime-validated** — the wedge-incident caveat on
the promotion is resolved. Evidence:
`evidence/recommended-c12-bench-20261016T040048Z.json`,
`evidence/recommended-c12-server-info-20261016T035923Z.json` (+ post-bench).

## Caveats recorded during bring-up

- A 16-token request returned empty `content` with only `reasoning_content` —
  the model spends tokens on reasoning first; give short Q&A ≥ ~100–256 tokens.
- `/health` returned 503 for a moment immediately after "Application startup
  complete", then 200 on the next check.

## 2026-10-16 — long-context retrieval + prefix reuse (promoted c12 config, no relaunch)

Needle-in-a-haystack on the chat API, thinking disabled, canary
`orbit-slate-4192` planted only inside one function. All four probes
quoted the function exactly (`finish_reason=stop`):

| Probe | prompt tokens | needle depth | elapsed |
| --- | --- | --- | --- |
| control, mid | 4,131 | 0.50 | 0.76 s |
| near-limit, mid | 238,453 | 0.50 | 91.1 s |
| near-limit, early | 238,304 | 0.10 | 91.3 s |
| near-limit, late | 238,338 | 0.90 | 91.1 s |

238k is 91% of the 262,144 context. Narrative:
[`evidence/recommended-c12-longctx-20261016T041501Z.md`](../evidence/recommended-c12-longctx-20261016T041501Z.md).

Prefix reuse on the same server (`extra_buffer`): a second request sharing
a 2,704-token prefix cached **2,688** tokens; an unrelated prefix cached 0.
Evidence: `evidence/recommended-c12-prefix-reuse-20261016T040944Z.json`.

## 2026-10-16 — mem-fraction 0.90 + CUDA-graph trims (candidate, not promoted)

One launch of
[`configs/docker/candidates/qwen27b-mf90-graphtrim.sh`](../configs/docker/candidates/qwen27b-mf90-graphtrim.sh).
Decode graphs `[1, 2, 4, 8, 12]`, prefill graphs 10 shapes through 2048,
admission still 12. KV pool 708,463 → **779,104 (+10%)**, mamba states
254 → 277, free mem after init 10.89 → **6.95 GB**. Full benchmark 9/9,
zero retractions, no OOM; GPU memory did not grow under the suite.
Speed at c8+ matches the c12 run within noise. Not promoted: the KV gain
is modest and headroom is 4 GB tighter. Narrative:
[`evidence/mf90-graphtrim-20261016T042118Z.md`](../evidence/mf90-graphtrim-20261016T042118Z.md).

## Not measured yet

- Tool-call correctness under load.
- Backlog still open: `extra_buffer_lazy` / `no_buffer` mamba strategies.
  See [`NEXT_EXPERIMENT.md`](NEXT_EXPERIMENT.md).
