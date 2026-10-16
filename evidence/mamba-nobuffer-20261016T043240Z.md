# Candidate: mamba strategy no_buffer

Date: 2026-10-16 ~04:29–04:33 UTC. Launcher:
`configs/docker/candidates/qwen27b-mamba-nobuffer.sh` after the failed
first launch. Not promoted. The promoted launcher was started again
afterward and came up healthy (`max_running_requests=12`,
`extra_buffer`, KV pool 708,050 — within 0.1% of the validated 708,463).
`evidence/recommended-c12-restored-server-info-20261016T043617Z.json`.

## Failed first launch (retained)

04:27:41 UTC, config without `--disable-overlap-schedule`. SGLang asserted
`no_buffer do not support overlap schedule` during arg resolution, before
model load. GPU stayed healthy. Not retried as the same config.
`evidence/mamba-nobuffer-failed-20261016T042741Z.md`.

## Corrected launch (one attempt)

Added `--disable-overlap-schedule`, which this image requires for
`no_buffer`. Requested `--max-running-requests 96`. The engine capped
admission itself:

    max_running_requests is capped to 65 by the mamba state cache
    (max_mamba_cache_size=196, 3 state slots per request).

`/server_info` still reports the requested 96. Effective cap is 65.

| Pool | promoted c12 | extra_buffer_lazy | no_buffer |
| --- | --- | --- | --- |
| Slots/request | 5 | 4 | **3** |
| Mamba states | 254 | 229 | **196** |
| Effective cap | 12 | 57 | **65** |
| DFLASH intermediate | 3.66 GB | 16.31 GB | **18.56 GB** |
| Shared KV pool | 708,463 | 425,330 | **427,154** |
| Free mem after init | 10.89 GB | 8.97 GB | 8.62 GB |
| Overlap schedule | on | on | **off** |

Raising the cap did not grow the KV pool versus lazy. The intermediate
SSM pool grew again and the mamba state count fell (229 → 196), so the
slot-ratio win was mostly consumed.

## Prefix reuse

Not equivalent to `extra_buffer`. Cached tokens landed on multiples of
2048, the chunked-prefill size, leaving the last partial chunk uncached:

| Prefix | prompt tokens | cached | extra_buffer / lazy |
| --- | --- | --- | --- |
| ~16k chars | 2,704 | **2,048** | 2,688 |
| ~48k chars | 8,093 | **6,144** | (not re-run; same probe cached 99% on extra_buffer) |

Unrelated prefixes cached 0 both times, so this is a short match, not a
false hit. Evidence:
`mamba-nobuffer-prefix-reuse-20261016T043135Z.json`,
`mamba-nobuffer-longprefix-prefix-reuse-20261016T043258Z.json`.

## Load

Full benchmark 9/9, zero retractions, GPU memory flat at 85,428 MiB,
`/health` 200 after. c45 ran concurrently (aggregate 2,216 tok/s, decode
p50 62 tok/s, TTFT p50 0.82 s). Disabling overlap did not show up as a
failure on this short suite. Evidence:
`mamba-nobuffer-bench-20261016T043240Z.json`,
`mamba-nobuffer-server-info-20261016T043129Z.json` (+ post).

## Verdict

`no_buffer` raises the mamba-bound cap to 65 and still serves, but prefix
reuse stops at chunk boundaries, overlap scheduling must be off, and the
KV pool stays ~427k — 40% below the promoted cap-12 config. Not promoted.
