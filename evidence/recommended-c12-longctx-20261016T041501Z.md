# Long-context retrieval and prefix reuse (validated c12 config)

Date: 2026-10-16 ~04:09–04:15 UTC. Server: the promoted config already
running (`max_running_requests=12`, DFLASH D=4, ratio 2.0, KV pool
708,463, context 262,144). No relaunch. Thinking disabled on the chat
probes (`chat_template_kwargs.enable_thinking=false`) so the score is
retrieval, not reasoning-budget exhaustion. The canary
`orbit-slate-4192` appears only inside the planted function, not in the
question.

## Retrieval (`tests/long_context_retrieval.py`)

| Probe | prompt tokens | needle depth | elapsed | result |
| --- | --- | --- | --- | --- |
| control-4k-mid | 4,131 | 0.50 | 0.76 s | exact function quoted |
| near-mid | 238,453 | 0.50 | 91.1 s | exact function quoted |
| near-early | 238,304 | 0.10 | 91.3 s | exact function quoted |
| near-late | 238,338 | 0.90 | 91.1 s | exact function quoted |

238k tokens is 91% of the 262,144 context. All four finished `stop` and
quoted `needle_fn` including the canary and checksum. `/health` stayed
200. Evidence: `recommended-c12-longctx-20261016T041501Z.json`.

## Prefix reuse (`tests/prefix_reuse_probe.py`)

Same server, `/generate`, ~2,704-token unique prefix:

| Request | prompt tokens | cached tokens |
| --- | --- | --- |
| prime | 2,704 | 0 |
| same prefix, different suffix | 2,704 | 2,688 (99.4%) |
| unrelated prefix | 2,721 | 0 |

`extra_buffer` prefix reuse works at the cap-12 config. This is the
baseline for the mamba-strategy candidates. Evidence:
`recommended-c12-prefix-reuse-20261016T040944Z.json`.
