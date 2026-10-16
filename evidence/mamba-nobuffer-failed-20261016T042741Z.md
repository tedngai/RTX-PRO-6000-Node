# FAILED probe: no_buffer candidate, first launch

Date: 2026-10-16 04:27:41 UTC. Launcher as committed in `cce3766` /
`5ac9281` (no `--disable-overlap-schedule`). Container `qwen27b-sglang`
exited 1 in ~30 s. **Not a GPU failure and not retried as the same
config.**

## What happened

SGLang rejected the config during argument resolution, before model
load:

    AssertionError: no_buffer do not support overlap schedule.
    Try to set disable_overlap_schedule=True.

`torch.cuda.mem_get_info()` succeeded in the same log (97,838 MiB), and
GPU 0 was idle afterward (462 MiB, no compute apps). The vGPU did not
wedge. Raw log excerpt: `mamba-nobuffer-failed-20261016T042741Z.txt`.

## Follow-up

The candidate script was corrected with `--disable-overlap-schedule`
and launched once more. That is a different config, not a restart of
this failure. See the later `mamba-nobuffer-*` evidence if that launch
produced any.
