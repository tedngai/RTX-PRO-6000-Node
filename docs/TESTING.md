# Testing policy — manual runs only

Every test on this host is started **by hand, in the foreground**. Nothing
auto-starts and nothing auto-restarts.

## Hard rules

1. **No systemd, no cron, no supervisors, no retry loops.** Do not create
   `.service`/`.timer` units, `systemd-run` units, `Restart=always` policies, or
   wrap these scripts in any loop that relaunches them. (This was the failure
   mode that had to be torn out: a crashed vLLM kept being auto-restarted
   straight back into the crash.)
2. **One server at a time.** The current config claims ~85% of GPU 0's 96 GiB
   slice and leaves ~8 GB headroom; running two servers at once has crashed the
   VM. The run script refuses to start if a server is already up.
3. **Run it in a terminal you can watch; stop with Ctrl+C.** If a run hangs or
   takes the VM down, nothing relaunches it — you decide whether to try again.
4. **Downloads run once, by hand.** If a model ever needs downloading again,
   run a single `hf download` pass and exit; it must never launch a server.

## Scripts (current repository)

| Script | What it does | Port | Log |
|---|---|---|---|
| `configs/docker/run_qwen27b_sglang.sh` | Qwen3.8-27B NVFP4 via Docker/SGLang, **single GPU 0** (MIG UUID), DFLASH speculative decoding | 30000 | `logs/qwen27b_sglang.log` |
| `configs/docker/setup_docker_gpu.sh` | one-time host setup (sudo): docker.io + NVIDIA Container Toolkit + nvidia runtime | — | — |
| `tests/smoke_test.py` | `/health` + one chat completion against a running server; can save an evidence report | — | stdout / `evidence/` |

Docker-based test (Qwen3.8-27B): one-time host setup is
`sudo bash configs/docker/setup_docker_gpu.sh` (docker engine + NVIDIA container
toolkit). The docker daemon is infrastructure only — the SGLang container runs
with `--restart no`, is pinned to GPU 0 by its **MIG device UUID** (resolved
dynamically by the script; index selection breaks CUDA on this vGPU), and is
started by hand with `configs/docker/run_qwen27b_sglang.sh`.

## Before starting a server

```bash
pgrep -af '^(/home/atadmin/vllmtest/)?(venv/bin/)?vllm serve'   # must print nothing
nvidia-smi                                                      # GPU 0 idle, GPU 1 idle
docker ps -a --filter name=qwen27b-sglang                       # no leftover container
```

## Start / verify / stop

```bash
configs/docker/run_qwen27b_sglang.sh     # foreground; Ctrl+C to stop
tail -f logs/qwen27b_sglang.log          # from another terminal
curl -s localhost:30000/health           # wait for HTTP 200 before sending requests
python3 tests/smoke_test.py              # optional end-to-end check
```

The script runs `docker run` **attached**: the container lives exactly as long as
the shell that started it. Closing or killing that shell forwards SIGTERM and the
server shuts down cleanly (exit 0, no crash) — with `--restart no` nothing brings
it back. When launching from a shell that will not stay alive (agent session,
scripted setup), detach instead:

```bash
sg docker -c 'setsid nohup bash configs/docker/run_qwen27b_sglang.sh \
    >> logs/launch_qwen27b.out 2>&1 < /dev/null &'
```

This is still a manual start with no restart policy; stop it with
`docker stop qwen27b-sglang`.

## After a crash

Nothing restarts by itself. Check `nvidia-smi`, make sure no server remains
(`pgrep -af '...vllm serve'`, `docker ps -a`), then restart by hand only if you
want to retry.

## Archived predecessor scripts

The DeepSeek-V4, Qwen3.8-Flash-Next, and OLMoE vLLM scripts (and their venv /
NCCL environment) were archived to `/home/atadmin/archive/vllmtest-20261015/`
when this repository was created. They are reference material only.
