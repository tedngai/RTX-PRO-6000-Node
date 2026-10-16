#!/bin/bash
# Qwen3.8-27B (RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead) — SGLang in Docker, SINGLE GPU: GPU 0 only.
# GPU 0 is selected by its MIG device UUID (this vGPU runs in MIG mode; selecting by
# index would expose the parent vGPU and CUDA would fail inside the container).
# CANDIDATE (not the promoted config): mamba radix strategy no_buffer.
# Same as configs/docker/run_qwen27b_sglang.sh except
# --mamba-radix-cache-strategy extra_buffer -> no_buffer (3 state
# slots/request instead of 5) and --max-running-requests 96, set above the
# expected mamba cap (~84 at 254 states) so the engine's mamba cap is what
# binds. DFLASH otherwise forces 48 when the flag is omitted, which would
# hide the raise. Production admission stays 12; this candidate only
# measures the higher cap and whether prefix reuse still works.
# Do not promote without evidence. One manual run; no retry if it OOMs.
#
# MANUAL USE ONLY. Runs in the foreground; Ctrl+C stops it. There is NO restart
# policy, no systemd unit, and no auto-restart: if SGLang dies or takes the VM
# down, nothing brings it back — you decide when to try again (see docs/TESTING.md).
#
# Preconditions:
#   - one-time host setup done:  sudo bash configs/docker/setup_docker_gpu.sh
#   - GPU 0 idle, port 30000 free (the script checks both and refuses otherwise)
#
# Stop:              Ctrl+C here, or from another terminal: docker stop qwen27b-sglang
# Follow logs:       tail -f logs/qwen27b_sglang.log
# After a crash:     docker logs --tail 200 qwen27b-sglang
# Remove leftover:   docker rm qwen27b-sglang
#
# NOTE (lifetime, 2026-10-15): this script runs `docker run` ATTACHED, so docker's
# signal proxy ties the container to the shell that launched it. Closing or
# killing that shell forwards SIGTERM to SGLang, which shuts down cleanly
# (exit 0) — and with `--restart no` nothing brings it back. When launching from
# a shell that will not stay alive (e.g. an agent session), detach it:
#   sg docker -c 'setsid nohup bash configs/docker/candidates/qwen27b-mamba-nobuffer.sh \
#       >> logs/launch_qwen27b.out 2>&1 < /dev/null &'
# Stopping is unchanged: docker stop qwen27b-sglang.

set -u
cd /home/atadmin/vllmtest || exit 1

NAME=qwen27b-sglang
LOG=logs/qwen27b_sglang.log
PORT=30000

# --- guard: docker usable? ---
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot talk to docker." >&2
  echo "       - setup not run? -> sudo bash configs/docker/setup_docker_gpu.sh" >&2
  echo "       - permission denied? -> run 'newgrp docker' or log out/in (docker group)." >&2
  exit 1
fi

# --- guard: no vllm server already running ---
if pgrep -f '^(/home/atadmin/vllmtest/)?(venv/bin/)?vllm serve' >/dev/null 2>&1; then
  echo "ERROR: a vllm serve is already running; stop it first (Ctrl+C in its terminal)." >&2
  exit 1
fi

# --- guard: our container already running? ---
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "ERROR: container $NAME is already running." >&2
  echo "       Stop it with:  docker stop $NAME" >&2
  exit 1
fi

# --- cleanup: exited leftover container (keep its last log lines first) ---
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Found exited container $NAME; saving last 200 log lines to logs/qwen27b_sglang_prev.log"
  echo "=== [$(date '+%F %T')] previous $NAME log tail ===" >> logs/qwen27b_sglang_prev.log
  docker logs --tail 200 "$NAME" >> logs/qwen27b_sglang_prev.log 2>&1 || true
  docker rm "$NAME" >/dev/null
fi

# --- guard: port free ---
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "ERROR: port $PORT is already in use:" >&2
  ss -ltnp 2>/dev/null | grep ":${PORT} " >&2
  exit 1
fi

# --- guard: GPU 0 idle ---
if nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "ERROR: something is already using GPU 0:" >&2
  nvidia-smi -i 0 >&2
  exit 1
fi

# --- resolve GPU 0's MIG device UUID (this vGPU runs in MIG mode) ---
MIG_UUID=$(nvidia-smi -L | awk '/^GPU 0:/{g=1; next} g && /MIG/{if (match($0,/MIG-[0-9a-f-]+/)){print substr($0,RSTART,RLENGTH); exit}}')
if [ -z "$MIG_UUID" ]; then
  echo "ERROR: could not resolve GPU 0's MIG UUID from nvidia-smi -L:" >&2
  nvidia-smi -L >&2
  exit 1
fi
echo "GPU 0 (MIG): $MIG_UUID"

# --- optional HF token passthrough (repos are public; normally not needed) ---
TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  TOKEN_ARGS=(--env "HF_TOKEN=$HF_TOKEN")
fi

echo "Starting SGLang (single GPU 0) on port $PORT — manual run, Ctrl+C to stop."
echo "Following log: tail -f /home/atadmin/vllmtest/$LOG"

# NOTE (NCCL): the image bundles NCCL 2.28.3, which cannot init on this vGPU
# (cudaMemPoolCreate "operation not supported") IF NCCL is used. Single-GPU mode
# should not touch NCCL. If it ever fails with that error (or for TP>1 later),
# mount and preload the host's fixed NCCL 2.32.3 (archived) by adding:
#   -v /home/atadmin/archive/vllmtest-20261015/infra/nccl/ext_cu13_2.32.3/nvidia/nccl/lib/libnccl.so.2:/opt/nccl/libnccl.so.2:ro \
#   --env LD_PRELOAD=/opt/nccl/libnccl.so.2 \

docker run \
  --name "$NAME" \
  --restart no \
  --gpus "device=$MIG_UUID" \
  --shm-size 32g \
  --ipc=host \
  -p ${PORT}:${PORT} \
  -v /home/atadmin/.cache/huggingface:/root/.cache/huggingface \
  "${TOKEN_ARGS[@]}" \
  lmsysorg/sglang:latest \
  sglang serve \
    --trust-remote-code \
    --model-path RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --chunked-prefill-size 2048 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --mamba-full-memory-ratio 2.0 \
    --max-running-requests 96 \
    --host 0.0.0.0 \
    --port 30000 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
    --speculative-num-draft-tokens 4 \
    --mamba-radix-cache-strategy no_buffer \
    --mamba-ssm-dtype bfloat16 \
  2>&1 | tee -a "$LOG"
