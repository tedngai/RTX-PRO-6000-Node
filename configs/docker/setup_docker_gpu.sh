#!/bin/bash
# One-time privileged setup: Docker engine + NVIDIA Container Toolkit (GPU containers).
#
# Run ONCE with sudo:
#   sudo bash /home/atadmin/vllmtest/configs/docker/setup_docker_gpu.sh
#
# What it does:
#   1. installs docker.io (Ubuntu 22.04 jammy: 29.1.3)
#   2. adds NVIDIA's repo and installs nvidia-container-toolkit
#   3. wires the nvidia runtime into Docker and restarts the docker daemon
#   4. adds atadmin to the docker group (log out/in, or run 'newgrp docker')
#   5. verifies GPU access with a throwaway container (nvidia-smi -L)
#
# Notes:
#   - The docker daemon itself runs as a system service (infrastructure, like sshd).
#     It is NOT set up to start any vLLM/SGLang workload: test containers run with
#     --restart no and are launched by hand (see docs/TESTING.md).
#   - Nothing here starts SGLang. Launching is a separate manual step.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: run me with sudo:  sudo bash $0" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "=== [1/4] docker.io ==="
apt-get update
apt-get install -y docker.io
systemctl enable --now docker

echo "=== [2/4] NVIDIA Container Toolkit ==="
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  install -m 0755 -d /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk --version

echo "=== [3/4] wiring nvidia runtime into docker ==="
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
sleep 2
systemctl is-active docker

echo "=== [4/4] adding atadmin to docker group ==="
usermod -aG docker atadmin

echo
echo "=== verify: GPU 0 visible inside a container ==="
docker run --rm --gpus '"device=0"' ubuntu:22.04 nvidia-smi -L || {
  echo "VERIFY FAILED - check the output above." >&2
  exit 1
}

echo
echo "SETUP COMPLETE."
echo "As atadmin: if 'docker' says permission denied, run 'newgrp docker' or log out/in."
echo "Then launch (manually, when ready):  configs/docker/run_qwen27b_sglang.sh"
