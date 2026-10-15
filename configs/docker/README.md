# Docker configurations

`run_qwen27b_sglang.sh` is the current validated launch configuration:
Qwen3.8-27B NVFP4 via SGLang in Docker, pinned to GPU 0 by its **MIG device
UUID**, port 30000, foreground, `--restart no` (manual runs only — see
`docs/TESTING.md`).

`setup_docker_gpu.sh` is the one-time host setup (docker.io + NVIDIA Container
Toolkit + nvidia runtime + docker group). Run once with sudo.

`candidates/` holds prepared or historical alternatives. Candidates are not
interchangeable: each must be validated on an idle GPU and evidenced under
`evidence/` before it is treated as working.

All files are host-specific: review the container image tag, model paths, cache
mount, MIG UUID resolution, port, and memory settings before reuse on another
host.
