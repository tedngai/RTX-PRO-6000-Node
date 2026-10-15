# Agent rules for the RTX-PRO-6000-Node repository

Packaged Qwen3.8-27B SGLang deployment project: launch and setup scripts,
validation tests, captured evidence, and documentation for one RTX PRO 6000
Blackwell vGPU node (single-GPU serving on GPU 0).

## Workflow

- **Always commit changes.** After any change to files in this repository,
  commit it with git using a descriptive message. Do not leave the working
  tree dirty at the end of a task.
- After adding, removing, renaming, or materially changing files, regenerate
  `docs/package-manifest.json` (paths, sizes, SHA-256) before committing.
- New experiment reports belong in `evidence/`, named by profile label and UTC
  timestamp (`YYYYMMDDTHHMMSSZ`). Never overwrite an existing report with
  different content. Retain failed or invalid probes and mark them clearly.
- Never commit model weights, Hugging Face caches, Docker images, live logs
  (`logs/`), shell history, or tokens. `.gitignore` is a guardrail, not
  permission.

## Operating policy (hard rules)

- Manual runs only: no systemd units, no cron, no supervisors, no restart
  policies, no retry loops. Containers run with `--restart no`. A crashed
  server stays down until a human retries it. See `docs/TESTING.md`.
- One server at a time on GPU 0; the launch script refuses to start otherwise.
- The live artifact is the container `qwen27b-sglang`, started by hand with
  `configs/docker/run_qwen27b_sglang.sh`. Verify with `/health` and
  `/server_info` before and after changes. Never claim a configuration is
  validated without runtime evidence recorded under `evidence/`.
- GPU selection must use the **MIG device UUID** (the launch script resolves it
  dynamically). Selecting by index exposes the parent vGPU and breaks CUDA
  inside the container.

## Publishing

- No public remote is configured yet. When one is added, use a credential
  helper or an ignored `.env` file for the token. Never print the token, embed
  it in a remote URL or command history, or stage it. Verify `.env` remains
  untracked before pushing.
