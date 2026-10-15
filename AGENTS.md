# Agent rules for the RTX-PRO-6000-Node repository

Packaged Qwen3.8-27B SGLang deployment project: launch and setup scripts,
validation tests, captured evidence, and documentation for one RTX PRO 6000
Blackwell vGPU node (single-GPU serving on GPU 0).

## Workflow

- **Always commit and push changes.** After any change to files in this
  repository, commit it with git using a descriptive message and push it to
  `origin` immediately. Make frequent, small, incremental commits and pushes
  rather than batching unrelated changes. Do not leave the working tree dirty
  or commits unpushed at the end of a task.
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

- The public GitHub remote is `https://github.com/tedngai/RTX-PRO-6000-Node.git`
  (remote name `origin`). Push with `git push origin main`.
- **Credential location:** the GitHub PAT is stored in the repository-root
  `.env` file — `/home/atadmin/vllmtest/.env`, variable name `GITHUB_PAT`.
  The file is gitignored and must stay untracked (verify with
  `git check-ignore .env` before committing).
- A repo-local git credential helper (`.git/config` → `credential.helper`)
  already reads the PAT from that file, so a plain `git push origin main` works
  with no extra steps. If the helper is ever missing, recreate it with:

  ```bash
  git config --local credential.helper \
    '!f() { . /home/atadmin/vllmtest/.env; echo username=x-access-token; echo "password=$GITHUB_PAT"; }; f'
  ```

- Never print the token, put it in a remote URL or command history, copy it
  into tracked files, or stage `.env`.
