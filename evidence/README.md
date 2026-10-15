# Evidence

Captured server information and experiment reports. These are data, not
executable code. File names identify the serving profile and the UTC capture
time (`YYYYMMDDTHHMMSSZ`).

Reports can contain host-specific paths and configuration details. Do not treat
them as proof that a configuration is currently running; verify the live server
(`/health`, `/server_info`) before making changes.

First entries (2026-10-15): `qwen27b-server-info-*.json` (full server
configuration snapshot) and `qwen27b-smoke-chat-*.json` (health + chat
completion report from `tests/smoke_test.py`).
