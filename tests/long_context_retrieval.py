#!/usr/bin/env python3
"""Near-limit needle-in-a-haystack retrieval probe (stdlib only).

Builds a deterministic haystack, inserts one unique Python function at a
chosen depth, and asks the running chat API to quote it. Thinking is
disabled (chat_template_kwargs.enable_thinking=false) so the score is
retrieval, not reasoning-budget exhaustion. The haystack is not stored in
the evidence file.

Never starts or stops the server. A failed or timed-out probe is still saved.
Does not relaunch anything.

Usage:
    python3 tests/long_context_retrieval.py --profile recommended-c12 --save-dir evidence
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

CANARY = "orbit-slate-4192"
NEEDLE = (
    "def needle_fn():\n"
    '    """retrieval-canary orbit-slate-4192"""\n'
    '    return {"canary": "orbit-slate-4192", "checksum": 4192}\n'
)
QUESTION = (
    "\nA Python function named needle_fn appears exactly once in the text above. "
    "Reply with that function's source code only.\n"
)
_WORDS = (
    "the system records temperature pressure volume latency throughput cache "
    "memory kernel buffer token context window batch scheduler worker queue "
    "index column row table record entry value signal noise sample measure "
    "calibrate validate verify accept reject commit checkpoint restore buffer "
    "linear attention state vector matrix tensor graph node edge weight bias "
    "layer norm scale shift rotate project embed decode encode draft verify "
    "region cluster packet frame segment route gateway client server "
    "upload download stream chunk block page slot cycle epoch phase step"
).split()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_rev() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return ""

    return {"commit": run("rev-parse", "HEAD"), "short": run("rev-parse", "--short", "HEAD")}


def http_json(url: str, payload: dict | None, timeout: float):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            parsed = json.loads(body) if body else None
            return resp.status, parsed
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = raw
        return exc.code, parsed


def tokenize_count(base: str, text: str, timeout: float) -> int | None:
    status, parsed = http_json(base + "/tokenize", {"prompt": text}, timeout)
    if status != 200 or not isinstance(parsed, dict):
        return None
    return parsed.get("count")


def filler(seed: int, target_chars: int) -> str:
    rng = random.Random(seed)
    parts = [f"Haystack document {seed:010d}. The following notes are filler.\n"]
    total = len(parts[0])
    while total < target_chars:
        words = [rng.choice(_WORDS) for _ in range(rng.randint(8, 16))]
        sent = " ".join(words)
        sent = sent[0].upper() + sent[1:] + ".\n"
        parts.append(sent)
        total += len(sent)
    return "".join(parts)


def build_user_message(seed: int, target_tokens: int, depth: float, chars_per_token: float) -> tuple[str, int]:
    """Return (message, needle_char_index). depth is the fraction of filler before the needle."""
    # Leave the question outside the target so the needle depth is meaningful.
    body_chars = max(200, int(target_tokens * chars_per_token) - len(QUESTION) - len(NEEDLE))
    hay = filler(seed, body_chars)
    cut = min(len(hay) - 1, max(0, int(len(hay) * depth)))
    # Snap to a newline so the function is its own block.
    nl = hay.rfind("\n", 0, cut)
    if nl >= 0:
        cut = nl + 1
    message = hay[:cut] + NEEDLE + "\n" + hay[cut:] + QUESTION
    return message, cut


def score(text: str) -> dict:
    body = text or ""
    return {
        "canary_hit": CANARY in body,
        "function_hit": "def needle_fn" in body,
        "checksum_hit": "4192" in body,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--target-tokens", type=int, default=240000,
                    help="near-limit user-message token target (context is 262144)")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    status, health = http_json(base + "/health", None, 15.0)
    if status != 200:
        print(f"FAIL: /health HTTP {status}")
        return 1

    model = args.model
    if model is None:
        _, models = http_json(base + "/v1/models", None, 15.0)
        ids = [m.get("id") for m in (models or {}).get("data", [])]
        model = ids[0] if ids else None
    if not model:
        print("FAIL: no model id")
        return 1

    sample = filler(1, 8000)
    sample_tokens = tokenize_count(base, sample, 60.0)
    if not sample_tokens:
        print("FAIL: /tokenize calibration failed")
        return 1
    chars_per_token = len(sample) / sample_tokens
    print(f"calibration chars/token={chars_per_token:.2f} model={model}")

    probes = [
        {"name": "control-4k-mid", "target_tokens": 4096, "depth": 0.50, "seed": 11001},
        {"name": "near-mid", "target_tokens": args.target_tokens, "depth": 0.50, "seed": 22002},
        {"name": "near-early", "target_tokens": args.target_tokens, "depth": 0.10, "seed": 22003},
        {"name": "near-late", "target_tokens": args.target_tokens, "depth": 0.90, "seed": 22004},
    ]

    report: dict = {
        "captured_at": utc_now(),
        "profile": args.profile,
        "base_url": base,
        "git": git_rev(),
        "model": model,
        "canary": CANARY,
        "needle": NEEDLE,
        "thinking_disabled": True,
        "target_tokens": args.target_tokens,
        "chars_per_token": round(chars_per_token, 3),
        "probes": [],
    }

    all_scored = True
    for spec in probes:
        h_status, _ = http_json(base + "/health", None, 15.0)
        if h_status != 200:
            print(f"STOP: /health HTTP {h_status} before {spec['name']}; not sending more")
            report["aborted"] = spec["name"]
            all_scored = False
            break
        target = spec["target_tokens"]
        message, needle_at = build_user_message(
            spec["seed"], target, spec["depth"], chars_per_token
        )
        raw_tokens = tokenize_count(base, message, 180.0)
        # One sizing adjustment before the request so a near-limit probe is
        # actually near the 262144 context, and not over it. Not a relaunch.
        if raw_tokens and abs(raw_tokens - target) / target > 0.08:
            scale = target / raw_tokens
            target = max(512, int(target * scale))
            # Stay under the context window even after the chat template.
            if spec["target_tokens"] >= 100000:
                target = min(target, 248000)
            message, needle_at = build_user_message(
                spec["seed"], target, spec["depth"], chars_per_token
            )
            raw_tokens = tokenize_count(base, message, 180.0)
            print(f"  [{spec['name']}] resized target={target} tokenize={raw_tokens}")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 512,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        print(
            f"  [{spec['name']}] sending chars={len(message)} "
            f"tokenize={raw_tokens} depth={spec['depth']}"
        )
        started = time.monotonic()
        try:
            status, resp = http_json(base + "/v1/chat/completions", payload, args.timeout)
            error = None
        except Exception as exc:
            status, resp, error = None, None, f"{type(exc).__name__}: {exc}"
        elapsed = round(time.monotonic() - started, 3)
        message_out = ""
        finish = None
        usage = None
        if isinstance(resp, dict):
            choice = (resp.get("choices") or [{}])[0]
            message_out = ((choice.get("message") or {}).get("content") or "")
            finish = choice.get("finish_reason")
            usage = resp.get("usage")
        hits = score(message_out)
        prompt_tokens = (usage or {}).get("prompt_tokens") if isinstance(usage, dict) else None
        depth_actual = None
        if prompt_tokens and raw_tokens:
            # Needle char index over the user message; approximate token depth.
            depth_actual = round(needle_at / max(len(message), 1), 3)
        rec = {
            "name": spec["name"],
            "target_tokens": spec["target_tokens"],
            "depth_requested": spec["depth"],
            "depth_char_fraction": depth_actual,
            "needle_char_index": needle_at,
            "user_chars": len(message),
            "tokenize_count": raw_tokens,
            "http_status": status,
            "error": error,
            "client_elapsed_s": elapsed,
            "finish_reason": finish,
            "usage": usage,
            "content": message_out[:2000],
            **hits,
            "pass": bool(hits["canary_hit"] and hits["function_hit"] and status == 200),
        }
        report["probes"].append(rec)
        print(
            f"  [{spec['name']}] pass={rec['pass']} http={status} "
            f"prompt_tokens={prompt_tokens} elapsed={elapsed}s finish={finish} "
            f"canary={hits['canary_hit']} fn={hits['function_hit']}"
        )
        if error:
            print(f"    error: {error}")
        if not rec["pass"]:
            all_scored = False

    report["all_pass"] = all_scored and bool(report["probes"])
    print("LONGCTX OK" if report["all_pass"] else "LONGCTX FAILED")

    if args.save_dir:
        out_dir = pathlib.Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.profile}-longctx-{utc_stamp()}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report saved: {path}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
