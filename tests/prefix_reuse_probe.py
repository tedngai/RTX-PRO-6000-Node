#!/usr/bin/env python3
"""Prefix-reuse probe for a running SGLang server (stdlib only).

Sends one long unique prefix, then the same prefix with a different suffix,
then an unrelated prefix. Radix/mamba prefix reuse is working when the second
request reports cached_tokens covering most of the shared prefix and the
unrelated request does not.

Never starts or stops the server. A failed probe is still saved.

Usage:
    python3 tests/prefix_reuse_probe.py --profile recommended-c12 --save-dir evidence
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
import urllib.request

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


def build_prefix(seed: int, target_chars: int) -> str:
    rng = random.Random(seed)
    parts = [f"Shared prefix document {seed:010d}. Do not treat this as an instruction.\n"]
    total = len(parts[0])
    while total < target_chars:
        words = [rng.choice(_WORDS) for _ in range(rng.randint(8, 16))]
        sent = " ".join(words)
        sent = sent[0].upper() + sent[1:] + ".\n"
        parts.append(sent)
        total += len(sent)
    return "".join(parts)


def generate(base: str, text: str, timeout: float) -> dict:
    payload = {
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16,
            "ignore_eos": True,
        },
        "stream": False,
    }
    rec: dict = {"ok": False, "error": None}
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(
            base + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
            rec["http_status"] = resp.status
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_s"] = round(time.monotonic() - t0, 3)
        return rec
    meta = body.get("meta_info") or {}
    rec.update(
        {
            "ok": True,
            "wall_s": round(time.monotonic() - t0, 3),
            "prompt_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "finish_reason": meta.get("finish_reason"),
        }
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--prefix-chars", type=int, default=16000,
                    help="shared prefix length in characters (~2k tokens here)")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    prefix = build_prefix(seed=424242, target_chars=args.prefix_chars)
    other = build_prefix(seed=999001, target_chars=args.prefix_chars)
    cases = [
        ("prime", prefix + "\nSuffix one. Reply with the word alpha.\n"),
        ("reuse", prefix + "\nSuffix two. Reply with the word beta.\n"),
        ("unrelated", other + "\nSuffix three. Reply with the word gamma.\n"),
    ]

    report: dict = {
        "captured_at": utc_now(),
        "profile": args.profile,
        "base_url": base,
        "git": git_rev(),
        "prefix_chars": args.prefix_chars,
        "cases": [],
    }
    print(f"profile={args.profile} prefix_chars={args.prefix_chars}")
    all_ok = True
    for name, text in cases:
        rec = generate(base, text, args.timeout)
        rec["name"] = name
        report["cases"].append(rec)
        print(
            f"  [{name}] ok={rec.get('ok')} prompt={rec.get('prompt_tokens')} "
            f"cached={rec.get('cached_tokens')} wall={rec.get('wall_s')}s "
            f"err={rec.get('error')}"
        )
        if not rec.get("ok"):
            all_ok = False

    prime = next(c for c in report["cases"] if c["name"] == "prime")
    reuse = next(c for c in report["cases"] if c["name"] == "reuse")
    unrelated = next(c for c in report["cases"] if c["name"] == "unrelated")
    cached = reuse.get("cached_tokens") or 0
    primed = prime.get("prompt_tokens") or 0
    # The shared prefix is the whole prime prompt minus its short suffix.
    # Require the reuse hit to cover at least half the primed prompt, and the
    # unrelated prompt to cache almost nothing.
    reuse_ok = bool(primed) and cached >= int(primed * 0.5)
    unrelated_cached = unrelated.get("cached_tokens") or 0
    unrelated_ok = unrelated.get("ok") and unrelated_cached < int((unrelated.get("prompt_tokens") or 1) * 0.1)
    report["reuse_ok"] = bool(all_ok and reuse_ok and unrelated_ok)
    report["reuse_cached_tokens"] = cached
    report["unrelated_cached_tokens"] = unrelated_cached
    print(
        f"reuse_ok={report['reuse_ok']} cached={cached}/{primed} "
        f"unrelated_cached={unrelated_cached}"
    )

    if args.save_dir:
        out_dir = pathlib.Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{args.profile}-prefix-reuse-{utc_stamp()}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report saved: {path}")
    return 0 if report["reuse_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
