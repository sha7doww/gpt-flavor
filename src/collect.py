#!/usr/bin/env python3
"""
Collect GPT 中文味 corpus via any OpenAI-compatible API, concurrently.

Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL for non-default endpoints
or third-party proxies) in .env before running.

Usage:
  python collect.py probe                                       # list models
  python collect.py run                                         # full async collection
  python collect.py run --concurrency 50                        # tune concurrency
  python collect.py run --models gpt-5.4 --conditions C_poster  # subset
  python collect.py run --runs-per-seed 2                       # tune runs
  python collect.py run --dry-run                               # print plan only
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm as atqdm

ROOT = Path(__file__).parent.parent
SEEDS_FILE = ROOT / "seeds" / "seeds.json"
PROMPTS_FILE = ROOT / "seeds" / "system_prompts.json"
OUT_DIR = ROOT / "data"

PRICING_USD_PER_1M = {
    "default": {"in": 2.50, "out": 10.00},
}


def _env():
    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    if not key or key == "sk-your-key-here":
        sys.exit("OPENAI_API_KEY missing. Copy .env.example to .env and fill it in.")
    return key, base


def probe():
    key, base = _env()
    client = OpenAI(api_key=key, base_url=base)
    try:
        resp = client.models.list()
    except Exception as e:
        print(f"models.list() failed: {e}")
        return
    ids = sorted(m.id for m in resp.data)
    gpt_related = [m for m in ids if "gpt" in m.lower() or m.startswith("o")]
    print(f"Found {len(ids)} total models.\n")
    print("GPT / OpenAI-family models:")
    for m in gpt_related:
        print(f"  {m}")


def record_id(model, condition, seed, run):
    return hashlib.sha256(f"{model}|{condition}|{seed}|{run}".encode()).hexdigest()[:16]


def load_existing(path):
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                pass
    return ids


def estimate_cost(usage, model):
    p = PRICING_USD_PER_1M.get(model, PRICING_USD_PER_1M["default"])
    return (usage.prompt_tokens * p["in"] + usage.completion_tokens * p["out"]) / 1e6


def build_jobs(models, conditions, runs_per_seed):
    seeds_by_cat = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))
    system_prompts = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    jobs = []
    paths = {}
    for model in models:
        for condition in conditions:
            if condition not in system_prompts:
                print(f"Skipping unknown condition: {condition}")
                continue
            out_path = OUT_DIR / model / f"{condition}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            existing = load_existing(out_path)
            paths[(model, condition)] = out_path
            for category, seeds in seeds_by_cat.items():
                for seed in seeds:
                    for run in range(runs_per_seed):
                        rid = record_id(model, condition, seed, run)
                        if rid in existing:
                            continue
                        jobs.append({
                            "rid": rid,
                            "model": model,
                            "condition": condition,
                            "category": category,
                            "seed": seed,
                            "run": run,
                            "system": system_prompts[condition],
                        })
    return jobs, paths


async def call_api(client, model, sys_prompt, user_msg):
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_msg})
    try:
        resp = await client.chat.completions.create(
            model=model, messages=messages, temperature=1.0
        )
        return resp, False
    except Exception as e:
        msg = str(e).lower()
        if "temperature" in msg or "unsupported" in msg:
            resp = await client.chat.completions.create(model=model, messages=messages)
            return resp, True
        raise


async def worker(client, sem, job, state, queue):
    async with sem:
        if state["stop"]:
            return
        if state["cost"] >= state["budget"]:
            state["stop"] = True
            return
        try:
            resp, fallback_no_temp = await call_api(
                client, job["model"], job["system"], job["seed"]
            )
        except Exception as e:
            state["err"] += 1
            short = str(e).replace("\n", " ")[:140]
            state["errors"].append(f"{job['model']}/{job['condition']}/{job['seed'][:16]}: {short}")
            return
        reply = resp.choices[0].message.content or ""
        cost = estimate_cost(resp.usage, job["model"])
        state["cost"] += cost
        state["ok"] += 1
        if fallback_no_temp:
            state["fallback"] = state.get("fallback", 0) + 1
        rec = {
            "id": job["rid"],
            "model": job["model"],
            "condition": job["condition"],
            "category": job["category"],
            "seed": job["seed"],
            "run": job["run"],
            "system": job["system"],
            "reply": reply,
            "fallback_no_temp": fallback_no_temp,
            "usage": {
                "prompt": resp.usage.prompt_tokens,
                "completion": resp.usage.completion_tokens,
                "total": resp.usage.total_tokens,
            },
            "cost_usd_approx": round(cost, 6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await queue.put((job["model"], job["condition"], rec))


async def writer(queue, paths, total, state):
    handles = {}
    pbar = atqdm(total=total, desc="collecting", mininterval=0.5)
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            model, condition, rec = item
            key = (model, condition)
            if key not in handles:
                handles[key] = paths[key].open("a", encoding="utf-8")
            handles[key].write(json.dumps(rec, ensure_ascii=False) + "\n")
            handles[key].flush()
            pbar.update(1)
            pbar.set_postfix_str(
                f"${state['cost']:.3f} | ok {state['ok']} | err {state['err']}"
            )
    finally:
        for fh in handles.values():
            fh.close()
        pbar.close()


async def run_async(models, conditions, runs_per_seed, budget, concurrency, dry_run):
    jobs, paths = build_jobs(models, conditions, runs_per_seed)
    planned = sum(len(json.loads(SEEDS_FILE.read_text(encoding="utf-8"))[c]) for c in json.loads(SEEDS_FILE.read_text(encoding="utf-8")))
    planned = planned * len(conditions) * len(models) * runs_per_seed
    print(f"Plan: {planned} total, {len(jobs)} new (skipping {planned - len(jobs)} already done), concurrency={concurrency}")
    if dry_run or not jobs:
        return

    key, base = _env()
    client = AsyncOpenAI(api_key=key, base_url=base, timeout=300.0, max_retries=2)
    sem = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue = asyncio.Queue()
    state = {"cost": 0.0, "ok": 0, "err": 0, "fallback": 0, "stop": False, "budget": budget, "errors": []}

    writer_task = asyncio.create_task(writer(queue, paths, len(jobs), state))

    await asyncio.gather(*(worker(client, sem, j, state, queue) for j in jobs),
                         return_exceptions=True)
    await queue.put(None)
    await writer_task
    await client.close()

    print(
        f"\nDone. ok={state['ok']} err={state['err']} "
        f"fallback_no_temp={state['fallback']} cost≈${state['cost']:.3f}"
    )
    if state["errors"]:
        print(f"\nFirst 10 errors:")
        for e in state["errors"][:10]:
            print(f"  {e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")

    runp = sub.add_parser("run")
    runp.add_argument("--models", nargs="+", default=None)
    runp.add_argument("--conditions", nargs="+",
                      default=["A_empty", "B_listener", "C_poster"])
    runp.add_argument("--runs-per-seed", type=int, default=3)
    runp.add_argument("--budget", type=float, default=None)
    runp.add_argument("--concurrency", type=int, default=30)
    runp.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.cmd == "probe":
        probe()
        return

    load_dotenv(ROOT / ".env")
    if args.models is None:
        primary = os.environ.get("PRIMARY_MODEL", "gpt-5.4").strip()
        compare = os.environ.get("COMPARE_MODEL", "").strip()
        args.models = [primary] + ([compare] if compare else [])
    budget = args.budget if args.budget is not None else float(
        os.environ.get("BUDGET_USD", "20")
    )
    asyncio.run(run_async(
        args.models, args.conditions, args.runs_per_seed,
        budget, args.concurrency, args.dry_run,
    ))


if __name__ == "__main__":
    main()
