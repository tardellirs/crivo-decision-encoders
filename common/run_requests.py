"""Run any engine of engines.py over a request file (Decision Index wire format), resumable.

    python run_requests.py --engine crivo --opt path=tardellirs/crivo-17m --requests req.jsonl --out res.jsonl \
        [--limit N] [--shard i/n] [--threads 16]

Output, one line per request: {"id", "status": ok|unsupported|error, "response": {"answers": ...}, "total_wall_ms",
"error"}. Wall time = engine call incl. tokenization, device-synchronized. Requests already in --out with status ok
or unsupported are skipped (errors are retried). Answers are validated (keys, types, probability sums).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engines import Unsupported, make, options_of  # noqa: E402


def validate(questions, answers):
    if set(answers) != set(questions):
        raise ValueError("question keys mismatch")
    for k, q in questions.items():
        a = answers[k]
        if a.get("type") != q["type"]:
            raise ValueError(f"type mismatch on {k}")
        if q["type"] in ("choice", "score"):
            keys, _ = options_of(q)
            p = a.get("probabilities", {})
            if set(map(str, p)) != set(keys) or abs(sum(p.values()) - 1) > 0.01 or any(not math.isfinite(v) for v in p.values()):
                raise ValueError(f"bad distribution on {k}")
        elif not (0 <= a["noul"] <= 1):
            raise ValueError(f"bad noul on {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0, help="requests per engine.batch_call (engines that have it); "
                    "total_wall_ms is then the chunk time / chunk size (throughput, not latency: use e6/latency.py)")
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    opts = dict(o.split("=", 1) for o in a.opt)
    si, sn = map(int, a.shard.split("/"))
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("status") in ("ok", "unsupported"):
                done.add(r["id"])
    reqs = [json.loads(l) for l in open(a.requests)]
    reqs = [r for i, r in enumerate(reqs) if i % sn == si]
    if a.limit:
        reqs = reqs[:a.limit]
    todo = [r for r in reqs if r["id"] not in done]
    print(f"{len(reqs)} requests, {len(todo)} to run", flush=True)
    t0 = time.time()
    eng = make(a.engine, opts)
    print(f"loaded {getattr(eng, 'name', a.engine)} in {time.time() - t0:.1f}s", flush=True)
    sync = getattr(eng, "sync", lambda: None)
    if a.batch and hasattr(eng, "batch_call"):
        with open(a.out, "a") as f:
            for c0 in range(0, len(todo), a.batch):
                chunk = todo[c0:c0 + a.batch]
                t = time.perf_counter()
                try:
                    outs = eng.batch_call([(r["state"], r["questions"]) for r in chunk]); sync()
                except Exception as e:  # noqa: BLE001
                    outs = [e] * len(chunk)
                ms = (time.perf_counter() - t) * 1000 / len(chunk)
                for r, ans in zip(chunk, outs):
                    rec = {"id": r["id"], "total_wall_ms": round(ms, 3), "batched": a.batch}
                    if isinstance(ans, Unsupported):
                        rec.update(status="unsupported", error=str(ans)[:300])
                    elif isinstance(ans, Exception):
                        rec.update(status="error", error=f"{type(ans).__name__}: {ans}"[:300])
                    else:
                        try:
                            validate(r["questions"], ans); rec.update(status="ok", response={"answers": ans})
                        except Exception as e:  # noqa: BLE001
                            rec.update(status="error", error=str(e)[:300])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if (c0 // a.batch) % 20 == 0:
                    print(f"{c0 + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        print(f"done {len(todo)} in {time.time() - t0:.0f}s", flush=True)
        return
    with open(a.out, "a") as f:
        for n, r in enumerate(todo):
            if hasattr(eng, "family"):
                eng.family = r.get("family")
            rec = {"id": r["id"]}
            t = time.perf_counter()
            try:
                ans = eng(r["state"], r["questions"])
                sync()
                validate(r["questions"], ans)
                rec.update(status="ok", response={"answers": ans})
            except Unsupported as e:
                rec.update(status="unsupported", error=str(e)[:300])
            except Exception as e:  # noqa: BLE001
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[:300], tb=traceback.format_exc()[-800:])
            rec["total_wall_ms"] = round((time.perf_counter() - t) * 1000, 2)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                f.flush()
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(todo)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
