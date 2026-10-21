"""E6 efficiency: latency (median / p95 / mean), throughput and peak memory of one engine on a fixed request set,
one request at a time, in one process (the Decision Index board protocol, on our hardware).

    python latency.py --engine crivo --opt path=... --requests lat1000.jsonl --out lat.json [--threads 1|16] [--warmup 20]

GPU: device-synchronized wall time per request, peak torch.cuda.max_memory_allocated and reserved.
CPU: wall time with torch.set_num_threads(--threads), peak RSS (resource.getrusage). The request set is the first
1,000 requests of a seeded shuffle of the E4 sample (orig), the same for every system. Also counted: encoder passes
(sequences) per request, so 1 pass (joint) vs n passes (per option, NLI) is explicit.
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from engines import Unsupported, make, options_of  # noqa: E402


def cpu_model():
    import platform
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        import subprocess
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    except Exception:  # noqa: BLE001
        return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True); ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--requests", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--per_request", help="also write one line per request: id, status (ok|unsupported), ms")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    opts = dict(o.split("=", 1) for o in a.opt)
    reqs = [json.loads(l) for l in open(a.requests)][:a.limit]
    t0 = time.time(); eng = make(a.engine, opts); load_s = time.time() - t0
    sync = getattr(eng, "sync", lambda: None)
    cuda = torch.cuda.is_available()
    for r in reqs[:a.warmup]:
        try:
            eng(r["state"], r["questions"]); sync()
        except Unsupported:
            pass
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    ts, unsup, opts_n, per = [], 0, [], []
    t_all = time.perf_counter()
    for r in reqs:
        t = time.perf_counter()
        try:
            eng(r["state"], r["questions"]); sync()
            ts.append((time.perf_counter() - t) * 1000)
            per.append({"id": r.get("id"), "status": "ok", "ms": round(ts[-1], 3)})
        except Unsupported as e:
            unsup += 1
            per.append({"id": r.get("id"), "status": "unsupported", "ms": None, "error": str(e)[:200]})
        opts_n.append(sum(len(options_of(q)[0]) for q in r["questions"].values()))
    wall = time.perf_counter() - t_all
    ts.sort()
    res = {"engine": getattr(eng, "name", a.engine), "device": "cuda" if cuda else "cpu",
           "gpu": torch.cuda.get_device_name() if cuda else None, "threads": torch.get_num_threads(),
           "requests": len(reqs), "timed": len(ts), "unsupported": unsup, "load_s": round(load_s, 1),
           "median_ms": statistics.median(ts), "p95_ms": ts[int(0.95 * (len(ts) - 1))], "mean_ms": statistics.mean(ts),
           "throughput_req_s": len(ts) / wall, "questions": sum(len(r["questions"]) for r in reqs),
           "options_per_request_mean": statistics.mean(opts_n),
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    res["cpu_model"] = cpu_model()
    if cuda:
        res["peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
        res["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    json.dump(res, open(a.out, "w"), indent=1)
    if a.per_request:
        with open(a.per_request, "w") as f:
            for x in per:
                f.write(json.dumps(x) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
