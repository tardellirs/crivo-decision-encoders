"""E6 summary over the same request subset: for each system (a latency.py --out json + its --per_request file),
median / p95 / mean ms on (a) every request it answered and (b) the requests EVERY listed system answered; skipped
(unsupported) counts; CPU model, thread count, GPU name.

    python gpu3/e6_common.py --dir results_gpu3/e6cpu --out results_gpu3/e6cpu/summary.json   # all *.json with *.per.jsonl
"""
import argparse
import json
import statistics
from pathlib import Path


def stats(ms):
    ms = sorted(ms)
    if not ms:
        return {"n": 0}
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 2), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 2),
            "mean_ms": round(statistics.mean(ms), 2), "sum_s": round(sum(ms) / 1000, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir); sysd = {}
    for per in sorted(d.glob("*.per.jsonl")):
        name = per.name[:-len(".per.jsonl")]
        meta = json.loads((d / f"{name}.json").read_text())
        rec = {json.loads(l)["id"]: json.loads(l) for l in open(per)}
        sysd[name] = (meta, rec)
    common = set.intersection(*[{i for i, r in rec.items() if r["status"] == "ok"} for _, rec in sysd.values()]) if sysd else set()
    out = {"common_requests": len(common), "systems": {}}
    for name, (meta, rec) in sysd.items():
        ok = [r["ms"] for r in rec.values() if r["status"] == "ok"]
        out["systems"][name] = {"engine": meta.get("engine"), "device": meta.get("device"), "gpu": meta.get("gpu"),
                                "cpu_model": meta.get("cpu_model"), "threads": meta.get("threads"),
                                "requests": len(rec), "skipped_unsupported": sum(r["status"] != "ok" for r in rec.values()),
                                "all_answered": stats(ok), "common_subset": stats([rec[i]["ms"] for i in common]),
                                "peak_rss_mb": meta.get("peak_rss_mb"), "peak_alloc_mb": meta.get("peak_alloc_mb")}
    Path(a.out).write_text(json.dumps(out, indent=1))
    w = max([len(n) for n in out["systems"]] + [6])
    print(f"common subset: {len(common)} requests")
    print(f"{'system':{w}s} {'req':>5s} {'skip':>5s} | {'med':>9s} {'p95':>9s} {'mean':>9s} (all answered) | {'med':>9s} {'p95':>9s} {'mean':>9s} (common)")
    for n, s in out["systems"].items():
        A, C = s["all_answered"], s["common_subset"]
        print(f"{n:{w}s} {s['requests']:5d} {s['skipped_unsupported']:5d} | {A.get('median_ms', 0):9.1f} {A.get('p95_ms', 0):9.1f} {A.get('mean_ms', 0):9.1f} | "
              f"{C.get('median_ms', 0):9.1f} {C.get('p95_ms', 0):9.1f} {C.get('mean_ms', 0):9.1f}")


if __name__ == "__main__":
    main()
