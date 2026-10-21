"""Check that engine.batch_call (many requests per call) gives the same answers as engine(state, questions).

    python check_batch.py --engine crivo --opt path=... --requests req.jsonl [--limit 200] [--batch 64]

Prints the share of questions with the same argmax, and the max / mean absolute probability difference.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engines import Unsupported, make  # noqa: E402


def probs(a):
    if a["type"] == "noul":
        return {"true": a["noul"]}
    return a["probabilities"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True); ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--requests", required=True); ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    eng = make(a.engine, dict(o.split("=", 1) for o in a.opt))
    reqs = [json.loads(l) for l in open(a.requests)][:a.limit]
    single = []
    for r in reqs:
        try:
            single.append(eng(r["state"], r["questions"]))
        except Unsupported as e:
            single.append(e)
    batched = []
    for i in range(0, len(reqs), a.batch):
        batched += eng.batch_call([(r["state"], r["questions"]) for r in reqs[i:i + a.batch]])
    same = n = 0; mx = tot = 0.0; cnt = 0; unsup_mismatch = 0
    for s, b in zip(single, batched):
        if isinstance(s, Exception) or isinstance(b, Exception):
            unsup_mismatch += isinstance(s, Exception) != isinstance(b, Exception)
            continue
        for k in s:
            n += 1
            ps, pb = probs(s[k]), probs(b[k])
            same += max(ps, key=ps.get) == max(pb, key=pb.get) if s[k]["type"] != "noul" else (ps["true"] >= .5) == (pb["true"] >= .5)
            for kk in ps:
                d = abs(ps[kk] - pb[kk]); mx = max(mx, d); tot += d; cnt += 1
    print(json.dumps({"engine": getattr(eng, "name", a.engine), "requests": len(reqs), "questions": n,
                      "same_decision": same / max(1, n), "max_abs_dp": mx, "mean_abs_dp": tot / max(1, cnt),
                      "unsupported_mismatch": unsup_mismatch}))


if __name__ == "__main__":
    main()
