"""Rebuild the E4 sample (`e4_orig.jsonl` = `~/e4/req/orig.jsonl` on the GPU box) on a small CPU machine.

Same selection as `e4/perturb.py sample` (same files, filters, per-benchmark shuffle with random.Random(0), same
stopping rule, same output dict and json.dumps), but streamed: pass 1 keeps only (file, line, n_questions) per
eligible row, pass 2 re-reads the files and keeps the ~2.8k picked rows. random.sample(rows, n) depends only on n, so
shuffling the index list gives the same order as perturb.py. The md5 of the output must equal the VMs' orig.jsonl
(4643cfec...).

    python stats/e4_rows.py --suite ../../kit/suite-0.3 --out stats/gpu1/req/e4_orig.jsonl
"""
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e4"))
from perturb import NOT_INDEX, read  # noqa: E402


def eligible(s, excl):
    files = [(s / "selected-rows.jsonl.gz", True), (s / "added-rows.jsonl.gz", False), (s / "gsm8k-rows.jsonl.gz", False)]
    for fi, (f, base) in enumerate(files):
        for li, r in enumerate(read(f)):
            ev = r.get("_evaluation", {})
            ds = ev.get("dataset") or r.get("family")
            if ds in NOT_INDEX or (base and ds == "GSM8K"):
                continue
            if ev.get("run_id") in excl or r.get("id") in excl:
                continue
            qs = r["questions"]
            if not qs or len(qs) > 32 or any(q["type"] != "choice" or not 2 <= len(q["criteria"]) <= 255 for q in qs.values()):
                continue
            yield fi, li, ds, len(qs), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per_bench", type=int, default=100); ap.add_argument("--q_per_bench", type=int, default=150)
    a = ap.parse_args()
    s = Path(a.suite)
    excl = set(json.load(open(s / "excluded-questions.json"))["rows"])
    by = defaultdict(list)
    for fi, li, ds, nq, _ in eligible(s, excl):
        by[ds].append((fi, li, nq))
    order = []                                   # (fi, li) in output order
    for ds in sorted(by):
        rows = by[ds]
        nq = 0; pick = []
        for i in random.Random(0).sample(range(len(rows)), len(rows)):
            if len(pick) >= a.per_bench or nq >= a.q_per_bench:
                break
            pick.append(rows[i][:2]); nq += rows[i][2]
        order += pick
    want = set(order)
    keep = {}
    for fi, li, ds, _, r in eligible(s, excl):
        if (fi, li) in want:
            keep[(fi, li)] = json.dumps({"id": r.get("_evaluation", {}).get("run_id") or r["id"], "family": ds,
                                         "state": r["state"], "questions": r["questions"],
                                         "expected": r.get("expected", {})}, ensure_ascii=False) + "\n"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for k in order:
            f.write(keep[k])
    md5 = hashlib.md5(open(a.out, "rb").read()).hexdigest()
    print(len(order), "requests from", len(by), "benchmarks; md5", md5)


if __name__ == "__main__":
    main()
