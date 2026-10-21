"""Evaluation sets for the A1/A2 grid (built once, identical for every arm and seed; no training on any of them).

    python build_eval.py --suite <kit/suite-0.3> --btzsc <btzsc data> --s1 <S1-Arena data> --outdir <dir>

  pub15.jsonl    15% of the requests of every public 0.3 benchmark in the index (seed 0, at most 600 per benchmark;
                 rebuilt GSM8K, no ForecastBench, the 442 exclusions dropped). Choice and noul questions.
  btzsc500.jsonl BTZSC, the harness's own max_samples=500 sample of every dataset (seed 0), fixed rendering.
  s1_60.jsonl    S1-Arena non-agentic, 60 items per task (seed 0).
The E4 files (orig, perm1-3, rev, yn) are reused as they are.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e4"))
from perturb import NOT_INDEX, read  # noqa: E402


def pub15(suite: Path, out: Path):
    excl = set(json.load(open(suite / "excluded-questions.json"))["rows"])
    by = defaultdict(list)
    for f, base in [(suite / "selected-rows.jsonl.gz", True), (suite / "added-rows.jsonl.gz", False), (suite / "gsm8k-rows.jsonl.gz", False)]:
        for r in read(f):
            ev = r.get("_evaluation", {})
            ds = ev.get("dataset") or r.get("family")
            if ds in NOT_INDEX or (base and ds == "GSM8K") or ev.get("run_id") in excl:
                continue
            by[ds].append(r)
    n = 0
    with open(out, "w") as fh:
        for ds in sorted(by):
            rows = by[ds]
            k = min(600, max(1, round(0.15 * len(rows))))
            for r in random.Random(0).sample(rows, k):
                fh.write(json.dumps({"id": r["_evaluation"]["run_id"], "family": ds, "state": r["state"], "questions": r["questions"],
                                     "expected": r.get("expected", {})}, ensure_ascii=False) + "\n")
                n += 1
    print("pub15", n, "requests,", len(by), "benchmarks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True); ap.add_argument("--btzsc", required=True); ap.add_argument("--s1", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
    pub15(Path(a.suite), od / "pub15.jsonl")
    import importlib.util

    def mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m
    bt = mod("btzsc_adapter", HERE.parent / "btzsc" / "adapter.py")
    with open(od / "btzsc500.jsonl", "w") as fh:
        for d in bt.DATASETS:
            ds = bt.load(d, a.btzsc, 500)
            for j in range(len(ds["texts"])):
                fh.write(json.dumps(bt.request(ds, j), ensure_ascii=False) + "\n")
    s1 = mod("s1_adapter", HERE.parent / "s1arena" / "adapter.py")
    by = defaultdict(list)
    for it in s1.items(a.s1, s1.NON_AGENTIC):
        by[it["task"]].append(it)
    with open(od / "s1_60.jsonl", "w") as fh:
        for t in sorted(by):
            for it in random.Random(0).sample(by[t], min(60, len(by[t]))):
                fh.write(json.dumps(s1.to_di(it), ensure_ascii=False) + "\n")
    print("done")


if __name__ == "__main__":
    main()
