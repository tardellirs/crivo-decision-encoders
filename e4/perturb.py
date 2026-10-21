"""E4 robustness: option-order permutations and the yes/no swap on a stratified sample of the public 0.3 suite.

    python perturb.py sample --suite <kit/suite-0.3> --per_bench 100 --out e4_orig.jsonl
    python perturb.py make   --rows e4_orig.jsonl --outdir e4/          # writes perm1..3, rev, yn (same ids + suffix)
    python perturb.py flips  --rows e4_orig.jsonl --outdir e4/ --results <dir with orig.jsonl perm1.jsonl ...>

Sample: per public benchmark (`_evaluation.dataset`, 0.3 rows: rebuilt GSM8K, no ForecastBench, 442 exclusions
dropped), up to --per_bench requests and --q_per_bench questions (seed 0) whose questions are all `choice` with 2-255 options and <= 32 questions; benchmarks outside the index
(MMLU, ARC, SimpleBench, RouterBench, SGD) are left out.
Perturbations (only the order of the `criteria` dict changes; keys, descriptions, state, instructions unchanged):
  perm1, perm2, perm3  seeded random permutations (seed = hash of request id and k)
  rev                  reversed order
  yn                   only binary yes/no-type questions (keys or descriptions yes/no or true/false): the two options
                       swapped; other questions keep their order. Requests without such a question are not in yn.
Flip rate = share of questions whose chosen key differs from the chosen key on the original order. Also reported:
accuracy per variant (gold = `expected`), and the share of questions answered identically under all 3 permutations.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

NOT_INDEX = {"ForecastBench", "MMLU", "ARC-Easy", "ARC-Challenge", "SimpleBench", "RouterBench", "SGD/SGD-X"}
YN = {"yes", "no", "true", "false", "y", "n", "yes.", "no."}


def is_yn(q):
    if q["type"] != "choice" or len(q["criteria"]) != 2:
        return False
    ks = [str(k).strip().lower() for k in q["criteria"]]
    ds = [str(v).strip().lower() if v is not None else "" for v in q["criteria"].values()]
    return set(ks) <= YN or set(ds) <= YN


def read(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def sample(a):
    s = Path(a.suite)
    excl = set(json.load(open(s / "excluded-questions.json"))["rows"])
    by = defaultdict(list)
    files = [(s / "selected-rows.jsonl.gz", True), (s / "added-rows.jsonl.gz", False), (s / "gsm8k-rows.jsonl.gz", False)]
    for f, base in files:
        for r in read(f):
            ev = r.get("_evaluation", {})
            ds = ev.get("dataset") or r.get("family")
            if ds in NOT_INDEX or (base and ds == "GSM8K"):
                continue
            if ev.get("run_id") in excl or r.get("id") in excl:
                continue
            qs = r["questions"]
            if not qs or len(qs) > 32 or any(q["type"] != "choice" or not 2 <= len(q["criteria"]) <= 255 for q in qs.values()):
                continue
            by[ds].append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(a.out, "w")
    n = 0
    for ds in sorted(by):
        rows = by[ds]
        pick, nq = [], 0
        for r in random.Random(0).sample(rows, len(rows)):
            if len(pick) >= a.per_bench or nq >= a.q_per_bench:
                break
            pick.append(r); nq += len(r["questions"])
        for r in pick:
            out.write(json.dumps({"id": r.get("_evaluation", {}).get("run_id") or r["id"], "family": ds, "state": r["state"],
                                  "questions": r["questions"], "expected": r.get("expected", {})}, ensure_ascii=False) + "\n")
            n += 1
    print(n, "requests from", len(by), "benchmarks")


def permute(q, order):
    keys = list(q["criteria"])
    return {**q, "criteria": {keys[i]: q["criteria"][keys[i]] for i in order}}


def make(a):
    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
    fs = {v: open(od / f"{v}.jsonl", "w") for v in ("orig", "perm1", "perm2", "perm3", "rev", "yn")}
    for r in read(a.rows):
        fs["orig"].write(json.dumps(r, ensure_ascii=False) + "\n")
        for k in (1, 2, 3):
            qs = {}
            for qk, q in r["questions"].items():
                n = len(q["criteria"])
                order = list(range(n))
                rng = random.Random(int(hashlib.sha256(f"{r['id']}|{qk}|{k}".encode()).hexdigest()[:12], 16))
                while n > 1 and order == list(range(n)):
                    rng.shuffle(order)
                qs[qk] = permute(q, order)
            fs[f"perm{k}"].write(json.dumps({**r, "questions": qs}, ensure_ascii=False) + "\n")
        fs["rev"].write(json.dumps({**r, "questions": {qk: permute(q, list(range(len(q["criteria"])))[::-1])
                                                        for qk, q in r["questions"].items()}}, ensure_ascii=False) + "\n")
        if any(is_yn(q) for q in r["questions"].values()):
            fs["yn"].write(json.dumps({**r, "questions": {qk: permute(q, [1, 0]) if is_yn(q) else q
                                                           for qk, q in r["questions"].items()}}, ensure_ascii=False) + "\n")


def flips(a):
    rows = {r["id"]: r for r in read(a.rows)}
    rd = Path(a.results)
    def load(v):
        out = {}
        p = rd / f"{v}.jsonl"
        if not p.exists():
            return None
        for x in read(p):
            if x.get("status") == "ok":
                out[x["id"]] = {k: v["choice"] for k, v in x["response"]["answers"].items()}
        return out
    base = load("orig")
    res = {}
    for v in ("perm1", "perm2", "perm3", "rev", "yn"):
        cur = load(v)
        if cur is None:
            continue
        nq = fl = hit0 = hit1 = 0
        per = defaultdict(lambda: [0, 0])
        for rid, ans in cur.items():
            if rid not in base:
                continue
            for qk, c in ans.items():
                if v == "yn" and not is_yn(rows[rid]["questions"][qk]):
                    continue
                nq += 1; f = c != base[rid].get(qk); fl += f
                g = rows[rid].get("expected", {}).get(qk)
                hit0 += base[rid].get(qk) == g; hit1 += c == g
                per[rows[rid]["family"]][0] += 1; per[rows[rid]["family"]][1] += f
        res[v] = {"questions": nq, "flip_rate": fl / max(1, nq), "acc_orig": hit0 / max(1, nq), "acc_perturbed": hit1 / max(1, nq),
                  "per_benchmark_flip": {k: round(b / a_, 4) for k, (a_, b) in sorted(per.items())}}
    ps = [load(f"perm{k}") for k in (1, 2, 3)]
    if all(p is not None for p in ps):
        same = tot = 0
        for rid, ans in base.items():
            for qk, c in ans.items():
                vs = [p.get(rid, {}).get(qk) for p in ps]
                if None in vs:
                    continue
                tot += 1; same += all(x == c for x in vs)
        res["consistent_all_perms"] = same / max(1, tot)
    json.dump(res, open(rd / "flips.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_benchmark_flip"}) for k, v in res.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "make", "flips"])
    ap.add_argument("--suite"); ap.add_argument("--per_bench", type=int, default=100); ap.add_argument("--q_per_bench", type=int, default=150); ap.add_argument("--out")
    ap.add_argument("--rows"); ap.add_argument("--outdir"); ap.add_argument("--results")
    a = ap.parse_args()
    {"sample": sample, "make": make, "flips": flips}[a.cmd](a)


if __name__ == "__main__":
    main()
