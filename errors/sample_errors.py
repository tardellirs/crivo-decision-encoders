"""Draw a fixed stratified sample of Crivo-68M errors on the public suite for hand coding (A5).

Two errors per MAIN benchmark (seed 0), each shown truncated (instructions 160 chars, state 360, options 70) for
reading only. The coded categories live in errors/sample68m_coded.tsv (run_id, key, category); no benchmark text is
stored in the repo.

    python sample_errors.py --tab ~/e5/tab --suite $KIT/suite-0.3 --per 2 > ~/e5/sample68m.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "e5"))
from calib import load  # noqa: E402
from decision_index.suite.io import Suite  # noqa: E402


def cut(x, n):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, sort_keys=True)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + " …"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--model", default="c68m")
    a = ap.parse_args()
    q, P = load(a.tab, [a.model, "c17m", "c1b", "prumo"])
    rng = np.random.default_rng(0)
    M = q.in_index.to_numpy()
    err = M & ~P[a.model]["correct"]
    pick = []
    for b in sorted(q.bench[M].unique()):
        idx = np.flatnonzero(err & (q.bench.to_numpy() == b))
        if len(idx):
            pick += list(rng.choice(idx, min(a.per, len(idx)), replace=False))
    want = {(q.run_id[i], q.key[i]): i for i in pick}
    rids = {r for r, _ in want}
    rows = {}
    for r in Suite(a.suite, "0.3").rows(apply_exclusions=True):
        if r["_evaluation"]["run_id"] in rids:
            rows[r["_evaluation"]["run_id"]] = r
    for (rid, key), i in sorted(want.items(), key=lambda t: t[1]):
        r = rows[rid]; qq = r["questions"][key]
        opts = list(qq["criteria"]) if qq["type"] == "choice" else ["false", "true"]
        pi = int(P[a.model]["pred_idx"][i]); gi = int(q.gold_idx[i])
        def opt(j):
            if qq["type"] != "choice":
                return opts[j]
            return f"{opts[j]} = {cut(qq['criteria'][opts[j]], 70)}"
        print(f"### {q.bench_name[i]} | {rid} | {key} | {qq['type']} k={q.k[i]} tok={q.state_tokens[i]} "
              f"conf={P[a.model]['conf'][i]:.2f} 17m={'R' if P['c17m']['correct'][i] else 'W'} "
              f"1b={'R' if P['c1b']['correct'][i] else 'W'} prumo={'R' if P['prumo']['correct'][i] else 'W'}")
        print("  INSTR:", cut(qq.get("instructions") or "", 160))
        print("  STATE:", cut(r["state"], 360))
        print("  GOLD :", opt(gi))
        print("  PRED :", opt(pi) if pi >= 0 else "-")


if __name__ == "__main__":
    main()
