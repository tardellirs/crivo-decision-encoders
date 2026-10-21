"""Summarize the A1/A2 grid: per arm mean +- sd over 3 seeds, and each arm minus J (joint, Brier, KD, reranker init)
with (a) a seed-level Welch 95% CI and (b) an item-level paired bootstrap on pub15 question correctness (each
question's correctness averaged over the 3 seeds; questions resampled, B = 2,000, seed 0).

    python summarize.py <results/a1a2>   -> prints a table and writes summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

O = Path(sys.argv[1])
BT_UNSEEN = ["appreviews", "empathetic", "trueteacher", "capsotu"]   # unseen by the v2 mix (btzsc/contamination.md)
HERE = Path(__file__).resolve().parent
S1_CLASS = json.load(open(HERE.parent / "s1arena" / "contamination.json"))


def metrics(ev: Path):
    m = {}
    p = json.load(open(ev / "pub15.score.json"))["summary"]
    m.update(pub15_skill=p["mean_skill"], pub15_ece=p["ece_choice"], pub15_brier=p["brier_choice"])
    b = json.load(open(ev / "btzsc500.score.json"))
    m["btzsc_all_f1"] = b["summary"]["overall_macro_f1"]
    u = [b["by_dataset"][d]["macro_f1"] for d in BT_UNSEEN if d in b["by_dataset"]]
    m["btzsc_unseen_f1"] = float(np.mean(u)) if u else None
    s = json.load(open(ev / "s1_60.score.json"))
    m["s1_acc"] = s["summary"]["accuracy"]; m["s1_brier"] = s["summary"]["brier"]
    un = [v["accuracy"] for t, v in s["tasks"].items() if S1_CLASS.get(t, {}).get("class") == "unseen"]
    m["s1_unseen_acc"] = float(np.mean(un)) if un else None
    f = json.load(open(ev / "e4" / "flips.json"))
    m["flip_perm"] = float(np.mean([f[v]["flip_rate"] for v in ("perm1", "perm2", "perm3") if v in f]))
    m["flip_rev"] = f.get("rev", {}).get("flip_rate"); m["flip_yn"] = f.get("yn", {}).get("flip_rate")
    return m


def correct(ev: Path):
    """pub15 question -> correct (0/1)."""
    res = {}
    for l in open(ev / "pub15.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok":
            res[r["id"]] = r["response"]["answers"]
    out = {}
    for l in open(Path(__import__("os").environ.get("EVALDIR", str(Path.home() / "evalsets"))) / "pub15.jsonl"):
        r = json.loads(l)
        for k, q in r["questions"].items():
            g = (r.get("expected") or {}).get(k)
            if g is None:
                continue
            x = res.get(r["id"], {}).get(k)
            if q["type"] == "noul":
                gold = g if isinstance(g, bool) else str(g).lower() in ("true", "1", "yes")
                out[(r["id"], k)] = float(x is not None and (x["noul"] >= 0.5) == gold)
            else:
                out[(r["id"], k)] = float(x is not None and x["choice"] == g)
    return out


arms = [l.split("\t")[0] for l in open(HERE / "grid.tsv") if l.strip() and not l.startswith("#")]
M, C = {}, {}
for a in arms:
    runs = [O / "ev" / f"{a}_s{s}" for s in range(3) if (O / "ev" / f"{a}_s{s}" / "DONE").exists()]
    if not runs:
        continue
    M[a] = [metrics(r) for r in runs]
    cs = [correct(r) for r in runs]
    keys = sorted(set.intersection(*[set(c) for c in cs]))
    C[a] = {k: np.mean([c[k] for c in cs]) for k in keys}
out = {"arms": {}, "vs_J": {}}
keys = list(next(iter(M.values()))[0]) if M else []
for a, ms in M.items():
    def agg(v):
        v = [x for x in v if x is not None]
        return {"mean": float(np.mean(v)) if v else None, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n_seeds": len(v)}
    out["arms"][a] = {k: agg([m[k] for m in ms]) for k in keys}
rng = np.random.default_rng(0)
if "J" in M:
    for a in M:
        if a == "J":
            continue
        d = {}
        for k in keys:
            x = [m[k] for m in M[a] if m[k] is not None]; y = [m[k] for m in M["J"] if m[k] is not None]
            if len(x) > 1 and len(y) > 1:
                t = stats.ttest_ind(x, y, equal_var=False)
                ci = t.confidence_interval() if hasattr(t, "confidence_interval") else (None, None)
                d[k] = {"diff": float(np.mean(x) - np.mean(y)), "welch_ci95": [float(ci[0]), float(ci[1])], "p": float(t.pvalue)}
        common = sorted(set(C[a]) & set(C["J"]))
        diff = np.array([C[a][k] - C["J"][k] for k in common])
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        d["pub15_question_acc_paired"] = {"diff": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                          "questions": len(common)}
        out["vs_J"][a] = d
json.dump(out, open(O / "summary.json", "w"), indent=1)
print("arm      " + " ".join(f"{k[:14]:>14s}" for k in keys))
for a, v in out["arms"].items():
    print(f"{a:8s} " + " ".join(f"{(v[k]['mean'] if v[k]['mean'] is not None else float('nan')):8.4f}±{(v[k]['sd'] or 0):.3f}" for k in keys))
for a, v in out["vs_J"].items():
    print(a, "- J:", {k: round(x["diff"], 4) for k, x in v.items()})
