"""GPU-3 comparisons (pre-registered before the run: docs/EXPERIMENT_LOG.md §18.1).

PRIMARY (pre-registered): P (GPU-2, per option, seeds 0-2) minus Jcomp (GPU-3, joint, P's optimizer-step count,
seeds 0-2), Welch 95% CI over seeds (3 + 3), on: pub15 skill, S1-Arena s1_60 accuracy, BTZSC500 macro-F1 (all 22),
E4 flip rate (permutations; and yes/no swap). Reading of each metric: CI above 0 = the per-option advantage
survives the compute-matched control; CI containing 0 = not distinguishable; CI below 0 = reversed.
SECONDARY (descriptive, not decision rules): Jcomp - J and Jcomp - Jsteps (Welch); Praw s0 - P s0 and Praw s0 vs
P's 3-seed mean ± sd (one seed: no CI over seeds); Praw s0 - Jraw mean; pub15 question accuracy P - Jcomp paired
over questions (seed-averaged, bootstrap B = 2,000); encoded training tokens of Jcomp / Praw.

    python compare.py --gpu2 results_gpu2 --gpu3 results_gpu3 --evaldir gpu3/inputs/evalsets --out compare.json
Metric extraction = a1a2/summarize.py's metrics() (copied: summarize.py runs on import).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
BT_UNSEEN = ["appreviews", "empathetic", "trueteacher", "capsotu"]
S1_CLASS = json.load(open(HERE.parent / "s1arena" / "contamination.json"))
PRIMARY = ["pub15_skill", "s1_acc", "btzsc_all_f1", "flip_perm", "flip_yn"]
ALL = PRIMARY + ["pub15_ece", "pub15_brier", "s1_brier", "s1_unseen_acc", "btzsc_unseen_f1", "flip_rev"]


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


def correct(ev: Path, evaldir: Path):
    res = {}
    p = ev / "pub15.jsonl"
    if not p.exists():
        return None
    for l in open(p):
        r = json.loads(l)
        if r.get("status") == "ok":
            res[r["id"]] = r["response"]["answers"]
    out = {}
    for l in open(evaldir / "pub15.jsonl"):
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


def welch(x, y):
    x = [v for v in x if v is not None]; y = [v for v in y if v is not None]
    if len(x) < 2 or len(y) < 2:
        return None
    t = stats.ttest_ind(x, y, equal_var=False); ci = t.confidence_interval()
    return {"diff": float(np.mean(x) - np.mean(y)), "welch_ci95": [float(ci[0]), float(ci[1])], "p": float(t.pvalue),
            "verdict": "above 0" if ci[0] > 0 else ("below 0" if ci[1] < 0 else "contains 0")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu2", required=True); ap.add_argument("--gpu3", required=True)
    ap.add_argument("--evaldir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g2, g3, evd = Path(a.gpu2), Path(a.gpu3), Path(a.evaldir)
    runs = {}
    for arm, root in [("J", g2), ("Jsteps", g2), ("Jraw", g2), ("P", g2), ("Jcomp", g3), ("Praw", g3)]:
        d = [root / "ev" / f"{arm}_s{s}" for s in range(3) if (root / "ev" / f"{arm}_s{s}" / "DONE").exists()]
        if d:
            runs[arm] = d
    M = {arm: [metrics(d) for d in ds] for arm, ds in runs.items()}
    out = {"seeds": {arm: [d.name for d in ds] for arm, ds in runs.items()}, "arms": {}, "primary_P_minus_Jcomp": {}, "secondary": {}}
    for arm, ms in M.items():
        out["arms"][arm] = {k: {"mean": float(np.mean([m[k] for m in ms if m[k] is not None])) if any(m[k] is not None for m in ms) else None,
                                "sd": float(np.std([m[k] for m in ms if m[k] is not None], ddof=1)) if sum(m[k] is not None for m in ms) > 1 else None,
                                "values": [m[k] for m in ms]} for k in ALL}
    if "P" in M and "Jcomp" in M:
        out["primary_P_minus_Jcomp"] = {k: welch([m[k] for m in M["P"]], [m[k] for m in M["Jcomp"]]) for k in PRIMARY}
        cp = [correct(d, evd) for d in runs["P"]]; cj = [correct(d, evd) for d in runs["Jcomp"]]
        if all(c is not None for c in cp + cj):
            keys = sorted(set.intersection(*[set(c) for c in cp + cj]))
            diff = np.array([np.mean([c[k] for c in cp]) - np.mean([c[k] for c in cj]) for k in keys])
            rng = np.random.default_rng(0)
            boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            out["secondary"]["pub15_question_acc_P_minus_Jcomp_paired"] = {"diff": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "questions": len(keys)}
    for x, y in [("Jcomp", "J"), ("Jcomp", "Jsteps")]:
        if x in M and y in M:
            out["secondary"][f"{x}_minus_{y}"] = {k: welch([m[k] for m in M[x]], [m[k] for m in M[y]]) for k in ALL}
    if "Praw" in M:
        pr = M["Praw"][0]
        for ref in ("P", "Jraw"):
            if ref in M:
                out["secondary"][f"Praw_s0_vs_{ref}"] = {k: {"praw": pr[k], f"{ref}_mean": out["arms"][ref][k]["mean"], f"{ref}_sd": out["arms"][ref][k]["sd"],
                                                            "diff_vs_mean": None if pr[k] is None or out["arms"][ref][k]["mean"] is None else pr[k] - out["arms"][ref][k]["mean"]} for k in ALL}
        if "P" in M:
            out["secondary"]["Praw_s0_minus_P_s0"] = {k: None if pr[k] is None or M["P"][0][k] is None else pr[k] - M["P"][0][k] for k in ALL}
    tok = {}
    for arm in ("Jcomp", "Praw"):
        for d in runs.get(arm, []):
            c = g3 / "ck" / d.name / "config.json"
            if c.exists():
                x = json.load(open(c))["decision"]
                tok[d.name] = {k: x.get(k) for k in ("step", "total_planned", "oom_skips", "tokens_total", "questions_total", "train_seconds")}
    out["training_runs"] = tok
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("PRIMARY (pre-registered): P - Jcomp, Welch 95% CI over seeds")
    for k, v in out["primary_P_minus_Jcomp"].items():
        if v:
            print(f"  {k:14s} {v['diff']:+.4f} [{v['welch_ci95'][0]:+.4f}, {v['welch_ci95'][1]:+.4f}]  {v['verdict']}")
    print("arm means: " + "; ".join(f"{arm} " + " ".join(f"{k}={out['arms'][arm][k]['mean']:.3f}" for k in PRIMARY if out['arms'][arm][k]['mean'] is not None) for arm in out["arms"]))
    for k, v in out["training_runs"].items():
        print("  run", k, v)


if __name__ == "__main__":
    main()
