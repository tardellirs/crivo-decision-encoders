"""GPU-4 comparisons: the initialisation 2x2 of the reader ablation (docs/EXPERIMENT_LOG.md §18.y, pre-registered before the run).

                     joint, compute-matched        per option
    reranker init    Jcomp   (GPU-3, s0-s2)        P     (GPU-2, s0-s2)
    raw Ettin init   JrawComp (GPU-4, s0-s2)       Praw  (GPU-3 s0 + GPU-4 s1, s2)

Cells: mean ± sd over seeds. Contrasts (each a Welch 95% CI over seeds, Welch-Satterthwaite df):
    C1  per-option - joint at raw init       Praw - JrawComp                     (PRIMARY for §18.y)
    C2  reranker - raw, per-option reader    P - Praw
    C3  reranker - raw, joint reader         Jcomp - JrawComp
    C4  interaction                          (P - Praw) - (Jcomp - JrawComp) = (P - Jcomp) - (Praw - JrawComp)
    ref per-option - joint at reranker init  P - Jcomp                           (GPU-3's primary, repeated)
Metrics read: pub15 skill, S1-Arena s1_60 accuracy, BTZSC500 macro-F1 (all 22). CI above 0 / contains 0 / below 0.
Secondary (descriptive): the same contrasts on calibration and unseen-subset metrics; JrawComp - Jraw (compute at raw
init, Jraw = GPU-2's 1-epoch joint raw arm); training tokens. No E4 here: GPU-4 runs none (per-option flips are 0 by
construction), so flips are left out of the 2x2.

    python compare.py --gpu2 results_gpu2 --gpu3 results_gpu3 --gpu4 results_gpu4 --out results_gpu4/compare.json
Metric extraction = gpu3/compare.py's metrics() with E4 optional.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
BT_UNSEEN = ["appreviews", "empathetic", "trueteacher", "capsotu"]
S1_CLASS = json.load(open(HERE.parent / "s1arena" / "contamination.json"))
PRIMARY = ["pub15_skill", "s1_acc", "btzsc_all_f1"]
ALL = PRIMARY + ["pub15_ece", "pub15_brier", "s1_brier", "s1_unseen_acc", "btzsc_unseen_f1"]
CELLS = {("reranker", "joint"): "Jcomp", ("reranker", "peropt"): "P", ("raw", "joint"): "JrawComp", ("raw", "peropt"): "Praw"}
CONTRASTS = [  # name, label, {arm: coefficient}
    ("C1_Praw_minus_JrawComp", "per-option - joint @ raw init       (Praw - JrawComp)", {"Praw": 1, "JrawComp": -1}),
    ("C2_P_minus_Praw", "reranker - raw @ per-option         (P - Praw)", {"P": 1, "Praw": -1}),
    ("C3_Jcomp_minus_JrawComp", "reranker - raw @ joint              (Jcomp - JrawComp)", {"Jcomp": 1, "JrawComp": -1}),
    ("C4_interaction", "interaction (P-Praw)-(Jcomp-JrawComp)", {"P": 1, "Praw": -1, "Jcomp": -1, "JrawComp": 1}),
    ("ref_P_minus_Jcomp", "per-option - joint @ reranker init (P - Jcomp) [GPU-3]", {"P": 1, "Jcomp": -1}),
]


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
    return m


def welch_lin(groups: dict, coef: dict):
    """Linear contrast sum_i c_i * mean_i of independent groups; Welch-Satterthwaite df, t 95% CI.
    For two groups this equals scipy's ttest_ind(equal_var=False)."""
    xs = {a: [v for v in groups.get(a, []) if v is not None] for a in coef}
    if any(len(x) < 2 for x in xs.values()):
        return None
    diff = sum(c * np.mean(xs[a]) for a, c in coef.items())
    w = {a: c * c * np.var(xs[a], ddof=1) / len(xs[a]) for a, c in coef.items()}
    se2 = sum(w.values())
    if se2 <= 0:
        return {"diff": float(diff), "welch_ci95": [float(diff), float(diff)], "se": 0.0, "df": None, "p": None,
                "verdict": "above 0" if diff > 0 else ("below 0" if diff < 0 else "contains 0"), "n": {a: len(x) for a, x in xs.items()}}
    df = se2 ** 2 / sum(w[a] ** 2 / (len(xs[a]) - 1) for a in coef if w[a] > 0)
    h = stats.t.ppf(0.975, df) * np.sqrt(se2)
    lo, hi = diff - h, diff + h
    return {"diff": float(diff), "welch_ci95": [float(lo), float(hi)], "se": float(np.sqrt(se2)), "df": float(df),
            "p": float(2 * stats.t.sf(abs(diff) / np.sqrt(se2), df)),
            "verdict": "above 0" if lo > 0 else ("below 0" if hi < 0 else "contains 0"), "n": {a: len(x) for a, x in xs.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu2", required=True); ap.add_argument("--gpu3", required=True); ap.add_argument("--gpu4", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g2, g3, g4 = Path(a.gpu2), Path(a.gpu3), Path(a.gpu4)
    src = {"P": [(g2, 0), (g2, 1), (g2, 2)], "Jraw": [(g2, 0), (g2, 1), (g2, 2)], "Jcomp": [(g3, 0), (g3, 1), (g3, 2)],
           "Praw": [(g3, 0), (g4, 1), (g4, 2)], "JrawComp": [(g4, 0), (g4, 1), (g4, 2)]}
    runs = {arm: [r / "ev" / f"{arm}_s{s}" for r, s in v if (r / "ev" / f"{arm}_s{s}" / "DONE").exists()] for arm, v in src.items()}
    M = {arm: [metrics(d) for d in ds] for arm, ds in runs.items() if ds}
    vals = {k: {arm: [m[k] for m in ms] for arm, ms in M.items()} for k in ALL}
    out = {"seeds": {arm: [str(d.relative_to(d.parents[2])) for d in ds] for arm, ds in runs.items()}, "arms": {},
           "cells": {f"{i}/{r}": arm for (i, r), arm in CELLS.items()}, "contrasts": {}, "secondary": {}}
    for arm, ms in M.items():
        out["arms"][arm] = {}
        for k in ALL:
            x = [m[k] for m in ms if m[k] is not None]
            out["arms"][arm][k] = {"mean": float(np.mean(x)) if x else None, "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
                                   "n": len(x), "values": [m[k] for m in ms]}
    for name, _, coef in CONTRASTS:
        out["contrasts"][name] = {k: welch_lin(vals[k], coef) for k in ALL}
    out["secondary"]["JrawComp_minus_Jraw"] = {k: welch_lin(vals[k], {"JrawComp": 1, "Jraw": -1}) for k in ALL}
    tok = {}
    for arm, v in src.items():
        if arm in ("P", "Jraw"):
            continue
        for r, s in v:
            c = r / "ck" / f"{arm}_s{s}" / "config.json"
            if c.exists():
                x = json.load(open(c))["decision"]
                tok[f"{arm}_s{s}"] = {k: x.get(k) for k in ("step", "total_planned", "oom_skips", "tokens_total", "questions_total", "train_seconds")}
    out["training_runs"] = tok
    Path(a.out).write_text(json.dumps(out, indent=1))

    f = lambda arm, k: (lambda c: "—" if c is None or c["mean"] is None else
                        f"{c['mean']:.3f} ± {c['sd']:.3f} (n={c['n']})" if c["sd"] is not None else f"{c['mean']:.3f} (n={c['n']})")(out["arms"].get(arm, {}).get(k))
    print("2x2 (mean ± sd over seeds)        joint, compute-matched            per option")
    for k in PRIMARY:
        print(f" {k}")
        for init in ("reranker", "raw"):
            j, p = CELLS[(init, "joint")], CELLS[(init, "peropt")]
            print(f"   {init:8s} init   {j:8s} {f(j, k):26s}  {p:5s} {f(p, k)}")
    print("contrasts (Welch 95% CI over seeds; pre-registered reading: CI above 0 / contains 0 / below 0)")
    for name, label, _ in CONTRASTS:
        print(f" {label}")
        for k in PRIMARY:
            v = out["contrasts"][name][k]
            print(f"   {k:14s} " + ("n/a (fewer than 2 seeds in a cell)" if not v else
                                  f"{v['diff']:+.4f} [{v['welch_ci95'][0]:+.4f}, {v['welch_ci95'][1]:+.4f}]  {v['verdict']}"))
    print(" secondary: JrawComp - Jraw (compute at raw init): " + "; ".join(
        f"{k} {v['diff']:+.4f} [{v['welch_ci95'][0]:+.4f}, {v['welch_ci95'][1]:+.4f}]" for k in PRIMARY
        if (v := out["secondary"]["JrawComp_minus_Jraw"][k])))
    for k, v in out["training_runs"].items():
        print("  run", k, v)


if __name__ == "__main__":
    main()
