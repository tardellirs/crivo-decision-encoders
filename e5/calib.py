"""E5 calibration on the Decision Index 0.3 public suite, from the per-question arrays of extract.py.

Everything is a fixed, post-hoc computation on the released runs' probabilities: no recalibration, no temperature,
nothing fitted. Labelled [ours-public] in the README (our computation on public items, not a board number).

Item set ("MAIN"): every scored field of the 35 index benchmarks with a per-field right/wrong answer (the 37 index
benchmarks minus ToolRet and BRIGHT, which are nDCG over candidates). Two weightings:
  bench  - each benchmark weighs 1/35, fields equally inside it (the board's calibration convention); headline
  pooled - every field weighs the same (dominated by ACOS, SATA-Bench, BFCL, HellaSwag ...)
Metrics (top-label): accuracy, mean confidence, ECE with 15 equal-width bins, adaptive ECE with 15 equal-mass bins,
multi-class Brier (sum over options; accepted set merged), NLL, over-confidence (conf - acc), over95 = share of fields
answered wrong with confidence >= 0.95, AURC (area under the selective risk-coverage curve).
CIs: stratified cluster bootstrap - requests resampled with replacement inside each benchmark (fields of a request
move together), B = 1000, seed 0; the same draws for every model, so model differences are paired.

    python calib.py --tab ~/e5/tab --board ../../../release/di-index-20261020.json \
        --board032 ../../../release/di-index-v0.3.2-20261020.json --out results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["c17m", "c32v1", "c32v2", "c68m", "c150v1", "c150v2", "c400m", "c1b", "prumo"]
LABEL = {"c17m": "Crivo-17M", "c32v1": "Crivo-32M v1", "c32v2": "Crivo-32M v2", "c68m": "Crivo-68M",
         "c150v1": "Crivo-150M v1", "c150v2": "Crivo-150M v2", "c400m": "Crivo-400M", "c1b": "Crivo-1B",
         "prumo": "Prumo"}
BOARD_ENGINE = {"c17m": ("04", "crivo-17m"), "c32v1": ("032", "crivo-32m"), "c32v2": ("04", "crivo-32m-v2"),
                "c68m": ("04", "crivo-68m"), "c150v1": ("032", "crivo-150m"), "c150v2": ("04", "crivo-150m-v2"),
                "c400m": ("04", "crivo-400m"), "c1b": ("04", "crivo-1b"), "prumo": ("04", "prumo")}
AREAS = ["knowledge", "language", "retrieval", "tools", "arts"]


def load(tab, models=MODELS):
    q = pd.read_parquet(Path(tab) / "questions.parquet")
    preds = {m: dict(np.load(Path(tab) / f"preds_{m}.npz")) for m in models}
    return q, preds


def bench_weights(bench, mask):
    """Weights summing to 1 over mask: each benchmark present equally, fields equally inside it."""
    w = np.zeros(len(bench))
    b = bench[mask]
    u, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    w[mask] = 1.0 / (len(u) * cnt[inv])
    return w


def ece_equal_width(conf, correct, w, bins=15):
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    num = np.bincount(idx, w * (correct - conf), bins)
    return float(np.abs(num).sum() / w.sum())


def ece_equal_mass(conf, correct, w, order, bins=15):
    """order = argsort of conf (stable); bins hold equal total weight."""
    ws = w[order]
    cw = np.cumsum(ws) - ws / 2
    idx = np.minimum((cw / ws.sum() * bins).astype(int), bins - 1)
    num = np.bincount(idx, ws * (correct[order] - conf[order]), bins)
    return float(np.abs(num).sum() / w.sum())


def aurc(conf, correct, w):
    """Selective risk integrated over coverage; tied confidences enter together."""
    m = w > 0
    c, y, ww = conf[m], correct[m], w[m]
    u, inv = np.unique(-c, return_inverse=True)  # groups from most to least confident
    gw = np.bincount(inv, ww)
    ge = np.bincount(inv, ww * (1 - y))
    cov = np.cumsum(gw)
    risk = np.cumsum(ge) / cov
    return float((gw * risk).sum() / gw.sum())


def metrics(p, w, order=None, bins=15):
    w = w / w.sum()
    conf, y = p["conf"], p["correct"].astype(float)
    if order is None:
        order = np.argsort(conf, kind="stable")
    acc = float((w * y).sum()); mc = float((w * conf).sum())
    hi = conf >= 0.95
    return dict(acc=acc, conf=mc, overconf=mc - acc, ece15=ece_equal_width(conf, y, w, bins),
                ece10=ece_equal_width(conf, y, w, 10), aece15=ece_equal_mass(conf, y, w, order, bins),
                brier=float((w * p["brier"]).sum()), nll=float((w * p["nll"]).sum()),
                over95=float((w * (hi & (y == 0))).sum()), share95=float((w * hi).sum()),
                acc_at95=float((w * hi * y).sum() / max((w * hi).sum(), 1e-12)), aurc=aurc(conf, y, w),
                n=int((w > 0).sum()))


def reliability(p, w, bins=15):
    w = w / w.sum()
    conf, y = p["conf"], p["correct"].astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    W = np.bincount(idx, w, bins)
    C = np.bincount(idx, w * conf, bins)
    A = np.bincount(idx, w * y, bins)
    return [dict(lo=i / bins, w=float(W[i]), conf=float(C[i] / W[i]) if W[i] else None,
                 acc=float(A[i] / W[i]) if W[i] else None) for i in range(bins)]


def bootstrap(q, preds, mask, scheme, B=1000, seed=0, models=MODELS):
    """Stratified cluster bootstrap over requests within benchmarks. Returns {model: {metric: array(B)}}."""
    rng = np.random.default_rng(seed)
    bench = q.bench.to_numpy(); sid = q.sid.to_numpy()
    idx_all = np.flatnonzero(mask)
    groups = []
    for b in np.unique(bench[idx_all]):
        items = idx_all[bench[idx_all] == b]
        u, local = np.unique(sid[items], return_inverse=True)
        groups.append((items, local, len(u)))
    orders = {m: np.argsort(preds[m]["conf"], kind="stable") for m in models}
    out = {m: {k: np.zeros(B) for k in ("acc", "ece15", "aece15", "brier", "nll")} for m in models}
    nb = len(groups)
    for r in range(B):
        w = np.zeros(len(q))
        for items, local, R in groups:
            cnt = np.bincount(rng.integers(0, R, R), minlength=R)
            mult = cnt[local].astype(float)
            if scheme == "bench":
                w[items] = mult / mult.sum() / nb
            else:
                w[items] = mult
        w /= w.sum()
        for m in models:
            p = preds[m]; y = p["correct"].astype(float)
            out[m]["acc"][r] = (w * y).sum()
            out[m]["ece15"][r] = ece_equal_width(p["conf"], y, w)
            out[m]["aece15"][r] = ece_equal_mass(p["conf"], y, w, orders[m])
            out[m]["brier"][r] = (w * p["brier"]).sum()
            out[m]["nll"][r] = (w * p["nll"]).sum()
    return out


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def board_rows(board04, board032):
    d04 = json.load(open(board04)); d032 = json.load(open(board032))
    by = {"04": {m["engine"]: m for m in d04["models"]}, "032": {m["engine"]: m for m in d032["models"]}}
    ours = {}
    for k, (ed, eng) in BOARD_ENGINE.items():
        m = by[ed][eng]
        ours[k] = dict(edition="0.4" if ed == "04" else "0.3.2", engine=eng, **m["calibration"])
    # neighbours: every 0.4 entry with served_params <= 1.1B that has calibration, plus Jev 1.13 and Zev
    neigh = []
    for m in d04["models"]:
        c = m.get("calibration"); p = (m.get("meta") or {}).get("served_params")
        if not c or m["engine"].startswith("crivo") or m["engine"] == "prumo":
            continue
        if (p and p <= 1.1e9) or m["engine"] in ("jev-1.13.0", "zev-latest"):
            neigh.append(dict(engine=m["engine"], name=m.get("name"), params=p, pub=(m.get("v03") or {}).get("pub"),
                              full=(m.get("v03") or {}).get("v"),
                              **{k: c[k] for k in ("acc", "conf", "ece", "brier", "over95", "n", "benchmarks")}))
    return ours, sorted(neigh, key=lambda r: (r["params"] or 0)), d04["suite"]["calibration_note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--board032", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    q, preds = load(a.tab)
    bench = q.bench.to_numpy()
    main_mask = q.in_index.to_numpy()
    res = dict(models=MODELS, labels=LABEL, n_fields=int(main_mask.sum()), n_benchmarks=int(len(np.unique(bench[main_mask]))),
               n_requests=int(q[main_mask].sid.nunique()), overall={}, by_qtype={}, by_area={}, by_bench={},
               reliability={}, prumo_claimed={}, ci={}, paired={}, board_replica={})
    W = {"bench": bench_weights(bench, main_mask), "pooled": main_mask.astype(float)}
    for m in MODELS:
        p = preds[m]
        res["overall"][m] = {s: metrics(p, W[s]) for s in W}
        res["reliability"][m] = {s: reliability(p, W[s]) for s in W}
        res["by_qtype"][m] = {}
        qt, kk = q.qtype.to_numpy(), q.k.to_numpy()
        tmask = {"choice": qt == "choice", "noul": qt == "noul", "choice k=2": (qt == "choice") & (kk == 2),
                 "choice k>2": (qt == "choice") & (kk > 2)}
        for t, tm in tmask.items():
            mk = main_mask & tm
            res["by_qtype"][m][t] = dict(bench=metrics(p, bench_weights(bench, mk)), pooled=metrics(p, mk.astype(float)),
                                         n_benchmarks=int(len(np.unique(bench[mk]))))
        res["by_area"][m] = {}
        for ar in AREAS:
            mk = main_mask & (q.area.to_numpy() == ar)
            res["by_area"][m][ar] = metrics(p, bench_weights(bench, mk))
        res["by_bench"][m] = {}
        for b in np.unique(bench[main_mask]):
            mk = main_mask & (bench == b)
            r = metrics(p, mk.astype(float))
            res["by_bench"][m][str(int(b))] = {k: r[k] for k in ("acc", "conf", "ece15", "brier", "n")}
    res["bench_names"] = {str(int(b)): n for b, n in q[main_mask].groupby("bench").bench_name.first().items()}
    # Prumo: claimed (non-uniform) vs unclaimed fields
    pr = preds["prumo"]
    claimed = main_mask & ~pr["uniform"]
    res["prumo_claimed"] = dict(share_pooled=float(claimed.sum() / main_mask.sum()),
                                share_bench=float((W["bench"] * claimed).sum()),
                                claimed_pooled=metrics(pr, claimed.astype(float)),
                                claimed_bench=metrics(pr, bench_weights(bench, claimed)),
                                unclaimed_pooled=metrics(pr, (main_mask & pr["uniform"]).astype(float)))
    # board-protocol replica on public rows: 10 equal-width bins, benchmark-equal (all fields; no 1-in-6 sample)
    nii = (~q.in_index.to_numpy())
    variants = {"index35": main_mask, "index35+notindex": main_mask | nii}
    for name, mk in variants.items():
        wv = bench_weights(bench, mk)
        res["board_replica"][name] = {m: {k: metrics(preds[m], wv)[k] for k in ("acc", "conf", "ece10", "brier", "over95")}
                                      for m in MODELS}
        res["board_replica"][name]["_n_benchmarks"] = int(len(np.unique(bench[mk])))
    # bootstrap
    for s in ("bench", "pooled"):
        bs = bootstrap(q, preds, main_mask, s, B=a.B)
        res["ci"][s] = {m: {k: ci(v) for k, v in bs[m].items()} for m in MODELS}
        pairs = [("c32v1", "c32v2"), ("c150v1", "c150v2"), ("c17m", "c32v2"), ("c32v2", "c68m"), ("c68m", "c150v2"),
                 ("c150v2", "c400m"), ("c400m", "c1b"), ("c17m", "prumo")]
        res["paired"][s] = {f"{x}-{y}": {k: dict(diff=float(res["overall"][x][s][k] - res["overall"][y][s][k]),
                                                   ci=ci(bs[x][k] - bs[y][k])) for k in ("ece15", "aece15", "brier", "acc")}
                            for x, y in pairs}
    ours, neigh, note = board_rows(a.board, a.board032)
    res["board"] = dict(note=note, crivo_prumo=ours, neighbours=neigh)
    res["B"] = a.B
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
