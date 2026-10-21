"""E7 statistics on the Decision Index board snapshot (CPU, JSON only).

    python board_stats.py <di-index-20261020.json> <di-index-v0.3.2-20261020.json> <out.json>

1. Public index (v03.pub) vs private same-skills (v03.same_skills) over all models: Pearson r and Spearman rho with
   percentile bootstrap CIs (models resampled, B = 10,000, seed 0). Also pub vs new_domains and pub vs Full.
2. The same inside size buckets of meta.served_params: <100M, 100-500M, 0.5-2B, 2-10B, >10B.
3. Partial correlation of pub and same_skills controlling for log10(served_params) (models with a size): Pearson on
   residuals of a linear fit on log-params, and the rank version (Spearman of residuals of ranks on rank of size).
4. Crivo v1 vs v2 (150M and 32M), per benchmark with a private test: chance-corrected private skill
   clip((private_eq - random)/(1 - random), 0, 1); mean paired difference with bootstrap CI over benchmarks, paired
   t-test, Wilcoxon signed-rank and sign test. Per-item data is not public, so the unit is the benchmark.
"""
import json
import math
import sys

import numpy as np
from scipy import stats

B = 10_000
rng = np.random.default_rng(0)


def corr_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r = stats.pearsonr(x, y)[0] if n > 2 else float("nan")
    rho = stats.spearmanr(x, y)[0] if n > 2 else float("nan")
    rs, rhos = [], []
    for _ in range(B if n > 3 else 0):
        i = rng.integers(0, n, n)
        if len(set(i)) < 3 or np.std(x[i]) == 0 or np.std(y[i]) == 0:
            continue
        rs.append(np.corrcoef(x[i], y[i])[0, 1]); rhos.append(stats.spearmanr(x[i], y[i])[0])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]
    return {"n": n, "pearson": float(r), "pearson_ci95": ci(rs), "spearman": float(rho), "spearman_ci95": ci(rhos)}


def resid(y, x):
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def partial(x, y, z):
    x, y, z = map(lambda v: np.asarray(v, float), (x, y, z))
    def one(ix):
        return np.corrcoef(resid(x[ix], z[ix]), resid(y[ix], z[ix]))[0, 1]
    def one_rank(ix):
        rx, ry, rz = (stats.rankdata(v[ix]) for v in (x, y, z))
        return np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1]
    n = len(x); full = np.arange(n)
    bp, br = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bp.append(one(i)); br.append(one_rank(i))
    pc = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"n": n, "partial_pearson": float(one(full)), "partial_pearson_ci95": pc(bp),
            "partial_spearman": float(one_rank(full)), "partial_spearman_ci95": pc(br),
            "r_pub_logparams": float(np.corrcoef(x, z)[0, 1]), "r_ss_logparams": float(np.corrcoef(y, z)[0, 1])}


def skill(b):
    pe, rnd = b.get("private_eq"), b.get("random")
    if pe is None or rnd is None:
        return None
    return min(1.0, max(0.0, (pe - rnd) / (1 - rnd))) if rnd < 1 else None


def paired(a_entry, b_entry, names):
    rows = []
    for k, ba in a_entry["benchmarks"].items():
        bb = b_entry["benchmarks"].get(k)
        if not bb:
            continue
        sa, sb = skill(ba), skill(bb)
        if sa is None or sb is None:
            continue
        rows.append({"id": k, "name": names.get(k, k), "v1": sa, "v2": sb, "d": sb - sa,
                     "v1_raw": ba["private_eq"], "v2_raw": bb["private_eq"]})
    d = np.array([r["d"] for r in rows]); raw = np.array([r["v2_raw"] - r["v1_raw"] for r in rows])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    nz = d[d != 0]
    return {"n_benchmarks": len(rows), "mean_v1": float(np.mean([r["v1"] for r in rows])),
            "mean_v2": float(np.mean([r["v2"] for r in rows])), "mean_diff": float(d.mean()),
            "mean_diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "down": int((d < 0).sum()), "up": int((d > 0).sum()), "same": int((d == 0).sum()),
            "paired_t_p": float(stats.ttest_rel([r["v2"] for r in rows], [r["v1"] for r in rows]).pvalue),
            "wilcoxon_p": float(stats.wilcoxon(nz).pvalue) if len(nz) else None,
            "sign_test_p": float(stats.binomtest(int((nz > 0).sum()), len(nz)).pvalue) if len(nz) else None,
            "raw_mean_diff": float(raw.mean()),
            "raw_wilcoxon_p": float(stats.wilcoxon(raw[raw != 0]).pvalue) if (raw != 0).any() else None,
            "rows": sorted(rows, key=lambda r: r["d"])}


def main():
    d04, d032, out = sys.argv[1:4]
    a = json.load(open(d04)); b = json.load(open(d032))
    M = [m for m in a["models"] if m["v03"].get("pub") is not None and m["v03"].get("same_skills") is not None]
    res = {"snapshot": d04, "n_models": len(M)}
    pub = [m["v03"]["pub"] for m in M]; ss = [m["v03"]["same_skills"] for m in M]
    nd = [m["v03"]["new_domains"] for m in M]; full = [m["v03"]["v"] for m in M]
    res["all"] = {"pub_vs_same_skills": corr_ci(pub, ss), "pub_vs_new_domains": corr_ci(pub, nd),
                  "pub_vs_full": corr_ci(pub, full), "same_skills_vs_new_domains": corr_ci(ss, nd)}
    buckets = [("<100M", 0, 1e8), ("100-500M", 1e8, 5e8), ("0.5-2B", 5e8, 2e9), ("2-10B", 2e9, 1e10), (">10B", 1e10, 1e15)]
    res["buckets"] = {}
    sized = [m for m in M if m["meta"].get("served_params")]
    for name, lo, hi in buckets:
        mm = [m for m in sized if lo <= m["meta"]["served_params"] < hi]
        res["buckets"][name] = {"pub_vs_same_skills": corr_ci([m["v03"]["pub"] for m in mm], [m["v03"]["same_skills"] for m in mm]),
                                "pub_vs_full": corr_ci([m["v03"]["pub"] for m in mm], [m["v03"]["v"] for m in mm]),
                                "models": [m["engine"] for m in mm] if len(mm) <= 12 else len(mm)}
    lp = [math.log10(m["meta"]["served_params"]) for m in sized]
    res["partial_logparams"] = {"pub_vs_same_skills": partial([m["v03"]["pub"] for m in sized], [m["v03"]["same_skills"] for m in sized], lp),
                                "pub_vs_full": partial([m["v03"]["pub"] for m in sized], [m["v03"]["v"] for m in sized], lp),
                                "n_unsized_excluded": len(M) - len(sized)}
    names = {str(x["id"]): x["name"] for x in a["benchmarks"]} if isinstance(a["benchmarks"], list) else {}
    e04 = {m["engine"]: m for m in a["models"]}; e032 = {m["engine"]: m for m in b["models"]}
    res["crivo_v1_vs_v2"] = {"150M": paired(e032["crivo-150m"], e04["crivo-150m-v2"], names),
                             "32M": paired(e032["crivo-32m"], e04["crivo-32m-v2"], names)}
    json.dump(res, open(out, "w"), indent=1)
    def fmt(c):
        return (f"n={c['n']:3d} r={c['pearson']:.3f} [{c['pearson_ci95'][0]:.3f},{c['pearson_ci95'][1]:.3f}] "
                f"rho={c['spearman']:.3f} [{c['spearman_ci95'][0]:.3f},{c['spearman_ci95'][1]:.3f}]") if c["n"] > 3 else f"n={c['n']}"
    for k, v in res["all"].items():
        print("ALL", k, fmt(v))
    for k, v in res["buckets"].items():
        print("BUCKET", k, "pub~ss", fmt(v["pub_vs_same_skills"]), "| pub~full", fmt(v["pub_vs_full"]))
    for k, v in res["partial_logparams"].items():
        if isinstance(v, dict):
            print("PARTIAL", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    for k, v in res["crivo_v1_vs_v2"].items():
        print("CRIVO", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "rows"})


if __name__ == "__main__":
    main()
