"""Public-suite error analysis (Decision Index 0.3) from the per-question arrays of e5/extract.py. CPU, descriptive.

Item set = MAIN of e5/calib.py (scored fields of the 35 per-field index benchmarks). "norm" = chance-normalised
accuracy, mean over fields of (correct - c)/(1 - c) with c = |accepted|/k (0 = chance, 1 = perfect); "bench" = every
benchmark weighs the same. Official per-benchmark skill comes from each run's own index.json (kit scorer).
No model output is combined with another: the Prumo section is a contingency table only, never a combined score.

    python errors.py --tab ~/e5/tab --runs ~/e5/runs --out results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "e5"))
from calib import LABEL, MODELS, bench_weights, load  # noqa: E402

CRIVOS = [m for m in MODELS if m != "prumo"]
V2 = ["c17m", "c32v2", "c68m", "c150v2", "c400m", "c1b"]
K_BUCKETS = [(2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 31, "10-31"), (32, 10**6, ">=32")]
L_BUCKETS = [(0, 128, "<128"), (128, 512, "128-511"), (512, 2048, "512-2047"), (2048, 8192, "2048-8191"),
             (8192, 10**9, ">=8192")]


def wmean(x, w):
    s = w.sum()
    return float((x * w).sum() / s) if s > 0 else None


def boot_within(q, mask, fn, B=500, seed=0):
    """Cluster bootstrap (requests within benchmark) of a statistic fn(weights) -> float."""
    rng = np.random.default_rng(seed)
    bench = q.bench.to_numpy(); sid = q.sid.to_numpy()
    idx_all = np.flatnonzero(mask)
    groups = []
    for b in np.unique(bench[idx_all]):
        items = idx_all[bench[idx_all] == b]
        u, local = np.unique(sid[items], return_inverse=True)
        groups.append((items, local, len(u)))
    vals = []
    for _ in range(B):
        w = np.zeros(len(q))
        for items, local, R in groups:
            w[items] = np.bincount(rng.integers(0, R, R), minlength=R)[local]
        vals.append(fn(w))
    vals = np.array([v for v in vals if v is not None])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=500)
    a = ap.parse_args()
    q, P = load(a.tab)
    M = q.in_index.to_numpy()
    bench = q.bench.to_numpy(); k = q.k.to_numpy(); qtype = q.qtype.to_numpy()
    c = q.n_accepted.to_numpy() / k
    Y = {m: P[m]["correct"].astype(float) for m in MODELS}
    with np.errstate(divide="ignore", invalid="ignore"):
        N = {m: np.where(c < 1, (Y[m] - c) / (1 - c), 0.0) for m in MODELS}
    names = q[M].groupby("bench").bench_name.first().to_dict()
    areas = q[M].groupby("bench").area.first().to_dict()
    out = dict(n_fields=int(M.sum()), labels=LABEL)

    # A. per benchmark: official skill (kit) + field accuracy
    off = {m: json.load(open(Path(a.runs) / m / "index.json"))["benchmarks"] for m in MODELS}
    rows = []
    for b in sorted(np.unique(bench[M])):
        mk = M & (bench == b)
        r = dict(bench=int(b), name=names[b], area=areas[b], fields=int(mk.sum()), requests=int(q[mk].sid.nunique()),
                 chance_field=float(c[mk].mean()), k_median=float(np.median(k[mk])),
                 state_tokens_median=float(np.median(q.state_tokens.to_numpy()[mk])),
                 input_tokens_median=float(np.median(q.input_tokens.to_numpy()[mk])))
        for m in MODELS:
            r[f"skill_{m}"] = off[m][str(int(b))]["skill"]
            r[f"acc_{m}"] = float(Y[m][mk].mean())
        r["gap_1b_17m"] = r["skill_c1b"] - r["skill_c17m"]
        r["gap_150v2_17m"] = r["skill_c150v2"] - r["skill_c17m"]
        # field-level disagreement between the smallest and the largest
        r["only_1b_right"] = float(((Y["c1b"] == 1) & (Y["c17m"] == 0))[mk].mean())
        r["only_17m_right"] = float(((Y["c1b"] == 0) & (Y["c17m"] == 1))[mk].mean())
        rows.append(r)
    out["per_bench"] = rows
    # official benchmarks not in MAIN (ToolRet, BRIGHT: nDCG)
    out["ranking_official"] = {n: {m: off[m][n]["skill"] for m in MODELS} for n in ("2", "36")}

    # B. per question type (binary choice / multi-option choice / noul)
    types = {"noul": M & (qtype == "noul"), "choice k=2": M & (qtype == "choice") & (k == 2),
             "choice k>2": M & (qtype == "choice") & (k > 2)}
    out["by_type"] = {}
    for t, mk in types.items():
        wb = bench_weights(bench, mk)
        out["by_type"][t] = dict(fields=int(mk.sum()), benchmarks=int(len(np.unique(bench[mk]))),
                                 benchmarks_list=sorted(names[b] for b in np.unique(bench[mk])),
                                 **{m: dict(acc=wmean(Y[m], wb), norm=wmean(N[m], wb)) for m in MODELS})

    # C. label count
    out["by_k"] = {}
    for lo, hi, lab in K_BUCKETS:
        mk = M & (k >= lo) & (k <= hi)
        wb = bench_weights(bench, mk)
        out["by_k"][lab] = dict(fields=int(mk.sum()), benchmarks=int(len(np.unique(bench[mk]))),
                                chance=wmean(c, wb), **{m: dict(acc=wmean(Y[m], wb), norm=wmean(N[m], wb)) for m in MODELS})
    # within-benchmark: GSM8K, same problems with 4 and 10 options (two tracks)
    tr = q.track.to_numpy()
    g = {}
    for t in sorted(set(tr[M & (bench == 30)])):
        mk = M & (bench == 30) & (tr == t)
        g[t] = dict(fields=int(mk.sum()), k=sorted(set(int(x) for x in k[mk])),
                    **{m: dict(acc=float(Y[m][mk].mean()), norm=float(N[m][mk].mean())) for m in MODELS})
    out["gsm8k_tracks"] = g
    # within-benchmark k effect: benchmarks whose k varies; low-k half vs high-k half by the benchmark's median
    wk = []
    for b in np.unique(bench[M]):
        mk = M & (bench == b)
        kk = k[mk]
        if len(np.unique(kk)) < 2:
            continue
        med = np.median(kk)
        lo_m = mk & (k <= med); hi_m = mk & (k > med)
        if lo_m.sum() < 50 or hi_m.sum() < 50:
            continue
        wk.append(dict(bench=names[b], k_low=float(np.median(k[lo_m])), k_high=float(np.median(k[hi_m])),
                       n_low=int(lo_m.sum()), n_high=int(hi_m.sum()),
                       **{m: dict(norm_low=float(N[m][lo_m].mean()), norm_high=float(N[m][hi_m].mean()),
                                  acc_low=float(Y[m][lo_m].mean()), acc_high=float(Y[m][hi_m].mean())) for m in MODELS}))
    out["within_bench_k"] = wk

    # D. input length: whole Crivo encoder sequence of the question (input_tokens); state-only as a secondary view
    # (many benchmarks carry the item text in `instructions` and leave `state` empty)
    for col, key in (("input_tokens", "by_len"), ("state_tokens", "by_state_len")):
        x = q[col].to_numpy()
        out[key] = {}
        for lo, hi, lab in L_BUCKETS:
            mk = M & (x >= lo) & (x < hi)
            wb = bench_weights(bench, mk)
            out[key][lab] = dict(fields=int(mk.sum()), benchmarks=int(len(np.unique(bench[mk]))),
                                 benchmarks_list=sorted(names[b] for b in np.unique(bench[mk])) if mk.sum() else [],
                                 **{m: dict(acc=wmean(Y[m], wb), norm=wmean(N[m], wb)) for m in MODELS})
    st = q.input_tokens.to_numpy()
    # within-benchmark median split: norm(long half) - norm(short half), benchmark-equal mean, cluster-bootstrap CI
    elig, lo_mask, hi_mask = [], np.zeros(len(q), bool), np.zeros(len(q), bool)
    for b in np.unique(bench[M]):
        mk = M & (bench == b)
        s = st[mk]
        q25, med, q75 = np.percentile(s, [25, 50, 75])
        lo_b = mk & (st <= med); hi_b = mk & (st > med)
        if q75 < 1.5 * max(q25, 1) or lo_b.sum() < 50 or hi_b.sum() < 50:
            continue
        elig.append(int(b)); lo_mask |= lo_b; hi_mask |= hi_b

    def len_effect(m, w=None):
        w = np.ones(len(q)) if w is None else w
        d = []
        for b in elig:
            l = lo_mask & (bench == b); h = hi_mask & (bench == b)
            if w[l].sum() == 0 or w[h].sum() == 0:
                return None
            d.append(wmean(N[m][h], w[h]) - wmean(N[m][l], w[l]))
        return float(np.mean(d))

    both = lo_mask | hi_mask
    out["len_within"] = dict(benchmarks=[names[b] for b in elig], per_bench=[], effect={})
    for b in elig:
        l = lo_mask & (bench == b); h = hi_mask & (bench == b)
        out["len_within"]["per_bench"].append(dict(bench=names[b], tok_short=float(np.median(st[l])),
                                                   tok_long=float(np.median(st[h])),
                                                   **{m: float(N[m][h].mean() - N[m][l].mean()) for m in MODELS}))
    for m in MODELS:
        out["len_within"]["effect"][m] = dict(diff=len_effect(m), ci=boot_within(q, both, lambda w, m=m: len_effect(m, w), B=a.B))
    out["len_within"]["gap_1b_17m"] = dict(
        diff=len_effect("c1b") - len_effect("c17m"),
        ci=boot_within(q, both, lambda w: (lambda x, y: None if x is None or y is None else x - y)(len_effect("c1b", w), len_effect("c17m", w)), B=a.B))

    # E. automatic error-category proxies
    cat = {}
    # E1 label-prior collapse on fixed-label benchmarks: share of fields on the modal predicted option vs its gold share
    pred = {m: P[m]["pred_idx"] for m in MODELS}
    gold = q.gold_idx.to_numpy()
    coll = {m: [] for m in MODELS}
    for b in np.unique(bench[M]):
        mk = M & (bench == b) & (k == 2)
        if mk.sum() < 100:
            continue
        for m in MODELS:
            pm = pred[m][mk]
            vals, cnt = np.unique(pm, return_counts=True)
            top = vals[cnt.argmax()]
            coll[m].append(dict(bench=names[b], pred_share=float(cnt.max() / mk.sum()), gold_share=float((gold[mk] == top).mean())))
    cat["binary_collapse"] = {m: dict(mean_excess=float(np.mean([x["pred_share"] - x["gold_share"] for x in coll[m]])),
                                      n_bench=len(coll[m]),
                                      over90=[x["bench"] for x in coll[m] if x["pred_share"] >= 0.9 and x["gold_share"] < 0.8])
                              for m in MODELS}
    # E2 position bias on k>=3 choice: P(pred = first option) vs P(gold = first option); same for last
    mk = M & (qtype == "choice") & (k >= 3)
    wb = bench_weights(bench, mk)
    cat["position"] = dict(fields=int(mk.sum()), gold_first=wmean((gold == 0).astype(float), wb),
                           gold_last=wmean((gold == k - 1).astype(float), wb), uniform_first=wmean(1 / k, wb),
                           **{m: dict(pred_first=wmean((pred[m] == 0).astype(float), wb),
                                      pred_last=wmean((pred[m] == k - 1).astype(float), wb)) for m in MODELS})
    # E3 confident errors: share of a model's errors made with confidence >= 0.9 (benchmark-weighted)
    wbm = bench_weights(bench, M)
    cat["confident_errors"] = {m: dict(err=wmean(1 - Y[m], wbm),
                                       err_conf90=wmean(((1 - Y[m]) * (P[m]["conf"] >= 0.9)).astype(float), wbm),
                                       share_of_errors=wmean(((1 - Y[m]) * (P[m]["conf"] >= 0.9)).astype(float), wbm) / wmean(1 - Y[m], wbm))
                               for m in MODELS}
    # E4 near-chance benchmarks (official skill < 0.05)
    cat["near_chance"] = {m: sorted(r["name"] for r in rows if r[f"skill_{m}"] < 0.05) for m in MODELS}
    out["categories"] = cat

    # F. error nesting across sizes (v2 family): of the larger model's errors, share also made by the smaller one
    nest = {}
    for x, y in [("c17m", "c1b"), ("c17m", "c150v2"), ("c68m", "c1b"), ("c150v2", "c1b"), ("c32v1", "c32v2"),
                 ("c150v1", "c150v2")]:
        ex, ey = (Y[x] == 0) & M, (Y[y] == 0) & M
        nest[f"{x}|{y}"] = dict(err_small=float(ex.sum() / M.sum()), err_large=float(ey.sum() / M.sum()),
                                large_err_also_small=float((ex & ey).sum() / ey.sum()),
                                small_err_fixed_by_large=float((ex & ~ey).sum() / ex.sum()),
                                agree_pred=float((P[x]["pred_idx"] == P[y]["pred_idx"])[M].mean()))
    out["nesting"] = nest

    # G. Prumo complementarity (descriptive): fields Prumo claims (non-uniform output) vs Crivo correctness
    pr = P["prumo"]
    cl = M & ~pr["uniform"]; un = M & pr["uniform"]
    comp = dict(claimed_fields=int(cl.sum()), claimed_share=float(cl.sum() / M.sum()),
                claimed_benchmarks=int(len(np.unique(bench[cl]))), prumo_acc_claimed=float(Y["prumo"][cl].mean()),
                prumo_norm_claimed=float(N["prumo"][cl].mean()), per_model={}, per_bench=[])
    for m in CRIVOS:
        yc, yp = Y[m][cl], Y["prumo"][cl]
        comp["per_model"][m] = dict(crivo_acc_claimed=float(yc.mean()), crivo_acc_unclaimed=float(Y[m][un].mean()),
                                    crivo_norm_claimed=float(N[m][cl].mean()), crivo_norm_unclaimed=float(N[m][un].mean()),
                                    both_right=float((yc * yp).mean()), prumo_only=float(((1 - yc) * yp).mean()),
                                    crivo_only=float((yc * (1 - yp)).mean()), both_wrong=float(((1 - yc) * (1 - yp)).mean()),
                                    p_crivo_wrong_given_prumo_right=float(((1 - yc) * yp).sum() / max(yp.sum(), 1)),
                                    p_crivo_wrong_overall=float(1 - Y[m][M].mean()))
    for b in np.unique(bench[cl]):
        mk = cl & (bench == b)
        if mk.sum() < 30:
            continue
        comp["per_bench"].append(dict(bench=names[b], claimed=int(mk.sum()), claimed_share=float(mk.sum() / (M & (bench == b)).sum()),
                                      prumo_acc=float(Y["prumo"][mk].mean()), c17m_acc=float(Y["c17m"][mk].mean()),
                                      c1b_acc=float(Y["c1b"][mk].mean()),
                                      prumo_only_vs_17m=float(((1 - Y["c17m"][mk]) * Y["prumo"][mk]).mean()),
                                      prumo_only_vs_1b=float(((1 - Y["c1b"][mk]) * Y["prumo"][mk]).mean())))
    comp["per_bench"].sort(key=lambda r: -r["claimed"])
    out["prumo_complementarity"] = comp
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
