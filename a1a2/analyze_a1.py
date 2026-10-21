"""A1 (joint vs per-option reader) per-benchmark analysis of the GPU-2 grid. CPU only, no models, streams files.

    python3 -I a1a2/analyze_a1.py      # from the repository root; KIT=<decision-index checkout>            # ~ a few minutes, < 2 GB RAM
    # compute-matched reference (GPU-3 J-comp) instead of the step-matched J-steps:
    python3 -I a1a2/analyze_a1.py --ref Jcomp --ev_ref results_gpu3/ev --out results_gpu3/analysis

Inputs (all local):
  results_gpu2/ev/<arm>_s<seed>/{pub15,btzsc500,s1_60}.jsonl (+ .score.json), e4/{orig,perm1-3,rev,yn}.jsonl
  pub15 requests: rebuilt in memory from $KIT/suite-0.3 exactly as a1a2/build_eval.py:pub15 (seed 0, 15% per
      benchmark, at most 600); the ids are checked against every result file and the per-benchmark n/acc against
      every pub15.score.json.
  btzsc500 / s1_60 requests: the rows of stats/gpu1/req/{btzsc_full,s1_all}.jsonl whose ids are in the result files
      (same adapters; BTZSC macro-F1 is recomputed and checked against btzsc500.score.json).
  E4: stats/gpu1/req/e4_orig.jsonl; perm1-3/rev/yn option orders regenerated with e4/perturb.py's rule.
  E6: results_gpu/e6/*.gpu.json (+ logs/e6_redo.log for validity), results_cpu/e6cpu/*.t16.json.
  Token counts: the released Crivo tokenizer (tokenizer.json of tardellirs/crivo-32m, `tokenizers` only).

Method:
  * Results: last line per id wins; a non-ok line or a missing answer is wrong (as score_pub.py).
  * pub15 per benchmark: accuracy over judged questions, chance = mean 1/n, skill = clip((acc-ch)/(1-ch), 0, 1),
    per run, then mean over 3 seeds. Differences (P-Jsteps, P-J, Jsteps-J): Welch 95% CI over seeds (3 vs 3) and a
    paired cluster bootstrap (B=2000): requests resampled within each benchmark (questions of one request stay
    together), the per-seed skill recomputed on every resample and averaged over seeds; the overall mean-skill CI
    resamples every benchmark independently (stratified).
  * Breakdowns (question type, number of options, input tokens of the joint sequence, questions per request):
    pooled accuracy and bench-weighted unclipped skill (benchmark = equal weight among benchmarks with >= 20
    questions in the cell), with the same stratified cluster bootstrap.
  * Option position: choice questions with >= 3 options, gold position first / middle / last, on pub15 (native
    order) and on E4 (orig + perm1-3 + rev: the same questions with the gold moved), plus binary questions on E4
    (gold first vs second, orig + yn + perms). Per-option P cannot see position, so it is the control.
  * S1-Arena: per-domain task-mean accuracy and Brier from s1_60.score.json; Welch over seeds and a task bootstrap
    within the domain. BTZSC: per-dataset macro-F1 from btzsc500.score.json; Welch over seeds and an item bootstrap
    of the seed-mean macro-F1 difference.
  * Cost: forward sequences per request (joint: 1 per question; per-option: 1 per option, noul = 2) and tokens
    encoded per request (joint: CLS+instr+SEP+state+SEP+sum(OPT+option); per-option: that prefix once per option).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

EXP = Path(__file__).resolve().parents[1]
KIT = Path(__import__("os").environ.get("KIT", str(Path.home() / "decision-index")))
sys.path.insert(0, str(EXP / "e4"))
from perturb import NOT_INDEX, is_yn  # noqa: E402

ARMS = ["J", "Jsteps", "P"]
SEEDS = [0, 1, 2]
RUNS = [(a, s) for a in ARMS for s in SEEDS]
PAIRS = [("P", "Jsteps"), ("P", "J"), ("Jsteps", "J")]
REF = "Jsteps"            # the joint comparison arm (--ref); "Jcomp" reads its runs from --ev_ref
PJ = "P-Jsteps"
EV_REF = None
NOUL_FALSE, NOUL_TRUE = "No.", "Yes."


# ---------------------------------------------------------------- shared helpers (copied from train/model.py, no torch)
def render(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    return json.dumps(c, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def option_texts(qtype, criteria):
    if qtype == "choice":
        out = []
        for k in criteria:
            d = render(criteria[k])
            out.append(f"{k}: {d}" if d and d != k else str(k))
        return out
    if qtype == "noul":
        c = criteria if isinstance(criteria, dict) else {}
        f, t = render(c.get("false")), render(c.get("true"))
        return [f"false: {f}" if f else NOUL_FALSE, f"true: {t}" if t else NOUL_TRUE]
    if qtype == "score":
        lv = list(criteria.values()) if isinstance(criteria, dict) else list(criteria)
        return [f"{i}: {render(x)}" for i, x in enumerate(lv)]
    raise ValueError(qtype)


def read(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


class Tok:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(path))

    def n(self, text):
        return len(self.t.encode(text, add_special_tokens=False).ids) if text else 0


def last_lines(p):
    """id -> answers dict (None when the last line is not ok). Last line per id wins."""
    out = {}
    for r in read(p):
        out[r["id"]] = r["response"]["answers"] if r.get("status") == "ok" else None
    return out


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = float(x.mean() - y.mean())
    if x.std(ddof=1) == 0 and y.std(ddof=1) == 0:
        return d, [d, d]
    ci = stats.ttest_ind(x, y, equal_var=False).confidence_interval()
    return d, [float(ci[0]), float(ci[1])]


def pct(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def f3(x, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.3f}" if sign else f"{x:.3f}"


def fci(d, ci):
    return f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


# ---------------------------------------------------------------- pub15 rebuild (= build_eval.py:pub15, streamed)
def pub15_requests(suite: Path, tok: Tok):
    excl = set(json.load(open(suite / "excluded-questions.json"))["rows"])
    files = [(suite / "selected-rows.jsonl.gz", True), (suite / "added-rows.jsonl.gz", False), (suite / "gsm8k-rows.jsonl.gz", False)]

    def keep(r, base):
        ev = r.get("_evaluation", {})
        ds = ev.get("dataset") or r.get("family")
        if ds in NOT_INDEX or (base and ds == "GSM8K") or ev.get("run_id") in excl:
            return None
        return ds
    cnt = defaultdict(int)
    for f, base in files:
        for r in read(f):
            ds = keep(r, base)
            if ds is not None:
                cnt[ds] += 1
    pick = {}
    for ds in sorted(cnt):
        n = cnt[ds]
        k = min(600, max(1, round(0.15 * n)))
        pick[ds] = set(random.Random(0).sample(range(n), k))   # same positions as sample(rows, k)
    seen = defaultdict(int)
    reqs = {}
    for f, base in files:
        for r in read(f):
            ds = keep(r, base)
            if ds is None:
                continue
            i = seen[ds]; seen[ds] += 1
            if i in pick[ds]:
                rid = r["_evaluation"]["run_id"]
                reqs[rid] = compact(rid, ds, r["state"], r["questions"], r.get("expected") or {}, tok)
    return reqs


def compact(rid, fam, state, questions, expected, tok, ref_key=None):
    st = tok.n(render(state))
    qs = []
    for k, q in questions.items():
        texts = option_texts(q["type"], q.get("criteria"))
        ins = tok.n(render(q.get("instructions")))
        ot = [tok.n(t) for t in texts]
        keys = list(q["criteria"]) if q["type"] == "choice" else None
        qs.append({"k": k, "type": q["type"], "n": len(texts), "keys": keys, "gold": expected.get(k),
                   "joint_tok": 3 + ins + st + sum(1 + x for x in ot),
                   "peropt_tok": sum(3 + ins + st + 1 + x for x in ot)})
    return {"id": rid, "fam": fam, "state_tok": st, "qs": qs}


def judge(q, x):
    """(judged, correct, chosen position or None, gold position or None) as score_pub.py."""
    g = q["gold"]
    if g is None:
        return False, False, None, None
    if q["type"] == "noul":
        gold = g if isinstance(g, bool) else str(g).lower() in ("true", "1", "yes")
        return True, x is not None and ((x["noul"] >= 0.5) == gold), None, None
    gp = q["keys"].index(g) if q["keys"] and g in q["keys"] else None
    cp = q["keys"].index(x["choice"]) if x is not None and q["keys"] and x.get("choice") in q["keys"] else None
    return True, x is not None and x.get("choice") == g, cp, gp


# ---------------------------------------------------------------- bootstrap machinery
class Cells:
    """Question table -> per benchmark request-level sums for the 9 runs; stratified cluster bootstrap."""

    def __init__(self, Q, hits, B, seed=0):
        self.Q, self.hits, self.B = Q, hits, B
        self.rng = np.random.default_rng(seed)

    def bench_sums(self, mask):
        Q = self.Q
        idx = np.nonzero(mask)[0]
        out = {}
        if not len(idx):
            return out
        b = Q["bench"][idx]; r = Q["req"][idx]
        for bi in np.unique(b):
            sel = idx[b == bi]
            rr, inv = np.unique(Q["req"][sel], return_inverse=True)
            n = np.bincount(inv).astype(float)
            ch = np.bincount(inv, weights=Q["chance"][sel])
            h = np.stack([np.bincount(inv, weights=self.hits[sel, j], minlength=len(rr)) for j in range(len(RUNS))], 1)
            out[int(bi)] = (n, ch, h)
        return out

    def stat(self, sums, min_q=1, clip=False, boot=True):
        """Per bench and bench-mean acc/skill per run; bootstrap draws (B, runs) of the bench-mean skill/acc."""
        per = {}
        accB, sklB = [], []
        for bi, (n, ch, h) in sums.items():
            if n.sum() < min_q:
                continue
            N, C, H = n.sum(), ch.sum(), h.sum(0)
            acc = H / N; c = C / N
            sk = (acc - c) / (1 - c) if c < 1 else np.zeros_like(acc)
            if clip:
                sk = np.clip(sk, 0, 1)
            per[bi] = {"n": int(N), "requests": len(n), "chance": float(c), "acc": acc, "skill": sk}
            if boot:
                R = len(n)
                w = self.rng.multinomial(R, np.full(R, 1 / R), size=self.B).astype(float)
                Nb, Cb, Hb = w @ n, w @ ch, w @ h
                ab = Hb / Nb[:, None]; cb = (Cb / Nb)[:, None]
                sb = np.where(cb < 1, (ab - cb) / np.where(cb < 1, 1 - cb, 1), 0.0)
                if clip:
                    sb = np.clip(sb, 0, 1)
                accB.append(ab); sklB.append(sb)
                per[bi]["accB"] = ab; per[bi]["skillB"] = sb
        if not per:
            return per, None
        mean = {"acc": np.mean([v["acc"] for v in per.values()], 0), "skill": np.mean([v["skill"] for v in per.values()], 0)}
        if boot:
            mean["accB"] = np.mean(accB, 0); mean["skillB"] = np.mean(sklB, 0)
        return per, mean


def arm_mean(v):
    """(runs,) or (B, runs) -> dict arm -> seed mean."""
    return {a: v[..., [i for i, (aa, _) in enumerate(RUNS) if aa == a]].mean(-1) for a in ARMS}


def arm_vals(v):
    return {a: [float(v[i]) for i, (aa, _) in enumerate(RUNS) if aa == a] for a in ARMS}


def diffs(point, bootv=None):
    """point: (runs,), bootv: (B, runs) -> {pair: {diff, welch, boot}}"""
    av = arm_vals(point)
    out = {}
    mb = arm_mean(bootv) if bootv is not None else None
    for x, y in PAIRS:
        d, w = welch(av[x], av[y])
        e = {"diff": d, "welch_ci95": w}
        if mb is not None:
            e["boot_ci95"] = pct(mb[x] - mb[y])
        out[f"{x}-{y}"] = e
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(KIT / "suite-0.3"))
    ap.add_argument("--ev", default=str(EXP / "results_gpu2" / "ev"))
    ap.add_argument("--req", default=str(EXP / "stats" / "gpu1" / "req"))
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (default: downloaded from tardellirs/crivo-32m)")
    ap.add_argument("--out", default=str(EXP / "results_gpu2" / "analysis"))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--ref", default="Jsteps", help="joint arm compared with P (Jsteps, or Jcomp from --ev_ref)")
    ap.add_argument("--ev_ref", default=None, help="run dir of the --ref arm when it is not under --ev "
                    "(Jcomp: results_gpu3/ev)")
    a = ap.parse_args()
    global ARMS, RUNS, PAIRS, REF, PJ, EV_REF
    REF, PJ = a.ref, f"P-{a.ref}"
    ARMS = ["J", REF, "P"]
    RUNS = [(x, s) for x in ARMS for s in SEEDS]
    PAIRS = [("P", REF), ("P", "J"), (REF, "J")]
    EV_REF = Path(a.ev_ref) if a.ev_ref else None
    EV, REQ, OUT = Path(a.ev), Path(a.req), Path(a.out)

    def D(arm, s):
        return (EV_REF if (arm == REF and EV_REF is not None) else EV) / f"{arm}_s{s}"
    OUT.mkdir(parents=True, exist_ok=True)
    if a.tokenizer is None:   # released Crivo-32M tokenizer (same Ettin vocabulary + [OPT] for every Crivo)
        from huggingface_hub import hf_hub_download
        a.tokenizer = hf_hub_download("tardellirs/crivo-32m", "tokenizer.json", revision="c14221dd5cb4953fa83dc40ccf3b90b37126668e")
    tok = Tok(a.tokenizer)
    J = {"meta": {"arms": ARMS, "ref": REF, "ev_ref": str(EV_REF) if EV_REF else None, "seeds": SEEDS, "B": a.B, "pairs": [f"{x}-{y}" for x, y in PAIRS]}, "checks": {}}
    md = []

    # ============================== pub15
    print("pub15: rebuilding requests from the suite ...", flush=True)
    reqs = pub15_requests(Path(a.suite), tok)
    ids = set(reqs)
    fams = sorted({r["fam"] for r in reqs.values()})
    bidx = {f: i for i, f in enumerate(fams)}
    rows = []                                  # judged questions
    order = sorted(reqs)                       # request index
    ridx = {rid: i for i, rid in enumerate(order)}
    for rid in order:
        r = reqs[rid]
        nq = len(r["qs"])
        for q in r["qs"]:
            if q["gold"] is None:
                continue
            gp = q["keys"].index(q["gold"]) if q["type"] == "choice" and q["gold"] in q["keys"] else -1
            rows.append((ridx[rid], bidx[r["fam"]], q, nq, gp))
    nQ = len(rows)
    Q = {"req": np.array([x[0] for x in rows]), "bench": np.array([x[1] for x in rows]),
         "chance": np.array([0.5 if x[2]["type"] == "noul" else 1 / x[2]["n"] for x in rows]),
         "type": np.array([x[2]["type"] for x in rows]), "nopt": np.array([x[2]["n"] for x in rows]),
         "tok": np.array([x[2]["joint_tok"] for x in rows]), "nq": np.array([x[3] for x in rows]),
         "gpos": np.array([x[4] for x in rows])}
    hits = np.zeros((nQ, len(RUNS)), np.float32)
    cpos = np.full((nQ, len(RUNS)), -1, np.int16)
    id_check, score_check = {}, {}
    for j, (arm, s) in enumerate(RUNS):
        d = D(arm, s)
        res = last_lines(d / "pub15.jsonl")
        id_check[f"{arm}_s{s}"] = {"result_ids": len(res), "request_ids": len(ids), "equal": set(res) == ids}
        assert set(res) == ids, f"pub15 ids differ for {arm}_s{s}"
        for i, (ri, bi, q, nq, gp) in enumerate(rows):
            ans = res[order[ri]]
            x = ans.get(q["k"]) if ans else None
            _, ok, cp, _ = judge(q, x)
            hits[i, j] = ok
            if cp is not None:
                cpos[i, j] = cp
        del res
        sc = json.load(open(d / "pub15.score.json"))["per_benchmark"]
        mx = 0.0
        for f in fams:
            m = Q["bench"] == bidx[f]
            mx = max(mx, abs(hits[m, j].mean() - sc[f]["acc"]), abs(m.sum() - sc[f]["n"]))
        score_check[f"{arm}_s{s}"] = mx
        print(f"  {arm}_s{s}: ids ok, max |acc - score.json| = {mx:.2e}", flush=True)
    J["checks"]["pub15_ids"] = id_check
    J["checks"]["pub15_max_abs_diff_vs_score_json"] = score_check
    assert max(score_check.values()) < 1e-9

    C = Cells(Q, hits, a.B, seed=0)
    allmask = np.ones(nQ, bool)
    sums_all = C.bench_sums(allmask)
    per_c, mean_c = C.stat(sums_all, clip=True)
    per_u, mean_u = C.stat(sums_all, clip=False)

    # pooled question accuracy, paired (as summarize.py) + bench-weighted
    seedmean = {arm: hits[:, [i for i, (aa, _) in enumerate(RUNS) if aa == arm]].mean(1) for arm in ARMS}
    acos = Q["bench"] == bidx.get("ACOS", -1)
    pooled = {}
    for x, y in PAIRS:
        dq = seedmean[x] - seedmean[y]
        pooled[f"{x}-{y}"] = {"all": float(dq.mean()), "excl_ACOS": float(dq[~acos].mean()),
                              "ACOS_share_of_questions": float(acos.mean())}

    bench_rows = []
    for f in fams:
        bi = bidx[f]
        pc, pu = per_c[bi], per_u[bi]
        m = Q["bench"] == bi
        reqs_b = len(np.unique(Q["req"][m]))
        e = {"benchmark": f, "questions": pc["n"], "requests": reqs_b, "q_per_request": pc["n"] / reqs_b,
             "mean_options": float(Q["nopt"][m].mean()), "noul_share": float((Q["type"][m] == "noul").mean()),
             "median_joint_tokens": float(np.median(Q["tok"][m])), "chance": pc["chance"],
             "acc": {k: float(v) for k, v in arm_mean(pc["acc"]).items()},
             "skill": {k: float(v) for k, v in arm_mean(pc["skill"]).items()},
             "skill_unclipped": {k: float(v) for k, v in arm_mean(pu["skill"]).items()},
             "skill_sd": {k: float(np.std(v, ddof=1)) for k, v in arm_vals(pc["skill"]).items()},
             "d_skill": diffs(pc["skill"], pc["skillB"]), "d_acc": diffs(pc["acc"], pc["accB"]),
             "d_skill_unclipped": diffs(pu["skill"], pu["skillB"])}
        bench_rows.append(e)
    overall = {"mean_skill": {k: float(v) for k, v in arm_mean(mean_c["skill"]).items()},
               "mean_skill_unclipped": {k: float(v) for k, v in arm_mean(mean_u["skill"]).items()},
               "mean_acc_benchweighted": {k: float(v) for k, v in arm_mean(mean_c["acc"]).items()},
               "d_mean_skill": diffs(mean_c["skill"], mean_c["skillB"]),
               "d_mean_skill_unclipped": diffs(mean_u["skill"], mean_u["skillB"]),
               "d_mean_acc_benchweighted": diffs(mean_c["acc"], mean_c["accB"]),
               "d_pooled_question_acc": pooled, "questions": nQ, "requests": len(order), "benchmarks": len(fams)}
    # clipping: how many bench x run cells are clipped at 0
    clipped = {arm: int(sum((np.array(arm_vals(per_u[b]["skill"])[arm]) < 0).sum() for b in per_u)) for arm in ARMS}
    overall["clipped_cells_of_111"] = clipped
    # decomposition: contribution of each benchmark to mean skill diff and to pooled acc diff
    N = nQ
    for e in bench_rows:
        for pr in (PJ, "P-J"):
            e.setdefault("contrib_mean_skill", {})[pr] = e["d_skill"][pr]["diff"] / len(fams)
            e.setdefault("contrib_pooled_acc", {})[pr] = e["d_acc"][pr]["diff"] * e["questions"] / N
    rob = {}
    for pr in (PJ, "P-J"):
        d = np.array([e["d_skill"][pr]["diff"] for e in bench_rows])
        lo = np.array([e["d_skill"][pr]["boot_ci95"][0] for e in bench_rows]); hi = np.array([e["d_skill"][pr]["boot_ci95"][1] for e in bench_rows])
        srt_ = np.sort(d)[::-1]
        rob[pr] = {"median_bench_d_skill": float(np.median(d)), "benchmarks_positive": int((d > 0).sum()), "benchmarks_negative": int((d < 0).sum()),
                   "ci_above_0": int((lo > 0).sum()), "ci_below_0": int((hi < 0).sum()),
                   "mean_without_top1": float(srt_[1:].mean()), "mean_without_top2": float(srt_[2:].mean()), "mean_without_top4": float(srt_[4:].mean()),
                   "floor_benchmarks_all_arms_skill_lt_0.03": [e["benchmark"] for e in bench_rows if max(e["skill"].values()) < 0.03]}
    overall["robustness"] = rob
    J["pub15"] = {"overall": overall, "per_benchmark": bench_rows}

    # benchmark-level association with options / q per request / length
    assoc = {}
    for pr in (PJ, "P-J"):
        ds_ = np.array([e["d_skill_unclipped"][pr]["diff"] for e in bench_rows])
        for feat in ("mean_options", "q_per_request", "median_joint_tokens", "noul_share", "chance"):
            v = np.array([e[feat] for e in bench_rows])
            rho = stats.spearmanr(v, ds_)
            assoc[f"{pr}|{feat}"] = {"spearman_rho": float(rho.statistic), "p": float(rho.pvalue), "benchmarks": len(v)}
    J["pub15"]["bench_level_spearman_vs_d_skill_unclipped"] = assoc

    # breakdowns
    def bins_of(v, edges, labels):
        out = np.empty(len(v), dtype=object)
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            out[(v >= lo) & (v < hi)] = lab
        return out
    breakdowns = {
        "question_type": Q["type"].astype(object),
        "n_options": bins_of(Q["nopt"], [2, 3, 5, 10, 30, 10**9], ["2", "3-4", "5-9", "10-29", "30+"]),
        "input_tokens_joint": bins_of(Q["tok"], [0, 128, 512, 2048, 8192, 10**12], ["<128", "128-511", "512-2047", "2048-8191", ">=8192"]),
        "questions_per_request": bins_of(Q["nq"], [1, 2, 5, 17, 10**9], ["1", "2-4", "5-16", "17+"]),
    }
    choice2 = (Q["type"] == "choice") & (Q["nopt"] == 2)
    breakdowns["type_x_options"] = np.where(Q["type"] == "noul", "noul", np.where(choice2, "choice-2", "choice-3+")).astype(object)
    BD = {}
    for name, lab in breakdowns.items():
        rows_ = []
        labels = [l for l in (["2", "3-4", "5-9", "10-29", "30+"] if name == "n_options" else
                              ["<128", "128-511", "512-2047", "2048-8191", ">=8192"] if name == "input_tokens_joint" else
                              ["1", "2-4", "5-16", "17+"] if name == "questions_per_request" else
                              sorted(set(lab.tolist())))]
        for L in labels:
            m = lab == L
            if not m.any():
                continue
            sums = C.bench_sums(m)
            per, mean = C.stat(sums, min_q=20, clip=False)
            hm = {arm: float(seedmean[arm][m].mean()) for arm in ARMS}
            ch = float(Q["chance"][m].mean())
            e = {"bin": L, "questions": int(m.sum()), "benchmarks": int(len(np.unique(Q["bench"][m]))),
                 "benchmarks_ge20": len(per), "pooled_acc": hm, "pooled_chance": ch,
                 "pooled_skill": {k: (v - ch) / (1 - ch) for k, v in hm.items()},
                 "d_pooled_acc": {f"{x}-{y}": hm[x] - hm[y] for x, y in PAIRS}}
            if mean is not None:
                e["bw_skill"] = {k: float(v) for k, v in arm_mean(mean["skill"]).items()}
                e["d_bw_skill"] = diffs(mean["skill"], mean["skillB"])
                e["d_bw_acc"] = diffs(mean["acc"], mean["accB"])
            rows_.append(e)
        BD[name] = rows_
    J["pub15"]["breakdowns"] = BD

    # position on pub15 (native order): choice with >= 3 options
    def pos_bin(gp, n):
        return np.where(gp == 0, "first", np.where(gp == n - 1, "last", "middle"))
    m3 = (Q["type"] == "choice") & (Q["nopt"] >= 3) & (Q["gpos"] >= 0)
    pb = np.full(nQ, None, dtype=object); pb[m3] = pos_bin(Q["gpos"][m3], Q["nopt"][m3])
    POS = {"pub15_native_order": []}
    for L in ("first", "middle", "last"):
        m = pb == L
        sums = C.bench_sums(m)
        per, mean = C.stat(sums, min_q=10, clip=False)
        POS["pub15_native_order"].append({"gold_position": L, "questions": int(m.sum()), "benchmarks": int(len(np.unique(Q["bench"][m]))),
                                          "pooled_acc": {k: float(seedmean[k][m].mean()) for k in ARMS},
                                          "bw_acc": {k: float(v) for k, v in arm_mean(mean["acc"]).items()} if mean else None,
                                          "d_bw_acc": diffs(mean["acc"], mean["accB"]) if mean else None})
    # chosen-position distribution (choice >= 3): share choosing first / last vs gold share
    cp_share = {}
    for j, (arm, s) in enumerate(RUNS):
        c = cpos[m3, j]; n = Q["nopt"][m3]
        cp_share[f"{arm}_s{s}"] = {"first": float((c == 0).mean()), "last": float((c == n - 1).mean())}
    gold_share = {"first": float((Q["gpos"][m3] == 0).mean()), "last": float((Q["gpos"][m3] == Q["nopt"][m3] - 1).mean()),
                  "uniform_first": float((1 / Q["nopt"][m3]).mean())}
    POS["pub15_chosen_position_share"] = {"runs": cp_share, "gold": gold_share,
                                          "mean": {arm: {k: float(np.mean([cp_share[f"{arm}_s{s}"][k] for s in SEEDS])) for k in ("first", "last")} for arm in ARMS}}

    # ============================== E4 position (same questions, gold moved)
    print("E4: position analysis ...", flush=True)
    e4 = {r["id"]: r for r in read(REQ / "e4_orig.jsonl")}

    def order_for(rid, qk, n, v):
        if v == "orig":
            return list(range(n))
        if v == "rev":
            return list(range(n))[::-1]
        if v == "yn":
            return [1, 0]
        k = int(v[-1])
        order = list(range(n))
        rng = random.Random(int(hashlib.sha256(f"{rid}|{qk}|{k}".encode()).hexdigest()[:12], 16))
        while n > 1 and order == list(range(n)):
            rng.shuffle(order)
        return order
    VARS = ["orig", "perm1", "perm2", "perm3", "rev", "yn"]
    erows = []          # (request index, bench, variant, n, gold pos in variant)
    eids = sorted(e4); eri = {r: i for i, r in enumerate(eids)}
    efams = sorted({r["family"] for r in e4.values()}); ebi = {f: i for i, f in enumerate(efams)}
    ekeys = []
    eqid, qid = [], {}
    for v in VARS:
        for rid in eids:
            r = e4[rid]
            for qk, q in r["questions"].items():
                if v == "yn" and not is_yn(q):
                    continue
                g = (r.get("expected") or {}).get(qk)
                keys0 = list(q["criteria"])
                if g is None or g not in keys0:
                    continue
                o = order_for(rid, qk, len(keys0), v)
                keys = [keys0[i] for i in o]
                erows.append((eri[rid], ebi[r["family"]], VARS.index(v), len(keys), keys.index(g)))
                ekeys.append((v, rid, qk, g))
                eqid.append(qid.setdefault((rid, qk), len(qid)))
    EQ = {"req": np.array([x[0] for x in erows]), "bench": np.array([x[1] for x in erows]), "var": np.array([x[2] for x in erows]),
          "n": np.array([x[3] for x in erows]), "gpos": np.array([x[4] for x in erows])}
    EQ["chance"] = 1 / EQ["n"]
    ehits = np.zeros((len(erows), len(RUNS)), np.float32)
    e4check = {}
    for j, (arm, s) in enumerate(RUNS):
        res = {v: last_lines(D(arm, s) / "e4" / f"{v}.jsonl") for v in VARS}
        miss = 0
        for i, (v, rid, qk, g) in enumerate(ekeys):
            ans = res[v].get(rid)
            if ans is None:
                miss += 1
            ehits[i, j] = ans is not None and ans.get(qk, {}).get("choice") == g
        e4check[f"{arm}_s{s}"] = {"missing_or_failed": miss, "orig_ids_equal": set(res["orig"]) == set(e4)}
        del res
    J["checks"]["e4"] = e4check
    EC = Cells(EQ, ehits, a.B, seed=1)
    ePOS = []
    five = EQ["var"] <= VARS.index("rev")
    m3e = five & (EQ["n"] >= 3)
    pbe = np.full(len(erows), None, dtype=object); pbe[m3e] = pos_bin(EQ["gpos"][m3e], EQ["n"][m3e])
    for L in ("first", "middle", "last"):
        m = pbe == L
        per, mean = EC.stat(EC.bench_sums(m), min_q=10, clip=False)
        ePOS.append({"set": "E4 orig+perm1-3+rev, choice >=3 options", "gold_position": L, "questions": int(m.sum()),
                     "bw_acc": {k: float(v) for k, v in arm_mean(mean["acc"]).items()},
                     "d_bw_acc": diffs(mean["acc"], mean["accB"])})
    # first - last within arm (bootstrap of the bench-weighted gap)
    gaps = {}
    mf, ml = pbe == "first", pbe == "last"
    pf, mfm = EC.stat(EC.bench_sums(mf), min_q=10, clip=False)
    pl, mlm = EC.stat(EC.bench_sums(ml), min_q=10, clip=False)
    common = sorted(set(pf) & set(pl))
    gpt = np.mean([pf[b]["acc"] - pl[b]["acc"] for b in common], 0)
    gB = np.mean([pf[b]["accB"] - pl[b]["accB"] for b in common], 0)
    ga, gb = arm_mean(gpt), arm_mean(gB)
    for arm in ARMS:
        gaps[arm] = {"first_minus_last_bw_acc": float(ga[arm]), "boot_ci95": pct(gb[arm]), "benchmarks": len(common)}
    for x, y in PAIRS:
        gaps[f"{x}-{y}"] = {"diff_of_gaps": float(ga[x] - ga[y]), "boot_ci95": pct(gb[x] - gb[y])}
    # binary questions: gold first vs second, all six variants
    m2 = EQ["n"] == 2
    bin2 = []
    for L, gp in (("gold first", 0), ("gold second", 1)):
        m = m2 & (EQ["gpos"] == gp)
        per, mean = EC.stat(EC.bench_sums(m), min_q=10, clip=False)
        bin2.append({"gold_position": L, "questions": int(m.sum()), "bw_acc": {k: float(v) for k, v in arm_mean(mean["acc"]).items()},
                     "d_bw_acc": diffs(mean["acc"], mean["accB"])})
    # accuracy per variant (pooled)
    pervar = {}
    for vi, v in enumerate(VARS):
        m = EQ["var"] == vi
        pervar[v] = {"questions": int(m.sum()), **{arm: float(arm_mean(ehits[m].mean(0))[arm]) for arm in ARMS}}
    POS["e4_gold_position_choice3plus"] = ePOS
    POS["e4_first_minus_last"] = gaps
    POS["e4_binary_gold_position"] = bin2
    POS["e4_pooled_acc_per_variant"] = pervar
    # within-question contrasts (each question against itself; P is position-blind, so its contrast is ~0 by construction)
    eqid = np.array(eqid); nqid = len(qid)
    q_req = np.zeros(nqid, int); q_b = np.zeros(nqid, int); q_n = np.zeros(nqid, int)
    q_req[eqid] = EQ["req"]; q_b[eqid] = EQ["bench"]; q_n[eqid] = EQ["n"]
    WQ = {}
    edge = five & ((EQ["gpos"] == 0) | (EQ["gpos"] == EQ["n"] - 1)) & (EQ["n"] >= 3)
    mid = five & (EQ["gpos"] > 0) & (EQ["gpos"] < EQ["n"] - 1)
    b_first = (EQ["n"] == 2) & (EQ["gpos"] == 0)
    b_second = (EQ["n"] == 2) & (EQ["gpos"] == 1)
    for name, ma, mb_ in (("edge_minus_middle_choice3plus", edge, mid), ("first_minus_second_binary", b_first, b_second)):
        ca = np.bincount(eqid[ma], minlength=nqid); cb = np.bincount(eqid[mb_], minlength=nqid)
        ok = (ca > 0) & (cb > 0)
        dq = np.stack([np.bincount(eqid[ma], weights=ehits[ma, j], minlength=nqid)[ok] / ca[ok]
                       - np.bincount(eqid[mb_], weights=ehits[mb_, j], minlength=nqid)[ok] / cb[ok] for j in range(len(RUNS))], 1)
        TQ = {"req": q_req[ok], "bench": q_b[ok], "chance": np.zeros(ok.sum())}
        per, mean = Cells(TQ, dq, a.B, seed=4).stat(Cells(TQ, dq, a.B, seed=4).bench_sums(np.ones(ok.sum(), bool)), min_q=10, clip=False)
        am, ab = arm_mean(mean["acc"]), arm_mean(mean["accB"])
        WQ[name] = {"questions": int(ok.sum()), "benchmarks": len(per),
                    **{arm: {"bw_contrast": float(am[arm]), "boot_ci95": pct(ab[arm])} for arm in ARMS},
                    **{f"{x}-{y}": {"diff": float(am[x] - am[y]), "boot_ci95": pct(ab[x] - ab[y])} for x, y in PAIRS}}
    POS["e4_within_question"] = WQ
    J["position"] = POS
    del ehits, EQ, EC

    # ============================== S1-Arena per domain (score files)
    print("S1 / BTZSC ...", flush=True)
    s1sc = {f"{arm}_s{s}": json.load(open(D(arm, s) / "s1_60.score.json")) for arm, s in RUNS}
    doms = sorted(next(iter(s1sc.values()))["domains"])
    rng = np.random.default_rng(2)
    S1 = []
    dchk = 0.0
    for dname in doms + ["ALL"]:
        accv = np.array([(s1sc[f"{arm}_s{s}"]["summary"]["accuracy"] if dname == "ALL" else s1sc[f"{arm}_s{s}"]["domains"][dname]["accuracy"]) for arm, s in RUNS])
        brv = np.array([(s1sc[f"{arm}_s{s}"]["summary"]["brier"] if dname == "ALL" else s1sc[f"{arm}_s{s}"]["domains"][dname]["brier"]) for arm, s in RUNS])
        tasks = sorted(t for t, v in s1sc["P_s0"]["tasks"].items() if dname == "ALL" or v["domain"] == dname)
        T = np.array([[s1sc[f"{arm}_s{s}"]["tasks"][t]["accuracy"] for arm, s in RUNS] for t in tasks])   # tasks x runs
        if dname != "ALL":
            dchk = max(dchk, float(np.abs(T.mean(0) - accv).max()))
        w = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=a.B) / len(tasks)
        Tb = w @ T
        e = {"domain": dname, "tasks": len(tasks), "acc": {k: float(v) for k, v in arm_mean(accv).items()},
             "brier": {k: float(v) for k, v in arm_mean(brv).items()}, "d_acc": diffs(accv), "d_brier": diffs(brv)}
        mb = arm_mean(Tb)
        tm = arm_mean(T)
        for x, y in PAIRS:
            e["d_acc"][f"{x}-{y}"]["task_boot_ci95"] = pct(mb[x] - mb[y])
            e["d_acc"][f"{x}-{y}"]["tasks_x_better"] = int((tm[x] > tm[y]).sum())
            e["d_acc"][f"{x}-{y}"]["tasks_y_better"] = int((tm[x] < tm[y]).sum())
        S1.append(e)
    J["checks"]["s1_domain_acc_equals_task_mean_maxdiff"] = dchk
    J["s1_60"] = S1

    # S1 / BTZSC requests (cost + BTZSC item bootstrap)
    def load_req(path, keep_ids):
        out = {}
        for r in read(path):
            if r["id"] in keep_ids:
                out[r["id"]] = r
        return out
    s1ids = set(last_lines(EV / "P_s0" / "s1_60.jsonl"))
    s1req = {}
    for r in read(REQ / "s1_all.jsonl"):
        if r["id"] in s1ids:
            s1req[r["id"]] = compact(r["id"], r["family"], r["state"], r["questions"], r.get("expected") or {}, tok)
    J["checks"]["s1_ids_found"] = {"result_ids": len(s1ids), "found_in_s1_all": len(s1req)}

    btz = {}
    bt_ids = set(last_lines(EV / "P_s0" / "btzsc500.jsonl"))
    for r in read(REQ / "btzsc_full.jsonl"):
        if r["id"] in bt_ids:
            c = compact(r["id"], r["family"], r["state"], r["questions"], r.get("expected") or {}, tok)
            c["ds"] = r["meta"]["ds"]; c["ref"] = r["meta"]["ref"]; c["keys"] = list(r["questions"]["label"]["criteria"])
            btz[r["id"]] = c
    J["checks"]["btzsc_ids_found"] = {"result_ids": len(bt_ids), "found_in_btzsc_full": len(btz)}

    def macro_f1(ref, pred):
        """sklearn f1_score(average='macro', zero_division=0): labels = union of ref and pred (pred -1 = unanswered)."""
        K = int(max(ref.max(), pred.max())) + 2
        r = ref + 1; p_ = pred + 1                      # shift so -1 -> 0
        tp = np.bincount(r[r == p_], minlength=K); rc = np.bincount(r, minlength=K); pc = np.bincount(p_, minlength=K)
        present = (rc > 0) | (pc > 0)
        return float(np.mean(2 * tp[present] / (rc[present] + pc[present])))

    bsc = {f"{arm}_s{s}": json.load(open(D(arm, s) / "btzsc500.score.json")) for arm, s in RUNS}
    dsets = list(next(iter(bsc.values()))["by_dataset"])
    bt_order = sorted(btz)
    preds = np.zeros((len(bt_order), len(RUNS)), int)
    for j, (arm, s) in enumerate(RUNS):
        res = last_lines(D(arm, s) / "btzsc500.jsonl")
        for i, rid in enumerate(bt_order):
            x = res.get(rid)
            preds[i, j] = btz[rid]["keys"].index(x["label"]["choice"]) if x else -1
        del res
    refs = np.array([btz[r]["ref"] for r in bt_order]); dsa = np.array([btz[r]["ds"] for r in bt_order])
    BT = []
    f1chk = 0.0
    rngb = np.random.default_rng(3)
    allB = []
    for dname in dsets:
        m = np.nonzero(dsa == dname)[0]
        f1 = np.array([bsc[f"{arm}_s{s}"]["by_dataset"][dname]["macro_f1"] for arm, s in RUNS])
        mine = np.array([macro_f1(refs[m], preds[m, j]) for j in range(len(RUNS))])
        f1chk = max(f1chk, float(np.abs(mine - f1).max()))
        Bb = min(a.B, 1000)
        fb = np.zeros((Bb, len(RUNS)))
        for b in range(Bb):
            ii = m[rngb.integers(0, len(m), len(m))]
            fb[b] = [macro_f1(refs[ii], preds[ii, j]) for j in range(len(RUNS))]
        allB.append(fb)
        e = {"dataset": dname, "items": int(len(m)), "n_labels": int(len(btz[bt_order[m[0]]]["keys"])),
             "f1": {k: float(v) for k, v in arm_mean(f1).items()}, "d_f1": diffs(f1, fb)}
        BT.append(e)
    f1all = np.array([bsc[f"{arm}_s{s}"]["summary"]["overall_macro_f1"] for arm, s in RUNS])
    BT.append({"dataset": "ALL (mean of 22)", "items": int(len(bt_order)), "f1": {k: float(v) for k, v in arm_mean(f1all).items()},
               "d_f1": diffs(f1all, np.mean(allB, 0))})
    J["checks"]["btzsc_f1_recomputed_maxdiff"] = f1chk
    J["btzsc500"] = BT

    # ============================== cost
    def cost(rs):
        jq = [len(r["qs"]) for r in rs]
        po = [sum(q["n"] for q in r["qs"]) for r in rs]
        jt = [sum(q["joint_tok"] for q in r["qs"]) for r in rs]
        pt = [sum(q["peropt_tok"] for q in r["qs"]) for r in rs]
        return {"requests": len(rs), "joint_seq_per_req": float(np.mean(jq)), "peropt_seq_per_req": float(np.mean(po)),
                "seq_ratio": float(np.sum(po) / np.sum(jq)), "options_per_question": float(np.sum(po) / np.sum(jq)),
                "joint_tokens_per_req": float(np.mean(jt)), "peropt_tokens_per_req": float(np.mean(pt)),
                "token_ratio_total": float(np.sum(pt) / np.sum(jt)), "token_ratio_median_req": float(np.median(np.array(pt) / np.array(jt))),
                "p95_peropt_seq_per_req": float(np.percentile(po, 95))}
    COST = {"pub15": cost(list(reqs.values())), "s1_60": cost(list(s1req.values())), "btzsc500": cost(list(btz.values()))}
    cpb = {}
    for f in fams:
        rs = [r for r in reqs.values() if r["fam"] == f]
        cpb[f] = cost(rs)
    COST["pub15_per_benchmark"] = cpb
    # E6 latency
    e6 = {}
    redo = (EXP / "results_gpu" / "logs" / "e6_redo.log").read_text()
    redo_engines = set(re.findall(r'"engine": "([^"]+)"', redo))
    for name, p in [("gpu_joint_32m", EXP / "results_gpu/e6/crivo-32m.gpu.json"), ("gpu_peropt_32m", EXP / "results_gpu/e6/crivo-32m-peropt.gpu.json"),
                    ("gpu_reranker_32m", EXP / "results_gpu/e6/reranker-32m.gpu.json"),
                    ("cpu16_joint_32m", EXP / "results_cpu/e6cpu/crivo-32m.t16.json"), ("cpu16_peropt_32m", EXP / "results_cpu/e6cpu/crivo-32m-peropt.t16.json"),
                    ("cpu16_reranker_32m", EXP / "results_cpu/e6cpu/reranker-32m.t16.json")]:
        d = json.load(open(p))
        e6[name] = {k: d.get(k) for k in ("engine", "device", "requests", "timed", "median_ms", "p95_ms", "mean_ms", "throughput_req_s",
                                           "options_per_request_mean", "peak_rss_mb", "peak_alloc_mb", "peak_reserved_mb")}
        if d["device"] == "cuda":
            e6[name]["in_redo_log"] = d["engine"] in redo_engines
    for dev in ("gpu", "cpu16"):
        jn, pn = e6[f"{dev}_joint_32m"], e6[f"{dev}_peropt_32m"]
        e6[f"{dev}_ratio_peropt_over_joint"] = {k: pn[k] / jn[k] for k in ("median_ms", "p95_ms", "mean_ms")}
        e6[f"{dev}_ratio_peropt_over_joint"]["throughput_joint_over_peropt"] = jn["throughput_req_s"] / pn["throughput_req_s"]
    COST["e6_latency"] = e6
    # training throughput (indicative; GPU shared with eval lanes; P with --grad_ckpt)
    tr = {}
    for arm in ARMS:
        qs_ = []
        for s in SEEDS:
            p = EXP / "results_gpu2" / "logs" / f"{arm}_s{s}.train.log"
            if not p.exists():
                p = EXP / "results_gpu3" / "logs" / f"{arm}_s{s}.train.log"
            if not p.exists():
                continue
            v = [json.loads(l)["q_per_s"] for l in open(p) if l.startswith('{"step"') and "q_per_s" in l]
            if v:
                qs_.append(v[-1])
        tr[arm] = {"final_q_per_s_by_seed": qs_}
    p1 = EXP / "results_gpu2_discarded/a1a2_try1/logs/P_s0.train.log"
    if p1.exists():
        v = [json.loads(l)["q_per_s"] for l in open(p1) if l.startswith('{"step"') and "q_per_s" in l]
        tr["P_try1_no_gradckpt"] = {"last_q_per_s": v[-1] if v else None, "steps_logged": len(v)}
    COST["training_throughput"] = tr
    J["cost"] = COST

    json.dump(J, open(OUT / "a1_analysis.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_md(J, OUT / "a1_analysis.md")
    print("wrote", OUT / "a1_analysis.json", OUT / "a1_analysis.md")


# ---------------------------------------------------------------- markdown
def write_md(J, path):
    L = []
    P = J["pub15"]; o = P["overall"]
    L += ["# A1 per-benchmark analysis (GPU-2 grid, 32M, 150k rows, 1 epoch)", "",
          "Generated by `a1a2/analyze_a1.py`. Arms: J (joint, 2,112 steps), " + ("Jsteps (joint, ~6,845 steps, step-matched to P), " if REF == "Jsteps" else
          f"{REF} (joint, P's optimizer steps 6,858-6,860 and per-step token budget, ~3.25 passes; compute-matched control), ") +
          "P (per-option, ~6,860 steps); 3 seeds each. CIs: `welch` = Welch 95% over 3+3 seed values; `boot` = paired "
          f"cluster bootstrap (requests resampled within benchmark, B={J['meta']['B']}, per-seed metric then seed mean).",
          "", "## Checks", "",
          f"- pub15 rebuilt from the suite: {o['requests']:,} requests, {o['questions']:,} judged questions, {o['benchmarks']} benchmarks; "
          f"ids equal to every result file: {all(v['equal'] for v in J['checks']['pub15_ids'].values())}; "
          f"max |acc − pub15.score.json| over 9 runs × 37 benchmarks: {max(J['checks']['pub15_max_abs_diff_vs_score_json'].values()):.1e}.",
          f"- BTZSC macro-F1 recomputed from `btzsc_full.jsonl` refs: max |Δ| vs score files {J['checks']['btzsc_f1_recomputed_maxdiff']:.1e}; "
          f"ids found {J['checks']['btzsc_ids_found']}. S1 ids found {J['checks']['s1_ids_found']}; "
          f"S1 domain acc = task mean (max |Δ| {J['checks']['s1_domain_acc_equals_task_mean_maxdiff']:.1e}).",
          "- E4 missing/failed answers per run: " + ", ".join(f"{k} {v['missing_or_failed']}" for k, v in J["checks"]["e4"].items()) + ".",
          "", "## 1. pub15 overall", "",
          "| Metric | J | Jsteps | P | P − Jsteps | P − J |", "|---|---|---|---|---|---|"]
    for lab, k, dk in [("mean skill (clipped, as score_pub)", "mean_skill", "d_mean_skill"),
                       ("mean skill (unclipped)", "mean_skill_unclipped", "d_mean_skill_unclipped"),
                       ("bench-weighted accuracy", "mean_acc_benchweighted", "d_mean_acc_benchweighted")]:
        v = o[k]; d = o[dk]
        L.append(f"| {lab} | {f3(v['J'])} | {f3(v[REF])} | {f3(v['P'])} | "
                 f"{fci(d[PJ]['diff'], d[PJ]['boot_ci95'])} boot; welch [{d[PJ]['welch_ci95'][0]:+.3f}, {d[PJ]['welch_ci95'][1]:+.3f}] | "
                 f"{fci(d['P-J']['diff'], d['P-J']['boot_ci95'])} boot |")
    pq = o["d_pooled_question_acc"]
    L += [f"| pooled question accuracy (paired, seed-mean) | | | | {pq[PJ]['all']:+.3f} (excl. ACOS {pq[PJ]['excl_ACOS']:+.3f}) | "
          f"{pq['P-J']['all']:+.3f} (excl. ACOS {pq['P-J']['excl_ACOS']:+.3f}) |", "",
          f"ACOS holds {pq['P-J']['ACOS_share_of_questions']:.1%} of the {o['questions']:,} judged questions. "
          f"Benchmark × run cells clipped at skill 0: {o['clipped_cells_of_111']} (of 111 per arm).", ""]
    # per benchmark
    L += ["## 2. pub15 per benchmark (seed means; sorted by P − Jsteps skill)", "",
          "q/r = questions per request; opts = mean options per question; tok = median joint-sequence tokens.", "",
          "| Benchmark | q | q/r | opts | noul | tok | chance | acc J / Js / P | skill J / Js / P | Δskill P−Js [boot] | Δacc P−Js [boot] | Δskill P−J [boot] |",
          "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"]
    for e in sorted(P["per_benchmark"], key=lambda e: -e["d_skill"][PJ]["diff"]):
        a_, s_ = e["acc"], e["skill"]
        L.append(f"| {e['benchmark']} | {e['questions']} | {e['q_per_request']:.1f} | {e['mean_options']:.1f} | {e['noul_share']:.0%} | "
                 f"{e['median_joint_tokens']:.0f} | {e['chance']:.3f} | {a_['J']:.3f} / {a_[REF]:.3f} / {a_['P']:.3f} | "
                 f"{s_['J']:.3f} / {s_[REF]:.3f} / {s_['P']:.3f} | {fci(e['d_skill'][PJ]['diff'], e['d_skill'][PJ]['boot_ci95'])} | "
                 f"{fci(e['d_acc'][PJ]['diff'], e['d_acc'][PJ]['boot_ci95'])} | {fci(e['d_skill']['P-J']['diff'], e['d_skill']['P-J']['boot_ci95'])} |")
    L += ["", "Welch-over-seeds CIs per benchmark are in the JSON (`d_skill.*.welch_ci95`).", ""]
    # contributions
    L += ["### 2.1 Why the mean-skill and pooled question-accuracy differences differ", "",
          "Contribution of each benchmark to the mean-skill difference (Δskill/37) and to the pooled question-accuracy "
          "difference (Δacc × share of questions), P − Jsteps, top and bottom 6:", "",
          "| Benchmark | share of questions | Δskill/37 | Δacc × share |", "|---|---:|---:|---:|"]
    N = o["questions"]
    srt = sorted(P["per_benchmark"], key=lambda e: -e["contrib_mean_skill"][PJ])
    for e in srt[:6] + srt[-6:]:
        L.append(f"| {e['benchmark']} | {e['questions'] / N:.1%} | {e['contrib_mean_skill'][PJ]:+.4f} | {e['contrib_pooled_acc'][PJ]:+.4f} |")
    rb = o["robustness"]
    for pr in (PJ, "P-J"):
        r = rb[pr]
        L.append(f"\n{pr}: median per-benchmark Δskill {r['median_bench_d_skill']:+.3f}; {r['benchmarks_positive']} benchmarks up, {r['benchmarks_negative']} down; "
                 f"boot CI above 0 in {r['ci_above_0']}, below 0 in {r['ci_below_0']}; mean Δ without the top 1 / 2 / 4 benchmarks "
                 f"{r['mean_without_top1']:+.3f} / {r['mean_without_top2']:+.3f} / {r['mean_without_top4']:+.3f}.")
    L.append(f"\nFloor benchmarks (every arm skill < 0.03): {', '.join(rb[PJ]['floor_benchmarks_all_arms_skill_lt_0.03'])}.")
    L += ["", "Benchmark-level Spearman ρ between a benchmark feature and the unclipped Δskill (37 benchmarks):", "",
          "| Feature | ρ (P−Jsteps) | p | ρ (P−J) | p |", "|---|---:|---:|---:|---:|"]
    A = P["bench_level_spearman_vs_d_skill_unclipped"]
    for feat in ("mean_options", "q_per_request", "median_joint_tokens", "noul_share", "chance"):
        x, y = A[f"{PJ}|{feat}"], A[f"P-J|{feat}"]
        L.append(f"| {feat} | {x['spearman_rho']:+.2f} | {x['p']:.3f} | {y['spearman_rho']:+.2f} | {y['p']:.3f} |")
    # breakdowns
    L += ["", "## 3. Breakdowns (pub15 questions)", "",
          "Pooled = all questions in the cell (ACOS-dominated where present). bw = benchmark-weighted unclipped skill over "
          "benchmarks with ≥ 20 questions in the cell, with the stratified cluster bootstrap.", ""]
    for name, rows in P["breakdowns"].items():
        L += [f"### {name}", "", "| Bin | questions | benchmarks (≥20) | pooled acc J / Js / P | Δpooled acc P−Js | bw skill J / Js / P | Δbw skill P−Js [boot] | Δbw skill P−J [boot] |",
              "|---|---:|---:|---|---:|---|---|---|"]
        for e in rows:
            pa = e["pooled_acc"]
            if "bw_skill" in e:
                bw = e["bw_skill"]; d1 = e["d_bw_skill"][PJ]; d2 = e["d_bw_skill"]["P-J"]
                bws = f"{bw['J']:.3f} / {bw[REF]:.3f} / {bw['P']:.3f}"; dd1 = fci(d1["diff"], d1["boot_ci95"]); dd2 = fci(d2["diff"], d2["boot_ci95"])
            else:
                bws = dd1 = dd2 = "–"
            L.append(f"| {e['bin']} | {e['questions']:,} | {e['benchmarks']} ({e['benchmarks_ge20']}) | {pa['J']:.3f} / {pa[REF]:.3f} / {pa['P']:.3f} | "
                     f"{e['d_pooled_acc'][PJ]:+.3f} | {bws} | {dd1} | {dd2} |")
        L.append("")
    # position
    POS = J["position"]
    L += ["## 4. Option position", "", "### 4.1 pub15, native order, choice questions with ≥ 3 options (bench-weighted accuracy)", "",
          "| Gold position | questions | J | Jsteps | P | P − Jsteps [boot] |", "|---|---:|---:|---:|---:|---|"]
    for e in POS["pub15_native_order"]:
        b = e["bw_acc"]; d = e["d_bw_acc"][PJ]
        L.append(f"| {e['gold_position']} | {e['questions']:,} | {b['J']:.3f} | {b[REF]:.3f} | {b['P']:.3f} | {fci(d['diff'], d['boot_ci95'])} |")
    cs = POS["pub15_chosen_position_share"]
    L += ["", f"Share of predictions on the first / last option (choice ≥ 3): "
          + "; ".join(f"{arm} {cs['mean'][arm]['first']:.3f} / {cs['mean'][arm]['last']:.3f}" for arm in ARMS)
          + f"; gold {cs['gold']['first']:.3f} / {cs['gold']['last']:.3f} (uniform-first {cs['gold']['uniform_first']:.3f}).", "",
          "### 4.2 E4 (same questions, gold moved by perm1-3 and rev), choice ≥ 3 options, bench-weighted accuracy", "",
          "| Gold position | questions×variants | J | Jsteps | P | P − Jsteps [boot] |", "|---|---:|---:|---:|---:|---|"]
    for e in POS["e4_gold_position_choice3plus"]:
        b = e["bw_acc"]; d = e["d_bw_acc"][PJ]
        L.append(f"| {e['gold_position']} | {e['questions']:,} | {b['J']:.3f} | {b[REF]:.3f} | {b['P']:.3f} | {fci(d['diff'], d['boot_ci95'])} |")
    g = POS["e4_first_minus_last"]
    L += ["", "First − last gap (bench-weighted accuracy, boot CI): " + "; ".join(
        f"{arm} {g[arm]['first_minus_last_bw_acc']:+.3f} [{g[arm]['boot_ci95'][0]:+.3f}, {g[arm]['boot_ci95'][1]:+.3f}]" for arm in ARMS)
        + f" ({g['J']['benchmarks']} benchmarks).", "",
        "### 4.3 E4 binary questions (all six variants), gold first vs second", "",
        "| Gold position | questions×variants | J | Jsteps | P | P − Jsteps [boot] |", "|---|---:|---:|---:|---:|---|"]
    for e in POS["e4_binary_gold_position"]:
        b = e["bw_acc"]; d = e["d_bw_acc"][PJ]
        L.append(f"| {e['gold_position']} | {e['questions']:,} | {b['J']:.3f} | {b[REF]:.3f} | {b['P']:.3f} | {fci(d['diff'], d['boot_ci95'])} |")
    L += ["", "### 4.4 E4 within-question position contrasts (each question against itself; bench-weighted, boot CI)", "",
          "| Contrast | questions | J | Jsteps | P | P − Jsteps |", "|---|---:|---|---|---|---|"]
    for name, w in POS["e4_within_question"].items():
        L.append(f"| {name} | {w['questions']:,} | " + " | ".join(fci(w[arm]['bw_contrast'], w[arm]['boot_ci95']) for arm in ARMS)
                 + f" | {fci(w[PJ]['diff'], w[PJ]['boot_ci95'])} |")
    L += ["", "E4 pooled accuracy per variant: " + "; ".join(
        f"{v} " + "/".join(f"{x[arm]:.3f}" for arm in ARMS) for v, x in POS["e4_pooled_acc_per_variant"].items()) + " (J/Jsteps/P).", ""]
    # S1
    L += ["## 5. S1-Arena s1_60 per domain (task-mean accuracy; CIs: welch over seeds, task bootstrap)", "",
          "| Domain | tasks | acc J / Js / P | Δacc P−Js welch | Δacc P−Js task-boot | tasks P>Js / P<Js | Δacc P−J welch | ΔBrier P−Js welch |",
          "|---|---:|---|---|---|---|---|---|"]
    for e in J["s1_60"]:
        a_ = e["acc"]; d = e["d_acc"][PJ]; d2 = e["d_acc"]["P-J"]; br = e["d_brier"][PJ]
        L.append(f"| {e['domain']} | {e['tasks']} | {a_['J']:.3f} / {a_[REF]:.3f} / {a_['P']:.3f} | {fci(d['diff'], d['welch_ci95'])} | "
                 f"[{d['task_boot_ci95'][0]:+.3f}, {d['task_boot_ci95'][1]:+.3f}] | {d['tasks_x_better']} / {d['tasks_y_better']} | "
                 f"{fci(d2['diff'], d2['welch_ci95'])} | {fci(br['diff'], br['welch_ci95'])} |")
    # BTZSC
    L += ["", "## 6. BTZSC500 per dataset (macro-F1; CIs: welch over seeds, item bootstrap of the seed-mean F1)", "",
          "| Dataset | labels | F1 J / Js / P | ΔF1 P−Js welch | ΔF1 P−Js boot | ΔF1 P−J boot |", "|---|---:|---|---|---|---|"]
    for e in J["btzsc500"]:
        f = e["f1"]; d = e["d_f1"][PJ]; d2 = e["d_f1"]["P-J"]
        L.append(f"| {e['dataset']} | {e.get('n_labels', '')} | {f['J']:.3f} / {f[REF]:.3f} / {f['P']:.3f} | {fci(d['diff'], d['welch_ci95'])} | "
                 f"[{d['boot_ci95'][0]:+.3f}, {d['boot_ci95'][1]:+.3f}] | {fci(d2['diff'], d2['boot_ci95'])} |")
    # cost
    Cc = J["cost"]
    L += ["", "## 7. Inference cost", "", "### 7.1 Forward sequences and tokens per request", "",
          "| Set | requests | joint seq/req | per-option seq/req | ratio | joint tokens/req | per-option tokens/req | token ratio (total) | token ratio (median req) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ("pub15", "s1_60", "btzsc500"):
        c = Cc[k]
        L.append(f"| {k} | {c['requests']:,} | {c['joint_seq_per_req']:.2f} | {c['peropt_seq_per_req']:.2f} | {c['seq_ratio']:.1f}× | "
                 f"{c['joint_tokens_per_req']:.0f} | {c['peropt_tokens_per_req']:.0f} | {c['token_ratio_total']:.1f}× | {c['token_ratio_median_req']:.1f}× |")
    e6 = Cc["e6_latency"]
    L += ["", "### 7.2 E6 latency, 32M (same released weights; one request at a time)", "",
          "| Run | requests | median ms | p95 ms | mean ms | req/s | in redo log |", "|---|---:|---:|---:|---:|---:|---|"]
    for k in ("gpu_joint_32m", "gpu_peropt_32m", "gpu_reranker_32m", "cpu16_joint_32m", "cpu16_peropt_32m", "cpu16_reranker_32m"):
        x = e6[k]
        L.append(f"| {k} | {x['timed']} | {x['median_ms']:.1f} | {x['p95_ms']:.1f} | {x['mean_ms']:.1f} | {x['throughput_req_s']:.2f} | {x.get('in_redo_log', 'n/a (CPU)')} |")
    for dev in ("gpu", "cpu16"):
        r = e6[f"{dev}_ratio_peropt_over_joint"]
        L.append(f"\n{dev}: per-option / joint = median {r['median_ms']:.2f}×, p95 {r['p95_ms']:.1f}×, mean {r['mean_ms']:.1f}×; "
                 f"joint throughput {r['throughput_joint_over_peropt']:.1f}× higher.")
    tr = Cc["training_throughput"]
    L += ["", "Training throughput (final logged q/s; GPU shared with eval lanes, P with `--grad_ckpt`, so indicative only): "
          + "; ".join(f"{k} {v.get('final_q_per_s_by_seed', v.get('last_q_per_s'))}" for k, v in tr.items()), ""]
    L += ["### 7.3 Per-option sequences per request by pub15 benchmark (top 10)", "", "| Benchmark | joint seq/req | per-option seq/req | token ratio |", "|---|---:|---:|---:|"]
    for f, c in sorted(Cc["pub15_per_benchmark"].items(), key=lambda kv: -kv[1]["peropt_seq_per_req"])[:10]:
        L.append(f"| {f} | {c['joint_seq_per_req']:.1f} | {c['peropt_seq_per_req']:.1f} | {c['token_ratio_total']:.1f}× |")
    txt = "\n".join(L) + "\n"
    if REF != "Jsteps":       # column labels and prose written for the default reference arm
        txt = txt.replace("Jsteps", REF).replace(" / Js / ", f" / {REF} / ").replace("P−Js", f"P−{REF}")
        txt = txt.replace("# A1 per-benchmark analysis (GPU-2 grid", f"# A1 per-benchmark analysis, P against {REF} (GPU-2/GPU-3 grid")
    Path(path).write_text(txt)


if __name__ == "__main__":
    main()
