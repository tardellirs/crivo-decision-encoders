"""GPU-1 statistics (E7): paired bootstrap CIs for E1 (BTZSC) and E2 (S1-Arena), E4 flip rates with CIs, the E6
latency table and calibration. CPU only, one process, every result file streamed line by line. B = 2,000, seed 0;
inside one call the same draws serve every system, so all differences are paired. Percentile 95% CIs.

Result files may repeat an id (retries): the LAST line per id wins, as in the scorers. Errors, `unsupported` and
missing answers count as wrong (uniform probabilities for Brier).

    python stats/paired_bootstrap.py integrity --res results_gpu --out stats/gpu1/integrity.json
    python stats/paired_bootstrap.py extract --bench s1 --requests stats/gpu1/req/s1_all.jsonl --data stats/gpu1/ext/s1arena_data \
        --code stats/gpu1/ext/s1arena_code --res results_gpu/s1 --out stats/gpu1/item/s1
    python stats/paired_bootstrap.py extract --bench btzsc --requests stats/gpu1/req/btzsc_full.jsonl --res results_gpu/btzsc --out stats/gpu1/item/btzsc
    python stats/paired_bootstrap.py s1    --res results_gpu/s1 --contam s1arena/contamination.json --items stats/gpu1/item/s1 --out stats/gpu1/s1.json
    python stats/paired_bootstrap.py btzsc --res results_gpu/btzsc --contam btzsc/contamination.json --items stats/gpu1/item/btzsc --out stats/gpu1/btzsc.json
    python stats/paired_bootstrap.py e4    --rows stats/gpu1/req/e4_orig.jsonl --res results_gpu/e4 --out stats/gpu1/e4.json
    python stats/paired_bootstrap.py e6    --gpu results_gpu/e6 --cpu results_cpu/e6cpu --redo_log results_gpu/logs/e6_redo.log --out stats/gpu1/e6.json
    python stats/paired_bootstrap.py md    --dir stats/gpu1                     # writes stats/gpu1/gpu1_stats.md
    python stats/paired_bootstrap.py headline --dir stats/gpu1 --out results_gpu/HEADLINE.md

E1/E2 have two kinds of CI, both written to the same JSON:
  item mode (s1/btzsc with --items: the arrays written by `extract` from the request files with gold and, for S1, the
      parquet gold of the S1-Arena snapshot) is the per-item paired bootstrap: items (requests) resampled with
      replacement inside each task (S1) or dataset (BTZSC), the questions of a request moving together; the statistic is
      then the official macro (mean over tasks of accuracy; mean over datasets of macro-F1). `extract` first recomputes
      every per-task/per-dataset number and aborts if any differs from the official .score.json by > 1e-9.
      Calibration (ECE with 10 equal-mass bins via e5/calib.py, task/dataset-weighted, and multi-class Brier) uses the
      same draws. See item_mode() for the details.
  score mode (always; uses only the per-task / per-dataset numbers in the .score.json files):
      (a) task-level paired bootstrap: tasks resampled with replacement inside each S1 domain (BTZSC: datasets
          resampled, all22 and seen18), the macro recomputed - uncertainty over which tasks the benchmark contains;
      (b) item_normal (superseded by item mode, kept for the record): SE(macro) = sqrt(sum_t se_t^2) / T with
          se_t the scorer's per-task accuracy_se (BTZSC: binomial SE of per-dataset accuracy), SEs added in
          quadrature for a difference (correlation 0, conservative).
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sst

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "e5")); sys.path.insert(0, str(EXP / "e4"))
from calib import ece_equal_mass  # noqa: E402
from perturb import is_yn  # noqa: E402

B, SEED = 2000, 0
LABEL = {"crivo-17m": "Crivo-17M", "crivo-32m": "Crivo-32M v2", "crivo-32m-v1": "Crivo-32M v1", "crivo-68m": "Crivo-68M",
         "crivo-150m": "Crivo-150M v2", "crivo-150m-v1": "Crivo-150M v1", "crivo-400m": "Crivo-400M",
         "crivo-1b": "Crivo-1B", "reranker-32m-raw": "Ettin-32M raw", "nli-deberta-v3-large-zs2": "DeBERTa-v3-large zs-v2",
         "nli-modernbert-large-zs2": "ModernBERT-large zs-v2", "nli-bart-large-mnli": "BART-large-MNLI",
         "letter-qwen3-0.6b": "Qwen3-0.6B letter", "letter-gemma-3-1b": "Gemma-3-1B-it letter", "uniform": "Uniform",
         "letter-qwen3-0.6b-rot4": "Qwen3-0.6B letter rot4 (s1_60)"}
CRIVOS = ["crivo-17m", "crivo-32m", "crivo-32m-v1", "crivo-68m", "crivo-150m", "crivo-150m-v1", "crivo-400m", "crivo-1b"]
BASELINES = ["nli-deberta-v3-large-zs2", "nli-modernbert-large-zs2", "nli-bart-large-mnli", "letter-qwen3-0.6b",
             "letter-gemma-3-1b", "reranker-32m-raw", "uniform"]
SMALL = ["crivo-17m", "crivo-32m", "crivo-32m-v1", "crivo-68m", "crivo-150m", "crivo-150m-v1"]   # Crivo <= 150M
E4_VARIANTS = ["perm1", "perm2", "perm3", "rev", "yn"]


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def order(names):
    rank = {n: i for i, n in enumerate(CRIVOS + BASELINES)}
    return sorted(names, key=lambda n: rank.get(n, 99))


def pairs(names):
    """Each Crivo vs each baseline, plus every Crivo pair (A = the first, diff = A - B)."""
    cr = [n for n in names if n in CRIVOS]; bl = [n for n in names if n in BASELINES]
    return [(a, b) for a in cr for b in bl] + [(a, b) for i, a in enumerate(cr) for b in cr[i + 1:]]


def stream(path):
    with open(path) as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def last_by_id(path, keep):
    """id -> keep(record) for the last line per id."""
    out = {}
    for x in stream(path):
        out[x["id"]] = keep(x)
    return out


RX = re.compile(r'"id": "((?:[^"\\]|\\.)*)".*?"status": "(\w+)"')


def integrity(a):
    """Lines, unique ids, duplicate lines and final (last-line) status per result file; regex on the line head."""
    out = {}
    for p in sorted(Path(a.res).rglob("*.jsonl")):
        if "contaminated" in p.parts:
            continue
        last = {}; n = 0; first_err = Counter()
        with open(p) as f:
            for l in f:
                m = RX.search(l[:4000])
                if not m:
                    continue
                n += 1; last[m.group(1)] = m.group(2)
                if m.group(2) != "ok":
                    first_err[m.group(2)] += 1
        key = str(p.relative_to(a.res))
        out[key] = {"lines": n, "ids": len(last), "repeated_lines": n - len(last), "non_ok_lines_any": dict(first_err),
                    "final_status": dict(Counter(last.values()))}
        print(key, out[key], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


# ----------------------------------------------------------------------------------------------- draws

def strat_counts(strata, n, rng):
    """One stratified bootstrap draw: multiplicity per unit (units resampled with replacement inside each stratum)."""
    m = np.zeros(n)
    for s in strata:
        m[s] += np.bincount(rng.integers(0, len(s), len(s)), minlength=len(s))
    return m


def strat_matrix(strata, n, b=B, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.stack([strat_counts(strata, n, rng) for _ in range(b)])


def strata_of(keys):
    g = defaultdict(list)
    for i, k in enumerate(keys):
        g[k].append(i)
    return [np.array(v) for _, v in sorted(g.items())]


def summarize(point, boot):
    return {"value": float(point), "ci95": ci(boot)}


def paired_block(names, point, boot, extra=None):
    """point: {sys: float}; boot: {sys: array(B)}; extra(a, b) -> dict merged into each pair."""
    out = []
    for x, y in pairs(names):
        d = boot[x] - boot[y]
        r = {"a": x, "b": y, "diff": point[x] - point[y], "ci95": ci(d), "p_boot_two_sided": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
        if extra:
            r.update(extra(x, y))
        r["excludes_0"] = r["ci95"][0] > 0 or r["ci95"][1] < 0
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------------- S1-Arena

def s1_load_scores(res):
    sc = {}
    for p in sorted(Path(res).glob("*.score.json")):
        n = p.name[:-len(".score.json")]
        if n.endswith("rot4"):
            continue                       # s1_60 subset only: not the same items as the others
        sc[n] = json.load(open(p))["tasks"]
    return sc


def s1(a):
    sc = s1_load_scores(a.res)
    names = order(sc)
    contam = json.load(open(a.contam))
    tasks = sorted(next(iter(sc.values())))
    for n in names:
        assert sorted(sc[n]) == tasks and all(sc[n][t]["n_questions"] == sc[names[0]][t]["n_questions"] for t in tasks), n
    dom = [sc[names[0]][t]["domain"] for t in tasks]
    cls = [contam[t]["class"] for t in tasks]
    subsets = {"all": list(range(len(tasks))), "unseen": [i for i, c in enumerate(cls) if c == "unseen"],
               "seen": [i for i, c in enumerate(cls) if c != "unseen"]}
    out = {"method": {}, "n_tasks": {k: len(v) for k, v in subsets.items()}, "systems": [LABEL.get(n, n) for n in names],
           "unanswered_questions": {n: round(sum(v["unanswered_rate"] * v["n_questions"] for v in sc[n].values())) for n in names},
           "score_mode": {}, "item_mode": None}
    out["method"]["score_mode"] = ("task-level paired bootstrap: tasks resampled with replacement inside each of the 13 "
                                   "domains (subset-restricted), macro = mean over resampled tasks; B=2000 seed 0, same draws "
                                   "for all systems. item_normal: SE = sqrt(sum se_t^2)/T from the scorer's accuracy_se "
                                   "(per-question binomial, ddof 1); differences add SEs in quadrature (correlation 0, "
                                   "conservative under positive correlation).")
    for sub, idx in subsets.items():
        idx = np.array(idx)
        M = strat_matrix(strata_of([dom[i] for i in idx]), len(idx))
        res = {}
        nq = np.array([sc[names[0]][tasks[i]]["n_questions"] for i in idx], float)
        for metric in ("accuracy", "accuracy_pooled", "brier", "ece"):
            if metric == "accuracy_pooled":     # question-weighted (the HEADLINE "S1 unseen" column), not the leaderboard macro
                A = {n: np.array([sc[n][tasks[i]]["accuracy"] for i in idx], float) * nq for n in names}
                point = {n: float(A[n].sum() / nq.sum()) for n in names}
                boot = {n: M @ A[n] / (M @ nq) for n in names}
                se = {n: math.sqrt(sum((sc[n][tasks[i]]["accuracy_se"] * sc[n][tasks[i]]["n_questions"]) ** 2 for i in idx)) / nq.sum()
                      for n in names}
                blk = {"own": {n: {**summarize(point[n], boot[n]), "item_normal_ci95": [point[n] - 1.96 * se[n], point[n] + 1.96 * se[n]]}
                               for n in names}}

                def extra(x, y, se=se, point=point):
                    s_ = math.sqrt(se[x] ** 2 + se[y] ** 2); dm = point[x] - point[y]
                    return {"item_normal_ci95": [dm - 1.96 * s_, dm + 1.96 * s_], "item_normal_excludes_0": abs(dm) > 1.96 * s_}
                blk["pairs"] = paired_block(names, point, boot, extra)
                res[metric] = blk
                continue
            A = {n: np.array([sc[n][tasks[i]][metric] for i in idx], float) for n in names}
            point = {n: float(A[n].mean()) for n in names}
            boot = {n: M @ A[n] / M.sum(1) for n in names}
            blk = {"own": {n: summarize(point[n], boot[n]) for n in names}}
            if metric == "accuracy":
                se = {n: math.sqrt(sum(sc[n][tasks[i]]["accuracy_se"] ** 2 for i in idx)) / len(idx) for n in names}
                for n in names:
                    blk["own"][n]["item_normal_ci95"] = [point[n] - 1.96 * se[n], point[n] + 1.96 * se[n]]

                def extra(x, y, A=A, se=se):
                    d = A[x] - A[y]; s = math.sqrt(se[x] ** 2 + se[y] ** 2); dm = float(d.mean())
                    w = sst.wilcoxon(d) if np.any(d != 0) else None
                    return {"item_normal_ci95": [dm - 1.96 * s, dm + 1.96 * s],
                            "item_normal_excludes_0": abs(dm) > 1.96 * s,
                            "tasks_a_better": int((d > 0).sum()), "tasks_b_better": int((d < 0).sum()), "tasks_tied": int((d == 0).sum()),
                            "wilcoxon_p_tasks": float(w.pvalue) if w else 1.0}
            else:
                extra = None
            blk["pairs"] = paired_block(names, point, boot, extra)
            res[metric] = blk
        out["score_mode"][sub] = res
    if a.items:
        cm = np.load(Path(a.items) / "common.npz")
        gi = [str(g) for g in cm["groups"]]
        assert sorted(gi) == tasks, "task sets differ between items and score files"
        isub = {k: [gi.index(tasks[i]) for i in v] for k, v in subsets.items()}
        out["item_mode"] = item_mode("s1", a.items, names, isub)
        out["item_mode"]["check"] = json.load(open(Path(a.items) / "check.json"))
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


# ----------------------------------------------------------------------------------------------- BTZSC

def btzsc(a):
    sc = {}
    for p in sorted(Path(a.res).glob("*.score.json")):
        sc[p.name[:-len(".score.json")]] = json.load(open(p))["by_dataset"]
    names = order(sc)
    contam = json.load(open(a.contam))
    ds = list(next(iter(sc.values())))
    unseen = [d for d in ds if contam[d]["v2"]["class"] == "unseen"]
    subsets = {"all22": list(range(len(ds))), "unseen4": [ds.index(d) for d in unseen],
               "seen18": [i for i, d in enumerate(ds) if d not in unseen]}
    out = {"method": {"score_mode": ("all22 and seen18: datasets resampled with replacement (B=2000, seed 0, paired); unseen4: 4 units are "
                                     "too few to bootstrap, so per-dataset differences are listed. item_normal (accuracy only): "
                                     "binomial SE of each dataset's accuracy, mean over datasets, quadrature for differences "
                                     "(conservative). Macro-F1 per item is not in the score files.")},
           "datasets": ds, "unseen4": unseen, "systems": [LABEL.get(n, n) for n in names],
           "unanswered": {n: sum(v["n"] - v["answered"] for v in sc[n].values()) for n in names}, "score_mode": {}, "item_mode": None}
    for sub, idx in subsets.items():
        idx = np.array(idx)
        res = {}
        for metric in ("macro_f1", "accuracy"):
            A = {n: np.array([sc[n][ds[i]][metric] for i in idx], float) for n in names}
            point = {n: float(A[n].mean()) for n in names}
            se = {n: math.sqrt(sum(sc[n][ds[i]]["accuracy"] * (1 - sc[n][ds[i]]["accuracy"]) / sc[n][ds[i]]["n"] for i in idx)) / len(idx)
                  for n in names}
            blk = {"own": {n: {"value": point[n]} for n in names}, "pairs": []}
            if len(idx) >= 10:                     # all22, seen18 (unseen4: too few units)
                M = strat_matrix([np.arange(len(idx))], len(idx))
                boot = {n: M @ A[n] / M.sum(1) for n in names}
                blk["own"] = {n: summarize(point[n], boot[n]) for n in names}
            for x, y in pairs(names):
                d = A[x] - A[y]
                r = {"a": x, "b": y, "diff": float(d.mean()), "per_dataset": {ds[i]: float(v) for i, v in zip(idx, d)},
                     "datasets_a_better": int((d > 0).sum()), "datasets_b_better": int((d < 0).sum())}
                if len(idx) >= 10:
                    bd = boot[x] - boot[y]; r["ci95"] = ci(bd); r["excludes_0"] = r["ci95"][0] > 0 or r["ci95"][1] < 0
                    w = sst.wilcoxon(d) if np.any(d != 0) else None
                    r["wilcoxon_p_datasets"] = float(w.pvalue) if w else 1.0
                if metric == "accuracy":
                    s = math.sqrt(se[x] ** 2 + se[y] ** 2)
                    r["item_normal_ci95"] = [r["diff"] - 1.96 * s, r["diff"] + 1.96 * s]
                    r["item_normal_excludes_0"] = abs(r["diff"]) > 1.96 * s
                blk["pairs"].append(r)
            if metric == "accuracy":
                for n in names:
                    blk["own"][n]["item_normal_ci95"] = [point[n] - 1.96 * se[n], point[n] + 1.96 * se[n]]
            res[metric] = blk
        out["score_mode"][sub] = res
    if a.items:
        cm = np.load(Path(a.items) / "common.npz")
        gi = [str(g) for g in cm["groups"]]
        isub = {k: [gi.index(ds[i]) for i in v] for k, v in subsets.items()}
        out["item_mode"] = item_mode("btzsc", a.items, names, isub)
        out["item_mode"]["groups"] = gi
        out["item_mode"]["check"] = json.load(open(Path(a.items) / "check.json"))
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


# ----------------------------------------------------------------------------------------------- item mode (E1/E2)
#
# Two steps, so that no big file is ever held in memory:
#   extract  streams the request file (and, for S1, the parquet gold) once, then each result file once, and writes
#            compact per-question arrays to <items>/<bench>/ (common.npz + <system>.npz). Every per-task / per-dataset
#            number is recomputed from the arrays and compared with the .score.json (abort if any differs by > 1e-9).
#   s1/btzsc --items <dir>  then runs the per-item paired bootstrap from those arrays (a few tens of MB).

def load_mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def extract(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = order([p.name[:-len(".score.json")] for p in Path(a.res).glob("*.score.json") if "rot4" not in p.name])
    (extract_s1 if a.bench == "s1" else extract_btzsc)(a, names, out)


def extract_s1(a, names, out):
    """Per question, exactly as s1arena.metrics.score: options = option_ids (boolean -> ["true", "false"], score ->
    "0".."n-1"); gold = schema.Gold (hard label, or argmax of the soft distribution; Brier against the soft target when
    there is one); probabilities through adapter.to_s1_answers and metrics._answer (restrict to options, clip,
    renormalize; answered = some option has p > 0); pred = first option with the max probability; conf = p(pred), or
    1/K when unanswered. Only questions with gold are scored (all of them in this snapshot)."""
    import pyarrow.parquet as pq
    sys.path.insert(0, str(Path(a.code).resolve()))
    from s1arena import metrics
    from s1arena.schema import Gold
    ad = load_mod("s1_adapter", EXP / "s1arena" / "adapter.py")
    gold = {}
    for dom in ad.NON_AGENTIC:
        for f in sorted(Path(a.data, dom).glob("test-*.parquet")):
            t = pq.read_table(f, columns=["id", "gold"]).to_pydict()
            gold.update(zip(t["id"], t["gold"])); del t
    unit_id, unit_task, tasks, qs = [], [], {}, []   # qs: (unit, name, DI question type, options, hard, target)
    exp_mismatch = 0
    for r in stream(a.requests):
        u = len(unit_id); unit_id.append(r["id"]); t = r["meta"]["task"]
        unit_task.append(tasks.setdefault(t, len(tasks)))
        for name, g in json.loads(gold[r["id"]]).items():
            q = r["questions"][name]
            opts = (["true", "false"] if q["type"] == "noul" else
                    [str(i) for i in range(len(q["criteria"]))] if q["type"] == "score" else list(q["criteria"]))
            G = Gold(**g)
            exp_mismatch += r.get("expected", {}).get(name) != g.get("label")
            qs.append((u, name, q, opts, G.hard(), G.target(opts)))
    del gold
    task_names = sorted(tasks, key=tasks.get)
    qu = np.array([x[0] for x in qs]); ug = np.array(unit_task)
    np.savez(out / "common.npz", qu=qu, ug=ug, groups=np.array(task_names), unit_id=np.array(unit_id),
             nopt=np.array([len(x[3]) for x in qs]))
    q_of = defaultdict(list)
    for i, x in enumerate(qs):
        q_of[x[0]].append(i)
    uidx = {u: i for i, u in enumerate(unit_id)}
    unans = [(1 / len(o), sum((1 / len(o) - tg[k]) ** 2 for k in o)) for _, _, _, o, _, tg in qs]
    check = {"expected_vs_parquet_label_mismatch": exp_mismatch, "questions": len(qs), "requests": len(unit_id),
             "max_abs_diff": {}}
    for n in names:
        corr = np.zeros(len(qs)); conf = np.array([c for c, _ in unans]); brier = np.array([b for _, b in unans])
        answered = np.zeros(len(qs), bool)
        for x in stream(Path(a.res) / f"{n}.jsonl"):
            u = uidx.get(x["id"])
            if u is None:
                continue
            ans = (x.get("response") or {}).get("answers") or {} if x.get("status") == "ok" else {}
            for i in q_of[u]:                                           # last line per id wins: reset, then fill
                _, name, q, opts, hard, tg = qs[i]
                corr[i] = 0; conf[i], brier[i] = unans[i]; answered[i] = False
                if name not in ans:
                    continue
                row = {"response": {"answers": {name: ad.to_s1_answers(ans[name], q)}}}
                probs, ok = metrics._answer(row, name, opts)
                if not ok:
                    continue
                pred = max(opts, key=lambda o: probs[o])
                corr[i] = float(pred == hard); conf[i] = probs[pred]; answered[i] = True
                brier[i] = sum((probs[o] - tg[o]) ** 2 for o in opts)
        np.savez(out / f"{n}.npz", correct=corr, conf=conf, brier=brier, answered=answered)
        sc = json.load(open(Path(a.res) / f"{n}.score.json"))["tasks"]
        qg = ug[qu]; d = Counter()
        for g, t in enumerate(task_names):
            s = np.flatnonzero(qg == g); ref = sc[t]; acc_q = corr[s]
            mine = {"n_questions": len(s), "accuracy": float(np.mean(acc_q)),
                    "accuracy_se": float(np.std(acc_q, ddof=1) / math.sqrt(len(s))), "brier": float(np.mean(brier[s])),
                    "ece": metrics.expected_calibration_error(list(conf[s]), list(corr[s])),
                    "unanswered_rate": float((~answered[s]).sum() / len(s))}
            for k, v in mine.items():
                d[k] = max(d[k], abs(v - ref[k]))
        check["max_abs_diff"][n] = dict(d)
        assert max(d.values()) < 1e-9, (n, dict(d))
        print(n, "ok", dict(d), flush=True)
    json.dump(check, open(out / "check.json", "w"), indent=1)


def extract_btzsc(a, names, out):
    """Per item, exactly as btzsc/adapter.py score: pred = index of `choice` in the label keys (-1 when not ok), gold =
    meta.ref (the harness index, gold-0 quirk kept). For calibration: conf = normalized probability of the chosen key
    (uniform 1/K and wrong when not ok), multi-class Brier against the one-hot ref."""
    ad = load_mod("btzsc_adapter", EXP / "btzsc" / "adapter.py")
    unit_id, ds_of, ref, keys, dsi = [], [], [], {}, {}
    for r in stream(a.requests):
        ds = r["meta"]["ds"]; k = list(r["questions"]["label"]["criteria"])
        if ds not in keys:
            keys[ds] = k; dsi[ds] = len(dsi)
        assert keys[ds] == k, ds
        unit_id.append(r["id"]); ds_of.append(dsi[ds]); ref.append(r["meta"]["ref"])
    groups = sorted(dsi, key=dsi.get)
    ug = np.array(ds_of); ref = np.array(ref); U = len(unit_id)
    K = np.array([len(keys[groups[g]]) for g in ug])
    np.savez(out / "common.npz", qu=np.arange(U), ug=ug, groups=np.array(groups), unit_id=np.array(unit_id), ref=ref, nopt=K)
    uidx = {u: i for i, u in enumerate(unit_id)}
    kidx = {ds: {k: j for j, k in enumerate(v)} for ds, v in keys.items()}
    check = {"requests": U, "max_abs_diff": {}, "choice_not_argmax": {}}
    for n in names:
        pred = np.full(U, -1); conf = 1.0 / K; brier = (1.0 / K) ** 2 * (K - 1) + (1 - 1.0 / K) ** 2
        conf = conf.copy(); brier = brier.copy(); nam = 0
        for x in stream(Path(a.res) / f"{n}.jsonl"):
            u = uidx.get(x["id"])
            if u is None:
                continue
            ds = groups[ug[u]]; kk = keys[ds]
            pred[u] = -1; conf[u] = 1 / len(kk); brier[u] = (1 / len(kk)) ** 2 * (len(kk) - 1) + (1 - 1 / len(kk)) ** 2
            if x.get("status") != "ok":
                continue
            a_ = x["response"]["answers"]["label"]
            c = kidx[ds][a_["choice"]]; pred[u] = c
            pr = a_.get("probabilities") or {}
            p = np.array([max(0.0, float(pr.get(k, 0.0) or 0.0)) for k in kk])
            if p.sum() > 0:
                p = p / p.sum(); conf[u] = p[c]; y = np.zeros(len(kk)); y[ref[u]] = 1; brier[u] = float(((p - y) ** 2).sum())
                nam += int(p[c] < p.max())
        corr = (pred == ref).astype(float)
        np.savez(out / f"{n}.npz", correct=corr, conf=conf, brier=brier, pred=pred, answered=pred >= 0)
        sc = json.load(open(Path(a.res) / f"{n}.score.json"))["by_dataset"]
        d = Counter()
        for g, ds in enumerate(groups):
            s = np.flatnonzero(ug == g)
            f1, acc = ad.macro_f1(ref[s].tolist(), pred[s].tolist(), None)
            f1v = f1_weighted(ref[s], pred[s], np.ones(len(s)))
            d["macro_f1"] = max(d["macro_f1"], abs(f1 - sc[ds]["macro_f1"]))
            d["accuracy"] = max(d["accuracy"], abs(acc - sc[ds]["accuracy"]))
            d["vectorized_f1_vs_sklearn"] = max(d["vectorized_f1_vs_sklearn"], abs(f1v - f1))
            d["n"] = max(d["n"], abs(len(s) - sc[ds]["n"]))
        check["max_abs_diff"][n] = dict(d); check["choice_not_argmax"][n] = nam
        assert max(d.values()) < 1e-9, (n, dict(d))
        print(n, "ok", dict(d), "choice != argmax:", nam, flush=True)
    json.dump(check, open(out / "check.json", "w"), indent=1)


def f1_weighted(ref, pred, w):
    """sklearn macro-F1 (zero_division=0) with item multiplicities w: classes = those present in ref or pred."""
    r = ref + 1; p = pred + 1; L = int(max(r.max(), p.max())) + 1
    tp = np.bincount(r[r == p], w[r == p], L); nt = np.bincount(r, w, L); npd = np.bincount(p, w, L)
    pres = (nt > 0) | (npd > 0)
    return float((2 * tp[pres] / (nt[pres] + npd[pres])).mean())


def item_mode(bench, items, names, subsets):
    """Per-item paired bootstrap from the extracted arrays: units (requests) resampled with replacement inside each
    task (S1) or dataset (BTZSC), the questions of a request together; for every draw each system gets
      macro   - the official statistic: mean over the subset's tasks of per-task accuracy (S1), or mean over the
                subset's datasets of per-dataset macro-F1 (BTZSC);
      pooled  - question-weighted accuracy over the subset (S1; BTZSC: mean per-dataset accuracy, `acc_macro`);
      ece10   - ECE, 10 equal-mass bins (e5/calib.py::ece_equal_mass), each task/dataset weighing equally;
      brier   - multi-class Brier, same weighting.
    B=2000, seed 0, the same draws for every system and subset (paired)."""
    d = Path(items)
    cm = np.load(d / "common.npz")
    qu, ug, groups = cm["qu"], cm["ug"], [str(g) for g in cm["groups"]]
    G = len(groups); U = len(ug); qg = ug[qu]
    per = {n: dict(np.load(d / f"{n}.npz")) for n in names}
    corder = {n: np.argsort(per[n]["conf"], kind="stable") for n in names}
    if bench == "btzsc":
        ref = cm["ref"]; L = int(cm["nopt"].max()) + 1                      # slot 0 = the impossible class -1
        cr = qg * L + ref + 1; tot = G * L; slot_g = np.repeat(np.arange(G), L)
        cp = {n: qg * L + per[n]["pred"] + 1 for n in names}
        hit = {n: per[n]["pred"] == ref for n in names}

    def group_stat(n, mq):
        """Per-group official statistic and per-group accuracy under question multiplicities mq."""
        den = np.bincount(qg, mq, G)
        acc = np.bincount(qg, mq * per[n]["correct"], G) / np.maximum(den, 1e-12)
        if bench == "s1":
            return acc, acc
        h = hit[n]
        tp = np.bincount(cr[h], mq[h], tot); nt = np.bincount(cr, mq, tot); npd = np.bincount(cp[n], mq, tot)
        pres = (nt > 0) | (npd > 0)
        f1 = np.where(pres, 2 * tp / np.maximum(nt + npd, 1e-12), 0.0)
        return np.bincount(slot_g, f1, G) / np.maximum(np.bincount(slot_g, pres, G), 1), acc

    def calib(n, mq, gmask):
        mq = mq * gmask[qg]
        den = np.bincount(qg, mq, G); ng = (den > 0).sum()
        w = np.where(den[qg] > 0, mq / np.maximum(den[qg], 1e-12), 0) / ng
        return ece_equal_mass(per[n]["conf"], per[n]["correct"], w, corder[n], 10), float((w * per[n]["brier"]).sum())

    masks = {}
    for sub, idx in subsets.items():
        mk = np.zeros(G); mk[idx] = 1; masks[sub] = mk
    nq_g = np.bincount(qg, minlength=G).astype(float)
    keys = ["macro", "pooled", "ece10", "brier"]

    def stats_for(mq):
        res = {}
        for n in names:
            g, acc = group_stat(n, mq)
            den = np.bincount(qg, mq, G)
            for sub, idx in subsets.items():
                r = res.setdefault(sub, {}).setdefault(n, {})
                r["macro"] = float(g[idx].mean())
                r["pooled"] = (float((acc[idx] * den[idx]).sum() / den[idx].sum()) if bench == "s1" else float(acc[idx].mean()))
                r["ece10"], r["brier"] = calib(n, mq, masks[sub])
        return res

    point = stats_for(np.ones(len(qu)))
    boot = {sub: {n: {k: np.zeros(B) for k in keys} for n in names} for sub in subsets}
    rng = np.random.default_rng(SEED); strata = strata_of(ug.tolist())
    for b in range(B):
        m = strat_counts(strata, U, rng)
        r = stats_for(m[qu])
        for sub in subsets:
            for n in names:
                for k in keys:
                    boot[sub][n][k][b] = r[sub][n][k]
        if b % 200 == 0:
            print("  draw", b, flush=True)
    out = {"method": " ".join(item_mode.__doc__.split()),
           "units": int(U), "questions": int(len(qu)), "metric_names": {"macro": "task-mean accuracy" if bench == "s1" else "dataset-mean macro-F1",
                                                                         "pooled": "question-pooled accuracy" if bench == "s1" else "dataset-mean accuracy"},
           "n_groups": {sub: len(idx) for sub, idx in subsets.items()},
           "questions_in": {sub: int(nq_g[idx].sum()) for sub, idx in subsets.items()}}
    for sub in subsets:
        out[sub] = {k: {"own": {n: summarize(point[sub][n][k], boot[sub][n][k]) for n in names},
                        "pairs": paired_block(names, {n: point[sub][n][k] for n in names}, {n: boot[sub][n][k] for n in names})}
                    for k in keys}
    return out


# ----------------------------------------------------------------------------------------------- E4

def e4(a):
    rows = list(stream(a.rows))
    ids = [r["id"] for r in rows]
    fam = [r["family"] for r in rows]; R = len(rows)
    in_yn = [any(is_yn(q) for q in r["questions"].values()) for r in rows]
    systems = order([p.name for p in Path(a.res).iterdir() if p.is_dir()])
    M = strat_matrix(strata_of(fam), R)
    out = {"method": ("perturb.py flips logic: flip = the chosen key differs from the original order's, counted on the "
                      "questions answered in both orders (perturb.py skips non-ok requests); yn counts only the yes/no "
                      "questions. 'own' = each system's answered set (= perturb.py, checked against the GPU box's "
                      "flips.json); 'common' = the requests every E4 system answered (status ok) in orig and in the variant, "
                      "used for the paired differences. Accuracy vs `expected` (top-1, no accepted sets) over ALL "
                      "questions, unanswered = wrong. Last line per id. CIs: stratified cluster bootstrap, requests "
                      "resampled inside each benchmark, questions of a request together, B=2000 seed 0, same draws for "
                      "every system and variant (paired)."), "requests": R, "systems": {}, "pairs": {}, "calibration_orig": {}}
    ans, probs, errs = {}, {}, {}
    for s in systems:
        ans[s], errs[s] = {}, {}
        for v in ["orig"] + E4_VARIANTS:
            last = last_by_id(Path(a.res) / s / f"{v}.jsonl", lambda x: (x.get("status"), (x.get("response") or {}).get("answers")))
            exp_ids = ids if v != "yn" else [i for i, y in zip(ids, in_yn) if y]
            st = Counter(last[i][0] if i in last else "missing" for i in exp_ids)
            miss_q = sum(len(set(rows[k]["questions"]) - set(last[i][1] or {})) for k, i in enumerate(ids)
                         if i in last and last[i][0] == "ok" and (v != "yn" or in_yn[k]))
            errs[s][v] = {"expected_requests": len(exp_ids), "ok": st.get("ok", 0), "unsupported": st.get("unsupported", 0),
                          "effective_errors": sum(c for k_, c in st.items() if k_ not in ("ok", "unsupported")),
                          "ok_but_missing_questions": miss_q, "extra_ids": len(set(last) - set(exp_ids))}
            ans[s][v] = {i: {k: q["choice"] for k, q in (x[1] or {}).items()} for i, x in last.items() if x[0] == "ok"}
            if v == "orig":
                probs[s] = {i: x[1] for i, x in last.items() if x[0] == "ok"}
            del last
    common = {v: {i for i in ids if all(i in ans[s]["orig"] and i in ans[s][v] for s in systems)} for v in E4_VARIANTS}
    out["common_requests"] = {v: len(c) for v, c in common.items()}
    boot = {}; point = {}
    for s in systems:
        res = {"errors": errs[s]}; bs = {}; ps = {}; own_b = {}
        for v in E4_VARIANTS:
            A = {k: np.zeros(R) for k in ("nq", "nb", "fl", "nc", "flc", "h0", "h1")}
            for k, r in enumerate(rows):
                if v == "yn" and not in_yn[k]:
                    continue
                b0 = ans[s]["orig"].get(r["id"], {}); b1 = ans[s][v].get(r["id"], {}); com = r["id"] in common[v]
                for qk, q in r["questions"].items():
                    if v == "yn" and not is_yn(q):
                        continue
                    g = r.get("expected", {}).get(qk); c0 = b0.get(qk); c1 = b1.get(qk)
                    A["nq"][k] += 1; A["h0"][k] += c0 is not None and c0 == g; A["h1"][k] += c1 is not None and c1 == g
                    if c0 is not None and c1 is not None:
                        A["nb"][k] += 1; A["fl"][k] += c0 != c1
                        if com:
                            A["nc"][k] += 1; A["flc"][k] += c0 != c1
            rt = lambda num, den: (num.sum() / den.sum(), M @ num / (M @ den))
            f, fb = rt(A["fl"], A["nb"]); fc, fcb = rt(A["flc"], A["nc"]); a0, a0b = rt(A["h0"], A["nq"]); a1, a1b = rt(A["h1"], A["nq"])
            res[v] = {"questions": int(A["nq"].sum()), "answered_both": int(A["nb"].sum()), "common_questions": int(A["nc"].sum()),
                      "flip_rate": summarize(f, fb), "flip_rate_common": summarize(fc, fcb),
                      "acc_orig": summarize(a0, a0b), "acc_perturbed": summarize(a1, a1b), "acc_drop": summarize(a0 - a1, a0b - a1b)}
            bs[f"flip_{v}"], ps[f"flip_{v}"] = fcb, fc; own_b[v] = fb
            if v == "perm1":
                bs["acc_orig"], ps["acc_orig"] = a0b, a0
        pm = lambda d_, key: sum(d_[f"{key}_perm{j}"] for j in (1, 2, 3)) / 3
        bs["flip_perm_mean"], ps["flip_perm_mean"] = pm(bs, "flip"), pm(ps, "flip")
        res["flip_perm_mean_common"] = summarize(ps["flip_perm_mean"], bs["flip_perm_mean"])
        res["flip_perm_mean"] = summarize(sum(res[f"perm{j}"]["flip_rate"]["value"] for j in (1, 2, 3)) / 3,
                                          sum(own_b[f"perm{j}"] for j in (1, 2, 3)) / 3)
        same = np.zeros(R); tot = np.zeros(R)
        for k, r in enumerate(rows):
            b0 = ans[s]["orig"].get(r["id"], {})
            for qk in r["questions"]:
                vs = [ans[s][f"perm{j}"].get(r["id"], {}).get(qk) for j in (1, 2, 3)]
                if b0.get(qk) is None or None in vs:
                    continue
                tot[k] += 1; same[k] += all(x == b0.get(qk) for x in vs)
        cons = M @ same / (M @ tot)
        res["consistent_all_perms"] = summarize(same.sum() / tot.sum(), cons)
        bs["consistent"] = cons; ps["consistent"] = same.sum() / tot.sum()
        gpu = Path(a.res) / s / "flips.json"
        if gpu.exists():
            g = json.load(open(gpu))
            res["check_vs_gpu_flips_json"] = {v: abs(g[v]["flip_rate"] - res[v]["flip_rate"]["value"]) for v in E4_VARIANTS if v in g}
            res["check_vs_gpu_flips_json"]["consistent"] = abs(g.get("consistent_all_perms", 0) - res["consistent_all_perms"]["value"])
        out["systems"][s] = res; boot[s] = bs; point[s] = ps
    cal_items = {}
    for s in systems:
        probs_s = probs[s]
        # calibration on the original order (substitute for E2 calibration; see the md)
        cq = []
        for k, r in enumerate(rows):
            x = probs_s.get(r["id"]) or {}
            for qk, q in r["questions"].items():
                keys = list(q["criteria"]); pr = (x.get(qk) or {}).get("probabilities") or {}
                p = np.array([float(pr.get(kk, 0.0)) for kk in keys]); p = p / p.sum() if p.sum() > 0 else np.full(len(keys), 1 / len(keys))
                g = r.get("expected", {}).get(qk); y = np.array([kk == g for kk in keys], float)
                c = (x.get(qk) or {}).get("choice")
                cq.append((k, float(p.max()), float(c is not None and c == g), float(((p - y) ** 2).sum())))
        cal_items[s] = np.array(cq)
    # paired differences
    stats_ = ["flip_perm_mean", "flip_rev", "flip_yn", "acc_orig", "consistent"]
    for st in stats_:
        out["pairs"][st] = paired_block(systems, {s: point[s][st] for s in systems}, {s: boot[s][st] for s in systems})
    # calibration with the same draws: bench weighting (each benchmark 1/34, fields equal inside)
    req_of = cal_items[systems[0]][:, 0].astype(int)
    fq = np.array([fam[k] for k in req_of]); _, finv = np.unique(fq, return_inverse=True); nb = finv.max() + 1
    def weights(m):
        mq = m[req_of]; den = np.bincount(finv, mq, nb); return mq / den[finv] / (den > 0).sum()
    cb = {s: {"ece10": np.zeros(B), "brier": np.zeros(B), "acc": np.zeros(B)} for s in systems}
    one = np.ones(R)
    def cal(s, w):
        c = cal_items[s]; o = np.argsort(c[:, 1], kind="stable")
        return ece_equal_mass(c[:, 1], c[:, 2], w, o, 10), float((w * c[:, 3]).sum()), float((w * c[:, 2]).sum())
    for b in range(B):
        w = weights(M[b])
        for s in systems:
            cb[s]["ece10"][b], cb[s]["brier"][b], cb[s]["acc"][b] = cal(s, w)
    w1 = weights(one)
    cp = {s: dict(zip(("ece10", "brier", "acc"), cal(s, w1))) for s in systems}
    for s in systems:
        out["calibration_orig"][s] = {k: summarize(cp[s][k], cb[s][k]) for k in ("ece10", "brier", "acc")}
        out["calibration_orig"][s]["questions"] = int(len(cal_items[s]))
    out["calibration_pairs"] = {k: paired_block(systems, {s: cp[s][k] for s in systems}, {s: cb[s][k] for s in systems})
                                for k in ("ece10", "brier")}
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


# ----------------------------------------------------------------------------------------------- E6

E6_NAME = {"crivo-32m-peropt": "Crivo-32M v2 per-option (A1 cost)", "letter-qwen06": "Qwen3-0.6B letter",
           "nli-deberta": "DeBERTa-v3-large zs-v2", "reranker-32m": "Ettin-32M raw", "prumo": "Prumo"}


def e6(a):
    redo = []
    for l in open(a.redo_log):
        if l.startswith("{"):
            redo.append(json.loads(l))
    out = {"gpu": [], "cpu": [], "note": ""}
    keep = ("requests", "timed", "unsupported", "median_ms", "p95_ms", "mean_ms", "throughput_req_s", "peak_rss_mb",
            "peak_alloc_mb", "peak_reserved_mb", "load_s", "threads", "questions")
    for p in sorted(Path(a.gpu).glob("*.gpu.json")):
        d = json.load(open(p)); n = p.name[:-len(".gpu.json")]
        in_redo = any(r["engine"] == d["engine"] and abs(r["median_ms"] - d["median_ms"]) < 1e-9 for r in redo)
        out["gpu"].append({"name": n, "label": E6_NAME.get(n, LABEL.get(n, n)), "engine": d["engine"], "valid_redo": in_redo,
                           **{k: d.get(k) for k in keep}})
    for p in sorted(Path(a.cpu).glob("*.json")):
        d = json.load(open(p)); n, t = p.name[:-len(".json")].rsplit(".t", 1)
        out["cpu"].append({"name": n, "label": E6_NAME.get(n, LABEL.get(n, n)), "engine": d["engine"], **{k: d.get(k) for k in keep}})
    out["note"] = ("GPU: A100-SXM4-40GB, 1,000 requests of ~/req/lat1000.jsonl one at a time, 20 warm-up, GPU unshared only "
                   "for files in the E6 redo (logs/e6_redo.log). letter-qwen06 and crivo-32m-peropt come from the first E6 "
                   "pass, measured right after the competing `par` job was killed (05:56:31 UTC; files 05:57:31 / 05:58:43 "
                   "UTC) and were not redone. CPU: VM1 16 vCPU AMD (shared cloud VM), first 300 (t16) / 100 (t1) of the same "
                   "file (Prumo 1,000; DeBERTa 100), 10 warm-up. Unsupported requests (too long / too many options) are not timed, "
                   "which flatters DeBERTa and Qwen (they skip the longest inputs). No per-request times are stored, so "
                   "latency has no CIs. GPU and CPU rows use different request counts (1,000 vs 300/100), so compare "
                   "within a device only.")
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


# ----------------------------------------------------------------------------------------------- markdown

def f3(x):
    return "—" if x is None else f"{x:.3f}"


def fci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def fci0(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


STAR = " \\*"
CHAIN = ["crivo-17m", "crivo-32m", "crivo-68m", "crivo-150m", "crivo-400m", "crivo-1b"]   # v2 family, by size


def get_pair(blk, x, y):
    """Pair x − y from a paired block, flipping (y, x) if that is how it is stored."""
    P = {(r["a"], r["b"]): r for r in blk["pairs"]}
    if (x, y) in P:
        return P[(x, y)]
    r = dict(P[(y, x)]); r["diff"] = -r["diff"]
    for k in ("ci95", "item_normal_ci95"):
        if k in r:
            r[k] = [-r[k][1], -r[k][0]]
    for k1, k2 in (("tasks_a_better", "tasks_b_better"), ("datasets_a_better", "datasets_b_better")):
        if k1 in r:
            r[k1], r[k2] = r[k2], r[k1]
    if "per_dataset" in r:
        r["per_dataset"] = {k: -v for k, v in r["per_dataset"].items()}
    return r


def pair_rows(names):
    """Crivo vs each baseline; consecutive Crivo sizes (larger − smaller) and v2 − v1; Crivo-1B − every other Crivo."""
    cr = [n for n in CRIVOS if n in names]; bl = [n for n in BASELINES if n in names]
    rows = [(a, b) for a in cr for b in bl]
    ch = [n for n in CHAIN if n in names]
    rows += [(ch[i + 1], ch[i]) for i in range(len(ch) - 1)]
    rows += [(v2, v1) for v2, v1 in (("crivo-32m", "crivo-32m-v1"), ("crivo-150m", "crivo-150m-v1")) if v1 in names and v2 in names]
    rows += [("crivo-1b", n) for n in cr if n != "crivo-1b" and ("crivo-1b", n) not in rows]
    return rows


def fv(x, ci_=None):
    return f"{x:.3f}" + (f" {fci0(ci_)}" if ci_ else "")


def fd(r, mark=True):
    return f"{r['diff']:+.3f} {fci(r['ci95'])}" + (STAR if mark and r["excludes_0"] else "")


def md_s1(s1j):
    im = s1j.get("item_mode"); S = s1j["score_mode"]
    L = ["## E2 S1-Arena (official statistic: mean over tasks of per-task accuracy)", ""]
    if im:
        ck = im["check"]
        L += [f"**Item CI** = per-item paired bootstrap: the {im['units']:,} requests ({im['questions']:,} questions) resampled with "
              "replacement inside each task, the questions of a request together, statistic recomputed exactly as the "
              "leaderboard does (mean over tasks), B = 2,000, seed 0, same draws for every system. **Task CI** = tasks "
              "resampled inside the 13 domains (uncertainty over which tasks the benchmark contains; wider). "
              "Before the bootstrap every per-task n, accuracy, accuracy_se, Brier, S1-ECE and unanswered rate was "
              "recomputed from the items and compared with the `.score.json` files: max |diff| = "
              f"{max(max(v.values()) for v in ck['max_abs_diff'].values()):.1g} over {len(ck['max_abs_diff'])} systems "
              f"(gold: parquet `gold`, {ck['expected_vs_parquet_label_mismatch']} mismatches with the request files' `expected`). "
              "Pooled = question-weighted accuracy (not the official statistic). ECE-10 = 10 equal-mass bins "
              "(`e5/calib.py::ece_equal_mass`) with each task weighing equally; Brier = multi-class Brier against S1's "
              "target (soft where the gold is a distribution), task mean (= the scorer's Brier). S1-ECE = S1-Arena's own "
              "per-task ECE, task mean, task CI. Qwen3-0.6B rot4 is excluded (other items).", ""]
    for sub in ("all", "unseen", "seen"):
        A = S[sub]["accuracy"]; I = im[sub] if im else None
        L += [f"### {sub} ({s1j['n_tasks'][sub]} tasks" + (f", {im['questions_in'][sub]:,} questions" if im else "") + "): own scores", "",
              "| System | Acc | item CI | task CI | Pooled acc [item CI] | ECE-10 [item CI] | Brier [item CI] | S1-ECE [task CI] |",
              "|---|---:|---|---|---|---|---|---|"]
        for n, v in A["own"].items():
            ec = S[sub]["ece"]["own"][n]
            if I:
                L.append(f"| {LABEL.get(n, n)} | {v['value']:.3f} | {fci0(I['macro']['own'][n]['ci95'])} | {fci0(v['ci95'])} | "
                         f"{fv(I['pooled']['own'][n]['value'], I['pooled']['own'][n]['ci95'])} | "
                         f"{fv(I['ece10']['own'][n]['value'], I['ece10']['own'][n]['ci95'])} | "
                         f"{fv(I['brier']['own'][n]['value'], I['brier']['own'][n]['ci95'])} | {fv(ec['value'], ec['ci95'])} |")
            else:
                L.append(f"| {LABEL.get(n, n)} | {v['value']:.3f} | — | {fci0(v['ci95'])} | — | — | — | {fv(ec['value'], ec['ci95'])} |")
        if sub == "seen":
            L += ["", f"(seen = the {s1j['n_tasks']['seen']} tasks with any exposure for the v2 family; pairs in `s1.json`.)", ""]
            continue
        L += ["", f"### {sub}: paired differences (A − B)", "",
              "| A | B | Δ acc | item CI | task CI | tasks A>B / A<B | Δ pooled [item CI] | Δ ECE-10 [item CI] | Δ Brier [item CI] |",
              "|---|---|---:|---|---|---|---|---|---|"]
        for x, y in pair_rows(list(A["own"])):
            t = get_pair(A, x, y)
            if I:
                it = get_pair(I["macro"], x, y)
                L.append(f"| {LABEL[x]} | {LABEL[y]} | {t['diff']:+.3f} | {fci(it['ci95'])}{STAR if it['excludes_0'] else ''} | "
                         f"{fci(t['ci95'])}{' †' if t['excludes_0'] else ''} | {t['tasks_a_better']} / {t['tasks_b_better']} | "
                         f"{fd(get_pair(I['pooled'], x, y))} | {fd(get_pair(I['ece10'], x, y))} | {fd(get_pair(I['brier'], x, y))} |")
            else:
                L.append(f"| {LABEL[x]} | {LABEL[y]} | {t['diff']:+.3f} | — | {fci(t['ci95'])}{' †' if t['excludes_0'] else ''} | "
                         f"{t['tasks_a_better']} / {t['tasks_b_better']} | — | — | — |")
        L += ["", "\\* item CI excludes 0; † task CI excludes 0. ECE/Brier: lower is better, so a negative Δ favours A.", ""]
    return L


def md_btzsc(bj):
    im = bj.get("item_mode"); S = bj["score_mode"]
    L = ["## E1 BTZSC (macro-F1 per dataset, averaged over datasets)", "",
         "**Caveat:** the zs-v2 models (DeBERTa-v3-large, ModernBERT-large) were fine-tuned on training splits of all 22 "
         "BTZSC datasets with BTZSC's verbalizers (`btzsc/zsv2_overlap.md`): an upper reference, not a zero-shot baseline. "
         f"Unseen-4 (Crivo v2 family) = {', '.join(bj['unseen4'])}; seen-18 = the other 18.", ""]
    if im:
        ck = im["check"]
        L += [f"**Item CI** = per-item paired bootstrap ({im['units']:,} items resampled with replacement inside each dataset; "
              "macro-F1 recomputed per dataset with the classes present in the resample, as sklearn does; mean over the "
              "subset's datasets), B = 2,000, seed 0, same draws for every system. **Dataset CI** = datasets resampled "
              "(all-22, seen-18; not done for 4 units). Per-dataset macro-F1 and accuracy recomputed from the items equal "
              f"the `.score.json` files (max |diff| {max(max(v.values()) for v in ck['max_abs_diff'].values()):.1g}). "
              "ECE-10 / Brier: confidence = probability of the chosen label, each dataset weighing equally.", ""]
    L += ["| System | F1 all-22 | item CI | dataset CI | F1 unseen-4 [item CI] | F1 seen-18 [item CI] | ECE-10 all-22 [item CI] | Brier all-22 [item CI] | ECE-10 unseen-4 | Brier unseen-4 |",
          "|---|---:|---|---|---|---|---|---|---:|---:|"]
    for n, v in S["all22"]["macro_f1"]["own"].items():
        if im:
            I = im
            L.append(f"| {LABEL.get(n, n)} | {v['value']:.3f} | {fci0(I['all22']['macro']['own'][n]['ci95'])} | {fci0(v['ci95'])} | "
                     f"{fv(I['unseen4']['macro']['own'][n]['value'], I['unseen4']['macro']['own'][n]['ci95'])} | "
                     f"{fv(I['seen18']['macro']['own'][n]['value'], I['seen18']['macro']['own'][n]['ci95'])} | "
                     f"{fv(I['all22']['ece10']['own'][n]['value'], I['all22']['ece10']['own'][n]['ci95'])} | "
                     f"{fv(I['all22']['brier']['own'][n]['value'], I['all22']['brier']['own'][n]['ci95'])} | "
                     f"{I['unseen4']['ece10']['own'][n]['value']:.3f} | {I['unseen4']['brier']['own'][n]['value']:.3f} |")
        else:
            L.append(f"| {LABEL.get(n, n)} | {v['value']:.3f} | — | {fci0(v['ci95'])} | {S['unseen4']['macro_f1']['own'][n]['value']:.3f} | "
                     f"{S['seen18']['macro_f1']['own'][n]['value']:.3f} | — | — | — | — |")
    L += ["", "Paired differences (A − B), macro-F1:", "",
          "| A | B | Δ all-22 [item CI] | dataset CI | datasets A>B / A<B | Δ unseen-4 [item CI] | per dataset (unseen-4) | Δ seen-18 [item CI] | seen-18 dataset CI | Δ ECE-10 all-22 [item CI] | Δ Brier all-22 [item CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x, y in pair_rows(list(S["all22"]["macro_f1"]["own"])):
        t = get_pair(S["all22"]["macro_f1"], x, y); u = get_pair(S["unseen4"]["macro_f1"], x, y); s8 = get_pair(S["seen18"]["macro_f1"], x, y)
        per = " ".join(f"{v:+.2f}" for v in u["per_dataset"].values())
        if im:
            g = lambda sub, k: fd(get_pair(im[sub][k], x, y))
            L.append(f"| {LABEL[x]} | {LABEL[y]} | {g('all22', 'macro')} | {fci(t['ci95'])}{' †' if t['excludes_0'] else ''} | "
                     f"{t['datasets_a_better']} / {t['datasets_b_better']} | {g('unseen4', 'macro')} | {per} | {g('seen18', 'macro')} | "
                     f"{fci(s8['ci95'])}{' †' if s8['excludes_0'] else ''} | {g('all22', 'ece10')} | {g('all22', 'brier')} |")
        else:
            L.append(f"| {LABEL[x]} | {LABEL[y]} | {t['diff']:+.3f} | {fci(t['ci95'])} | {t['datasets_a_better']} / {t['datasets_b_better']} | "
                     f"{u['diff']:+.3f} | {per} | {s8['diff']:+.3f} | {fci(s8['ci95'])} | — | — |")
    L += ["", "\\* item CI excludes 0; † dataset CI excludes 0. Unseen-4 per-dataset order: " + ", ".join(bj["unseen4"]) + ".", ""]
    return L


def md(a):
    d = Path(a.dir)
    L = ["# GPU-1 statistics (E7)", "", "Generated by `stats/paired_bootstrap.py md` from the JSON files in this folder. "
         "B = 2,000, seed 0, percentile 95% CIs; differences are A − B on identical items with the same draws.", ""]
    integ = json.load(open(d / "integrity.json")) if (d / "integrity.json").exists() else {}
    if integ:
        L += ["## Result-file integrity (last line per id)", "", "| File | Lines | Ids | Repeated lines | Final status |", "|---|---:|---:|---:|---|"]
        for k, v in integ.items():
            if v["repeated_lines"] or set(v["final_status"]) != {"ok"}:
                L.append(f"| {k} | {v['lines']} | {v['ids']} | {v['repeated_lines']} | {v['final_status']} |")
        L += ["", "Files not listed: every id once, every final status `ok`. `unsupported` = the engine refuses the input "
              "(too long, too many options); it is counted wrong by the scorers.", ""]
    s1j = json.load(open(d / "s1.json")); bj = json.load(open(d / "btzsc.json"))
    L += md_s1(s1j) + md_btzsc(bj)
    ej = json.load(open(d / "e4.json"))
    L += ["## E4 robustness (flip rate = share of questions whose choice changes vs the original order)", "",
          "Own answered set (perturb.py logic; unsupported requests skipped). Accuracy over all questions, unsupported = wrong.", "",
          "| System | errors / unsupported (orig) | perm1 | perm2 | perm3 | perm mean | rev | yn | consistent (3 perms) | acc orig |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in ej["systems"].items():
        ne = sum(x["effective_errors"] + x["ok_but_missing_questions"] for x in v["errors"].values())
        c = lambda x: f"{x['value']:.3f} {fci0(x['ci95'])}"
        L.append(f"| {LABEL.get(s, s)} | {ne} / {v['errors']['orig']['unsupported']} | {c(v['perm1']['flip_rate'])} | {c(v['perm2']['flip_rate'])} | "
                 f"{c(v['perm3']['flip_rate'])} | {c(v['flip_perm_mean'])} | {c(v['rev']['flip_rate'])} | {c(v['yn']['flip_rate'])} | "
                 f"{c(v['consistent_all_perms'])} | {c(v['perm1']['acc_orig'])} |")
    s0 = ej["systems"][next(iter(ej["systems"]))]
    chk = max(max(v.get("check_vs_gpu_flips_json", {0: 0}).values()) for v in ej["systems"].values())
    L += ["", f"Questions: perm/rev {s0['perm1']['questions']}, yn {s0['yn']['questions']} (in {s0['errors']['yn']['expected_requests']} requests). "
          f"Common set (requests every system answered): {ej['common_requests']}. "
          f"Max |our own-set flip rate − GPU-box flips.json|: {chk:.2g}. 'acc orig' here is pooled over questions "
          "(unsupported = wrong); the calibration table below weighs each benchmark equally, hence different values. "
          "Crivo-32M v1 / 150M v1 and BART-MNLI were not run on E4.", "",
          "Paired differences on the common set (flip rates) and on all questions (accuracy):", "",
          "| A | B | Δ flip perm-mean [CI] | Δ flip rev [CI] | Δ flip yn [CI] | Δ acc orig [CI] |", "|---|---|---|---|---|---|"]
    P = {st: {(r["a"], r["b"]): r for r in ej["pairs"][st]} for st in ej["pairs"]}
    for k, r in P["flip_perm_mean"].items():
        if r["b"] in BASELINES or "crivo-1b" in k:
            g = lambda st: f"{P[st][k]['diff']:+.3f} {fci(P[st][k]['ci95'])}"
            L.append(f"| {LABEL[r['a']]} | {LABEL[r['b']]} | {g('flip_perm_mean')} | {g('flip_rev')} | {g('flip_yn')} | {g('acc_orig')} |")
    L += ["", "### Calibration on the E4 original-order items (public 0.3 suite; complements the E1/E2 calibration above)", "",
          "Public 0.3 suite subsample (2,833 requests, 34 benchmarks, all `choice`); each benchmark weighs 1/34. "
          "ECE with 10 equal-mass bins (`e5/calib.py::ece_equal_mass`), multi-class Brier; correctness = top-1 == `expected`.", "",
          "| System | Acc [CI] | ECE-10 eq-mass [CI] | Brier [CI] |", "|---|---|---|---|"]
    for s, v in ej["calibration_orig"].items():
        L.append(f"| {LABEL.get(s, s)} | {v['acc']['value']:.3f} {fci0(v['acc']['ci95'])} | {v['ece10']['value']:.3f} {fci0(v['ece10']['ci95'])} | "
                 f"{v['brier']['value']:.3f} {fci0(v['brier']['ci95'])} |")
    e6j = json.load(open(d / "e6.json"))
    L += ["", "## E6 latency and memory", "", "GPU (A100-SXM4-40GB, one request at a time, 1,000 requests):", "",
          "| System | Valid (redo) | Timed / unsupported | Median ms | p95 ms | Mean ms | Req/s | Peak alloc MB | Peak reserved MB | Peak RSS MB |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(e6j["gpu"], key=lambda r: (not r["valid_redo"], r["median_ms"])):
        L.append(f"| {r['label']} | {'yes' if r['valid_redo'] else 'no (first pass)'} | {r['timed']} / {r['unsupported']} | {r['median_ms']:.1f} | "
                 f"{r['p95_ms']:.1f} | {r['mean_ms']:.1f} | {r['throughput_req_s']:.1f} | {r['peak_alloc_mb']:.0f} | {r['peak_reserved_mb']:.0f} | {r['peak_rss_mb']:.0f} |")
    L += ["", "CPU (VM1, 16 vCPU AMD, shared):", "", "| System | Threads | Timed / unsupported | Median ms | p95 ms | Mean ms | Req/s | Peak RSS MB |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(e6j["cpu"], key=lambda r: (r["threads"], r["median_ms"])):
        L.append(f"| {r['label']} | {r['threads']} | {r['timed']} / {r['unsupported']} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | "
                 f"{r['mean_ms']:.1f} | {r['throughput_req_s']:.2f} | {r['peak_rss_mb']:.0f} |")
    L += ["", e6j["note"], ""]
    (d / "gpu1_stats.md").write_text("\n".join(L) + "\n")
    print("wrote", d / "gpu1_stats.md")


def headline(a):
    """results_gpu/HEADLINE.md from s1.json / btzsc.json (score files; item-mode CIs when present)."""
    d = Path(a.dir); s1j = json.load(open(d / "s1.json")); bj = json.load(open(d / "btzsc.json"))
    S = s1j["score_mode"]; Bz = bj["score_mode"]; Si = s1j.get("item_mode") or {}; Bi = bj.get("item_mode") or {}
    names = order(set(s1j["score_mode"]["all"]["accuracy"]["own"]) | set(Bz["all22"]["macro_f1"]["own"]))

    def v(blk, n, item=None):
        if n not in blk["own"]:
            return "—"
        x = f"{blk['own'][n]['value']:.3f}"
        if item and n in item["own"]:
            c = item["own"][n]["ci95"]; x += f" [{c[0]:.3f}, {c[1]:.3f}]"
        return x
    it = lambda D, sub, k: (D.get(sub) or {}).get(k)
    L = ["# GPU-1 headline (our runs; S1-Arena accuracy, BTZSC macro-F1)", "",
         f"Generated by `stats/paired_bootstrap.py headline` from `stats/gpu1/{{s1,btzsc}}.json`. Brackets: 95% per-item "
         "paired-bootstrap CI (B=2000, seed 0; items resampled inside each task/dataset). Task- and dataset-resampling CIs "
         "and all paired differences: `stats/gpu1/gpu1_stats.md`.", "",
         "- **S1 all / S1 unseen = the official S1-Arena statistic: mean over tasks of per-task accuracy** (leaderboard "
         f"definition). Unseen = the {s1j['n_tasks']['unseen']} tasks unseen by the Crivo v2 family (`s1arena/contamination.json`; "
         "approximate for the v1 models).",
         "- *S1 unseen pooled* = question-weighted accuracy over the unseen tasks (Σ correct / Σ questions). It is **not** the "
         "official statistic; an earlier version of this file showed it under the label \"S1 unseen\". The few large unseen "
         "tasks dominate it, and there DeBERTa-zs-v2 is strong.",
         f"- BTZSC: macro-F1 per dataset, mean over datasets. Unseen-4 (v2 family) = {', '.join(bj['unseen4'])}; seen-18 = the rest.", "",
         "| System | S1 all | S1 unseen | S1 unseen pooled | BTZSC all-22 | BTZSC unseen-4 | BTZSC seen-18 |",
         "|---|---|---|---:|---|---|---|"]
    for n in names:
        L.append(f"| {LABEL.get(n, n)} | {v(S['all']['accuracy'], n, it(Si, 'all', 'macro'))} | "
                 f"{v(S['unseen']['accuracy'], n, it(Si, 'unseen', 'macro'))} | {v(S['unseen']['accuracy_pooled'], n)} | "
                 f"{v(Bz['all22']['macro_f1'], n, it(Bi, 'all22', 'macro'))} | {v(Bz['unseen4']['macro_f1'], n, it(Bi, 'unseen4', 'macro'))} | "
                 f"{v(Bz['seen18']['macro_f1'], n, it(Bi, 'seen18', 'macro'))} |")
    L += ["", "Published S1-Arena (whole benchmark): Decision-2.0-Kai-0.6B 0.503, Laya 421M 0.445, Qwen3-0.6B 0.392.", "",
          "**BTZSC caveat (zs-v2 rows).** The MoritzLaurer `zeroshot-v2.0` checkpoints we ran (DeBERTa-v3-large, "
          "ModernBERT-large; non-commercial variants) were fine-tuned on training splits of **all 22 BTZSC datasets** (up "
          "to 500 examples per class) with BTZSC's exact label verbalizers, and a few BTZSC test items are in their "
          "training data (yelpreviews 4.5%, agnews 1.7%; trueteacher shares the source article for 6.1%). Their BTZSC "
          "scores are therefore supervised few-shot, not zero-shot: report them as an **upper reference, not a "
          "like-for-like zero-shot baseline**. On BTZSC unseen-4 the exposure is asymmetric in their favour (unseen by "
          "Crivo v2, seen by zs-v2). Evidence: `btzsc/zsv2_overlap.md`. Overlap between zs-v2's training datasets "
          "and S1-Arena tasks was not audited.", ""]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["integrity", "extract", "s1", "btzsc", "e4", "e6", "md", "headline"])
    for k in ("res", "contam", "out", "requests", "rows", "gpu", "cpu", "redo_log", "dir", "items", "bench", "data", "code"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args()
    {"integrity": integrity, "extract": extract, "s1": s1, "btzsc": btzsc, "e4": e4, "e6": e6, "md": md,
     "headline": headline}[a.cmd](a)


if __name__ == "__main__":
    main()
