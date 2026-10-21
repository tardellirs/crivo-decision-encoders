"""Released Crivo models on the A1/A2 grid's evaluation subsets (pub15, s1_60, btzsc500), with the official scorers.

    python3 -I a1a2/released_on_subsets.py      # from the repository root; needs KIT, RUNS_DIR and the request/ext files [--work <scratch dir>] [--models crivo-32m,crivo-68m]

CPU only, no model is run: the released models' existing outputs are restricted to the subset ids.
  * S1-Arena / BTZSC: GPU-1 outputs results_gpu/{s1,btzsc}/<model>.jsonl (last line per id), restricted to the ids of
    results_gpu2/ev/P_s0/{s1_60,btzsc500}.jsonl. Request files = the rows of stats/gpu1/req/{s1_all,btzsc_full}.jsonl
    with those ids (same adapters as a1a2/build_eval.py). Scored with `s1arena/adapter.py score` (data and code in
    stats/gpu1/ext/) and `btzsc/adapter.py score`.
  * pub15: request file rebuilt with a1a2/build_eval.py:pub15 from $KIT/suite-0.3; the released model's complete
    public-suite run ($RUNS_DIR/<run>/results.jsonl.gz; HF dataset tardellirs/decision-index-results) restricted to the
    pub15 run_ids. 525 pub15 requests (ACOS, BRIGHT, ToolRet, Home appliance) are absent from the released runs, so
    pub15 is compared on the common requests ("pub15c"), with every system (released and grid arms) re-scored by
    a1a2/score_pub.py on that same request file.
  * Check: the arm P seed 0 outputs are re-scored with the same request files and must reproduce its own score files.
Outputs: results_gpu3/analysis/released_subsets.{json,md} and the per-model score files in
results_gpu3/analysis/released_subsets/.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
_env = __import__("os").environ
KIT = Path(_env.get("KIT", str(Path.home() / "decision-index")))
RUNS_DIR = Path(_env.get("RUNS_DIR", str(Path.home() / "decision-index-results" / "runs")))
SEEDS_JSON = Path(_env.get("OUT_DIR", str(EXP / "build" / "tables"))) / "a1a2_seeds.json"   # written by paper_artifacts/make_tables.py
PY = sys.executable
RUNS = {"crivo-17m": "crivo-17m", "crivo-32m": "crivo-32m-v2", "crivo-68m": "crivo-68m", "crivo-150m": "crivo-150m-v2"}
U4 = ["appreviews", "empathetic", "trueteacher", "capsotu"]


def last_lines(p, key="id", keep=None):
    """id -> raw line (key "id") or parsed row; last line per id wins; only ids in `keep` when given (streamed)."""
    out = {}
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                if keep is None or r[key] in keep:
                    out[r[key]] = l if key == "id" else r
    return out


def ids_of(p):
    return set(last_lines(p))


def subset_requests(src, ids, out):
    n = 0
    with open(src) as f, open(out, "w") as g:
        for l in f:
            if json.loads(l)["id"] in ids:
                g.write(l); n += 1
    return n


def subset_results(src, ids, out):
    res = last_lines(src, keep=ids)
    with open(out, "w") as g:
        for i in ids:
            if i in res:
                g.write(res[i])
    return sum(i in res for i in ids)


def run(args):
    subprocess.run([PY, "-I", *map(str, args)], check=True, stdout=subprocess.DEVNULL)


def s1_metrics(p):
    s = json.load(open(p))
    s1c = json.load(open(EXP / "s1arena" / "contamination.json"))
    un = [v["accuracy"] for t, v in s["tasks"].items() if s1c.get(t, {}).get("class") == "unseen"]
    return {"s1_acc": s["summary"]["accuracy"], "s1_unseen_acc": sum(un) / len(un), "s1_brier": s["summary"]["brier"],
            "s1_tasks": s["summary"]["tasks"], "s1_unseen_tasks": len(un)}


def bt_metrics(p):
    b = json.load(open(p))
    u = [b["by_dataset"][d]["macro_f1"] for d in U4]
    return {"btzsc_f1": b["summary"]["overall_macro_f1"], "btzsc_unseen_f1": sum(u) / len(u)}


def pub_metrics(p):
    s = json.load(open(p))["summary"]
    return {"pub15_skill": s["mean_skill"], "pub15_ece": s["ece_choice"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="scratch dir for the subset request/result files")
    ap.add_argument("--models", default="crivo-32m,crivo-68m")
    a = ap.parse_args()
    W = Path(a.work); W.mkdir(parents=True, exist_ok=True)
    OUT = EXP / "results_gpu3" / "analysis"; SC = OUT / "released_subsets"; SC.mkdir(parents=True, exist_ok=True)
    P0 = EXP / "results_gpu2" / "ev" / "P_s0"
    ext = EXP / "stats" / "gpu1" / "ext"
    J = {"models": {}, "checks": {}}

    # ---- request subsets
    s1ids, btids = ids_of(P0 / "s1_60.jsonl"), ids_of(P0 / "btzsc500.jsonl")
    J["checks"]["s1_60_requests"] = subset_requests(EXP / "stats/gpu1/req/s1_all.jsonl", s1ids, W / "s1_60.jsonl")
    J["checks"]["btzsc500_requests"] = subset_requests(EXP / "stats/gpu1/req/btzsc_full.jsonl", btids, W / "btzsc500.jsonl")
    spec = importlib.util.spec_from_file_location("build_eval", EXP / "a1a2" / "build_eval.py")
    be = importlib.util.module_from_spec(spec); spec.loader.exec_module(be)
    be.pub15(KIT / "suite-0.3", W / "pub15.jsonl")
    pubids = ids_of(W / "pub15.jsonl")
    J["checks"]["pub15_requests"] = len(pubids)
    J["checks"]["pub15_ids_equal_P_s0"] = pubids == ids_of(P0 / "pub15.jsonl")

    def s1score(res, out):
        run([EXP / "s1arena/adapter.py", "score", "--data", ext / "s1arena_data", "--code", ext / "s1arena_code",
             "--requests", W / "s1_60.jsonl", "--results", res, "--out", out])

    def btscore(res, out):
        run([EXP / "btzsc/adapter.py", "score", "--requests", W / "btzsc500.jsonl", "--results", res, "--out", out])

    # ---- check: P seed 0 re-scored with these request files equals its own score files
    s1score(P0 / "s1_60.jsonl", W / "P_s0.s1.json"); btscore(P0 / "btzsc500.jsonl", W / "P_s0.bt.json")
    run([EXP / "a1a2/score_pub.py", "--requests", W / "pub15.jsonl", "--results", P0 / "pub15.jsonl", "--out", W / "P_s0.pub.json"])
    mine = {**s1_metrics(W / "P_s0.s1.json"), **bt_metrics(W / "P_s0.bt.json"), **pub_metrics(W / "P_s0.pub.json")}
    own = {**s1_metrics(P0 / "s1_60.score.json"), **bt_metrics(P0 / "btzsc500.score.json"), **pub_metrics(P0 / "pub15.score.json")}
    J["checks"]["P_s0_rescore_max_abs_diff"] = max(abs(mine[k] - own[k]) for k in mine)

    # ---- released models
    for m in a.models.split(","):
        e = {}
        n1 = subset_results(EXP / "results_gpu/s1" / f"{m}.jsonl", s1ids, W / f"{m}.s1_60.res.jsonl")
        s1score(W / f"{m}.s1_60.res.jsonl", SC / f"{m}.s1_60.score.json")
        n2 = subset_results(EXP / "results_gpu/btzsc" / f"{m}.jsonl", btids, W / f"{m}.btzsc500.res.jsonl")
        btscore(W / f"{m}.btzsc500.res.jsonl", SC / f"{m}.btzsc500.score.json")
        e.update(s1_metrics(SC / f"{m}.s1_60.score.json")); e.update(bt_metrics(SC / f"{m}.btzsc500.score.json"))
        e["s1_results_found"], e["btzsc_results_found"] = n1, n2
        # full-set values from the GPU-1 score files, for the fidelity of the restriction
        e["full_s1_acc"] = json.load(open(EXP / "results_gpu/s1" / f"{m}.score.json"))["summary"]["accuracy"]
        e["full_btzsc_f1"] = json.load(open(EXP / "results_gpu/btzsc" / f"{m}.score.json"))["summary"]["overall_macro_f1"]
        J["models"][m] = e

    # ---- pub15: the released public-suite runs use the same kit 0.3 run_ids, but for four benchmarks (ACOS, BRIGHT,
    # ToolRet, Home appliance) some pub15 requests are absent from them (different request grouping in the released
    # suite files). Compare on the common requests only: ids present in every released run with the same question
    # keys; every system (released models and grid arms) is scored with score_pub.py on that same request file.
    req_keys = {}
    for l in open(W / "pub15.jsonl"):
        r = json.loads(l); req_keys[r["id"]] = sorted(r["questions"])
    common = set(pubids)
    for m in a.models.split(","):
        rp = RUNS_DIR / RUNS[m] / "results.jsonl.gz"
        rel = last_lines(rp, key="run_id", keep=pubids)
        ok = {i for i, r in rel.items() if r.get("status") == "ok" and sorted(r["response"]["answers"]) == req_keys[i]}
        common &= ok
        with open(W / f"{m}.pub15.res.jsonl", "w") as g:
            for i in sorted(ok):
                r = rel[i]
                g.write(json.dumps({"id": i, "status": r["status"], "response": r["response"]}) + "\n")
        J["models"][m]["pub15_run"] = f"runs/{RUNS[m]}/results.jsonl.gz"; J["models"][m]["pub15_ok_same_keys"] = len(ok)
        del rel
    fam = {}
    with open(W / "pub15.jsonl") as f, open(W / "pub15_common.jsonl", "w") as g:
        for l in f:
            r = json.loads(l)
            fam.setdefault(r["family"], [0, 0]); fam[r["family"]][0] += 1
            if r["id"] in common:
                g.write(l); fam[r["family"]][1] += 1
    J["checks"]["pub15_common_requests"] = len(common)
    J["checks"]["pub15_common_reduced_benchmarks"] = {k: v for k, v in fam.items() if v[0] != v[1]}

    def pubc(res, out):
        run([EXP / "a1a2/score_pub.py", "--requests", W / "pub15_common.jsonl", "--results", res, "--out", out])
    for m in a.models.split(","):
        pubc(W / f"{m}.pub15.res.jsonl", SC / f"{m}.pub15_common.score.json")
        J["models"][m]["pub15c_skill"] = pub_metrics(SC / f"{m}.pub15_common.score.json")["pub15_skill"]
    arms_c = {}
    for arm, d, ns in (("P", EXP / "results_gpu2/ev", 3), ("Jcomp", EXP / "results_gpu3/ev", 3), ("Praw", EXP / "results_gpu3/ev", 1)):
        v = []
        for k in range(ns):
            pubc(d / f"{arm}_s{k}" / "pub15.jsonl", SC / f"{arm}_s{k}.pub15_common.score.json")
            v.append(pub_metrics(SC / f"{arm}_s{k}.pub15_common.score.json")["pub15_skill"])
        arms_c[arm] = v
    J["arms_pub15c_skill"] = arms_c
    for m, e in J["models"].items():
        print(m, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()}), flush=True)
    print("arms pub15c", arms_c, J["checks"], flush=True)

    # ---- grid arms (seed values from their score files) for the comparison table
    seeds = json.load(open(SEEDS_JSON))
    J["arms"] = {arm: seeds[arm] for arm in ("P", "Jcomp", "Praw") if arm in seeds}
    json.dump(J, open(OUT / "released_subsets.json", "w"), indent=1)
    for arm, v in J["arms"].items():
        for s_, x in zip(v, J["arms_pub15c_skill"][arm]):
            s_["pub15c_skill"] = x
    keys = ["pub15c_skill", "s1_acc", "s1_unseen_acc", "btzsc_f1", "btzsc_unseen_f1"]
    L = ["# Released Crivo models on the grid's evaluation subsets (official scorers)", "",
         "Generated by `a1a2/released_on_subsets.py`. Checks: " + json.dumps(J["checks"]), "",
         "| System | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for arm, v in J["arms"].items():
        L.append(f"| arm {arm} (seeds) | " + " | ".join(" / ".join(f"{s[k]:.3f}" for s in v) for k in keys) + " |")
    for m, e in J["models"].items():
        L.append(f"| {m} (released) | " + " | ".join(f"{e[k]:.3f}" if k in e else "–" for k in keys) + " |")
    L += ["", "Full-set values of the released models (GPU-1 score files): " + "; ".join(
        f"{m} S1 {e['full_s1_acc']:.3f}, BTZSC {e['full_btzsc_f1']:.3f}" for m, e in J["models"].items())]
    (OUT / "released_subsets.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "released_subsets.json")


if __name__ == "__main__":
    main()
