"""E5 / error analysis, step 1: per-question table from official-kit public-suite runs (Decision Index 0.3).

Reads the frozen 0.3 suite through the kit (exclusions and edition subsets applied, exactly as `score` does) and the
`results.jsonl.gz` of each run. Writes, to --out:

  questions.parquet   one row per scored field (model-independent): run_id, key, bench, area, track, qtype, k,
                      gold_idx, n_accepted, yesno, state_tokens (Ettin tokenizer on Crivo's render(state)),
                      instr_tokens, opt_tokens, input_tokens (whole Crivo encoder sequence for the question),
                      proxy_tokens (kit: len(payload json)//4), nq (questions in the request), in_index
                      (ToolRet, BRIGHT and RouterBench are left out: no per-field gold)
  preds_<model>.npz   arrays aligned with questions.parquet: conf (top-label probability), correct (pred in accepted,
                      the kit's per-field rule), pacc (probability on the accepted set), brier (multi-class, accepted
                      set merged into one class), nll (-log pacc, clipped at 1e-12), pred_idx, uniform (all options
                      equal: Prumo's "not claimed" output)

Gold and "correct" follow the kit: choice -> a["choice"] in accepted (accepted = scoring.accepted for ChessBench and
Habermas, else [gold]); noul -> (a["noul"] >= 0.5) == gold. Added benchmarks use the kit's scored_keys (unscored
fields dropped; PhishNChips only `verdict`). iSarcasmEval track B is skipped (the kit does not score it). Nothing here
is tuned; there is no fitting of any kind.

Run (CPU):  python extract.py --suite $KIT/suite-0.3 --tok <crivo-17m dir> --out ~/e5/tab \
                    c17m=~/e5/runs/c17m ...
"""
import argparse
import gzip
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from decision_index.suite.io import Suite
from decision_index.scoring.added import scored_keys
from decision_index.scoring.index02 import spec

RANKING = {2, 36}  # ToolRet, BRIGHT: nDCG over candidate yes/no fields (unjudged candidates are not verified negatives)


def render(content):  # identical to the released Crivo model.render
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def option_texts(q):  # identical to the released Crivo model.option_texts (texts only)
    if q["type"] == "choice":
        out = []
        for k, v in q["criteria"].items():
            d = render(v)
            out.append(f"{k}: {d}" if d and d != k else str(k))
        return out
    c = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
    f, t = render(c.get("false")), render(c.get("true"))
    return [f"false: {f}" if f else "No.", f"true: {t}" if t else "Yes."]


_TOK = None


def _init(tok_dir):
    global _TOK
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))


def _count(texts):
    return [len(e.ids) for e in _TOK.encode_batch(texts, add_special_tokens=False)]


def build_questions(suite_dir, tok_dir, workers):
    s3 = spec("0.3")
    area_of = {n: a["id"] for a in s3["areas"] for n in a["benchmarks"]}
    added = {int(k) for k in s3["added"]}
    bmeta = json.loads((Path(__import__("decision_index").__file__).parent / "data" / "benchmarks.json").read_text())
    name_of = {b["catalog_id"]: b["dataset"] for b in bmeta}
    area_any = {b["catalog_id"]: b["area"] for b in bmeta}
    for k, v in s3["added"].items():
        name_of[int(k)] = v["short"]
    for it in s3["not_in_index"]:
        name_of.setdefault(it["id"], it["name"])
    suite = Suite(suite_dir, "0.3")
    recs, states, qtexts, n_req, skipped = [], [], [], 0, {}
    for r in suite.rows(apply_exclusions=True):
        e = r["_evaluation"]
        n = e["catalog_id"]
        if n == 40 and "-B-" in e["track"]:
            continue
        if n in RANKING or n == 6:  # ranking (nDCG) and RouterBench (utility) have no per-field gold
            continue
        keys = scored_keys(n, r) if n in added else list(r["questions"])
        if not keys:
            continue
        sid = len(states)
        states.append(render(r["state"]))
        n_req += 1
        acc_list = (r.get("scoring") or {}).get("accepted") if n in (31, 50) else None
        for key in keys:
            q = r["questions"][key]
            gold = r["expected"][key]
            if q["type"] == "choice":
                opts = list(q["criteria"])
                acc = acc_list if acc_list is not None else [gold]
                gi = [opts.index(a) for a in acc if a in opts]
                k = len(opts)
                yesno = set(o.lower() for o in opts) in ({"yes", "no"}, {"true", "false"})
            else:
                opts = [False, True]
                gi = [opts.index(bool(gold))]
                k = 2
                yesno = True
            if not gi:
                skipped[n] = skipped.get(n, 0) + 1
                continue
            qtexts.append((render(q.get("instructions")), " ".join(option_texts(q))))
            recs.append(dict(run_id=e["run_id"], key=key, bench=n, bench_name=name_of.get(n, str(n)),
                             area=area_of.get(n, area_any.get(n, "?")), track=e["track"], qtype=q["type"], k=k,
                             gold_idx=gi[0], gold_set=json.dumps(gi), n_accepted=len(gi), yesno=yesno, sid=sid,
                             proxy_tokens=e["proxy_tokens"], nq=len(keys), in_index=n in area_of))
    df = pd.DataFrame(recs)
    # Ettin tokens of the rendered state, once per request (all Crivo sizes share the tokenizer)
    chunks = [states[i:i + 2000] for i in range(0, len(states), 2000)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(tok_dir,)) as ex:
        counts = [c for part in ex.map(_count, chunks) for c in part]
    df["state_tokens"] = np.asarray(counts, dtype=np.int64)[df["sid"].to_numpy()]
    # per question: instructions and option texts as the Crivo reader renders them; input_tokens = the whole
    # encoder sequence ([CLS] instr [SEP] state [SEP] + one [OPT] per option + option texts)
    for col, j in (("instr_tokens", 0), ("opt_tokens", 1)):
        texts = [t[j] for t in qtexts]
        chunks = [texts[i:i + 5000] for i in range(0, len(texts), 5000)]
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(tok_dir,)) as ex:
            df[col] = np.asarray([c for part in ex.map(_count, chunks) for c in part], dtype=np.int64)
    df["input_tokens"] = df.state_tokens + df.instr_tokens + df.opt_tokens + df.k + 3
    print(f"fields skipped (gold not among options): {skipped}", file=sys.stderr)
    print(f"questions: {len(df)} fields in {n_req} requests; in index (per-field): "
          f"{int(df.in_index.sum())}", file=sys.stderr)
    return df


def load_answers(run_dir):
    out = {}
    with gzip.open(Path(run_dir) / "results.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            out[r["run_id"]] = (r["status"], ((r.get("response") or {}).get("answers")) or {})
    return out


def model_arrays(df, suite_dir, run_dir):
    """Per-question metrics for one run; unanswered/invalid = wrong with a uniform distribution."""
    ans = load_answers(run_dir)
    # criteria order per (run_id, key) has to come from the suite (results files are compact)
    suite = Suite(suite_dir, "0.3")
    want = set(df.run_id)
    crit = {}
    for r in suite.rows(apply_exclusions=True):
        rid = r["_evaluation"]["run_id"]
        if rid in want:
            crit[rid] = {k: (list(q["criteria"]) if q["type"] == "choice" else None) for k, q in r["questions"].items()}
    n = len(df)
    conf = np.zeros(n); pacc = np.zeros(n); brier = np.zeros(n); nll = np.zeros(n)
    correct = np.zeros(n, bool); pred_idx = np.full(n, -1, np.int16); uniform = np.zeros(n, bool)
    bad = 0
    for i, (rid, key, qtype, gs) in enumerate(zip(df.run_id, df.key, df.qtype, df.gold_set)):
        gset = json.loads(gs)
        status, answers = ans.get(rid, ("missing", {}))
        a = answers.get(key) if status == "ok" else None
        opts = crit[rid][key]
        if qtype == "choice":
            p = None
            if a is not None and isinstance(a.get("probabilities"), dict):
                p = np.array([float(a["probabilities"].get(o, 0.0)) for o in opts])
            if p is None or not np.isfinite(p).all() or p.sum() <= 0:
                bad += 1
                p = np.full(len(opts), 1 / len(opts)); choice_idx = -1
            else:
                p = p / p.sum()
                choice_idx = opts.index(a["choice"]) if a.get("choice") in opts else -1
        else:
            if a is not None and isinstance(a.get("noul"), (int, float)) and math.isfinite(a["noul"]):
                t = min(1.0, max(0.0, float(a["noul"])))
                p = np.array([1 - t, t]); choice_idx = int(t >= 0.5)
            else:
                bad += 1
                p = np.array([0.5, 0.5]); choice_idx = -1
        pa = float(p[gset].sum())
        rest = np.delete(p, gset)
        conf[i] = p.max()
        pacc[i] = pa
        brier[i] = (pa - 1) ** 2 + float((rest ** 2).sum())
        nll[i] = -math.log(max(pa, 1e-12))
        correct[i] = choice_idx in gset
        pred_idx[i] = choice_idx
        uniform[i] = float(p.max() - p.min()) < 1e-9
    return dict(conf=conf, correct=correct, pacc=pacc, brier=brier, nll=nll, pred_idx=pred_idx, uniform=uniform), bad


def _one(args):
    name, run_dir, suite_dir, tab = args
    df = pd.read_parquet(Path(tab) / "questions.parquet", columns=["run_id", "key", "qtype", "gold_set"])
    arrs, bad = model_arrays(df, suite_dir, run_dir)
    np.savez_compressed(Path(tab) / f"preds_{name}.npz", **arrs)
    return name, bad, float(arrs["correct"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("runs", nargs="+", help="name=run_dir")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not (out / "questions.parquet").exists():
        build_questions(a.suite, a.tok, 16).to_parquet(out / "questions.parquet")
    jobs = [(x.split("=", 1)[0], x.split("=", 1)[1], a.suite, str(out)) for x in a.runs]
    with ProcessPoolExecutor(a.workers) as ex:
        for name, bad, acc in ex.map(_one, jobs):
            print(f"{name}: invalid/unanswered fields {bad}; pooled field accuracy {acc:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
