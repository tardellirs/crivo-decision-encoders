"""BTZSC -> Decision Index `choice` requests, and BTZSC scoring of our results.

Loading replicates btzsc.data.load_btzsc_dataset (harness commit 4eac6a9) exactly, reading the local parquet
snapshot of btzsc/btzsc: grouping by the binary-label period, the yahootopics "Business & Finance" filter with the
first 20 rows kept, the gold index = first entailment row (0 when a group has none: 200 banking77 and 169 massive
items, as in the harness), and the max_samples sampling (seed 0, first two groups kept).

One fixed rendering for every dataset (decided before any run, never tuned):
    state        = the text
    instructions = INSTRUCTIONS (below, identical for all 22 datasets)
    criteria     = {label_name: BTZSC hypothesis}   in BTZSC's own label order
label_name is the dataset's label_text for that hypothesis (the class name); the description is the BTZSC
verbalization every published BTZSC model receives. `--render names` (names only, no description) exists only as a
declared secondary check.

    python adapter.py requests --root <btzsc data dir> --out req.jsonl [--max_samples 50] [--datasets a,b] [--render desc]
    python adapter.py targets  --root <btzsc data dir> --out targets.jsonl
    python adapter.py score    --requests req.jsonl --results results.jsonl [--out score.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DATASETS = ["amazonpolarity", "imdb", "appreviews", "yelpreviews", "rottentomatoes", "financialphrasebank",
            "emotiondair", "empathetic", "banking77", "biasframes_intent", "massive", "agnews", "yahootopics",
            "trueteacher", "manifesto", "capsotu", "biasframes_offensive", "biasframes_sex",
            "wikitoxic_toxicaggregated", "wikitoxic_obscene", "wikitoxic_threat", "wikitoxic_insult"]
TASK = {**{d: "sentiment" for d in DATASETS[:6]}, "emotiondair": "emotion", "empathetic": "emotion",
        "banking77": "intent", "biasframes_intent": "intent", "massive": "intent",
        **{d: "topic" for d in DATASETS[11:]}}
INSTRUCTIONS = "Which label best describes the text?"
SAMPLING_SEED = 0


def load(name: str, root: str, max_samples: int | None = None) -> dict:
    import pyarrow.parquet as pq
    files = sorted(Path(root, name).glob("test-*.parquet"))
    t = pq.read_table(files).to_pydict() if len(files) == 1 else None
    if t is None:
        import pyarrow as pa
        t = pa.concat_tables([pq.read_table(f) for f in files]).to_pydict()
    text, hyp, lab, lt = t["text"], t["hypothesis"], t["labels"], t["label_text"]
    if name == "yahootopics":   # harness: first 20 rows + every row whose label_text != "Business & Finance"
        keep = list(range(min(20, len(text)))) + [i for i in range(len(text)) if lt[i] != "Business & Finance"]
        text, hyp, lab, lt = ([c[i] for i in keep] for c in (text, hyp, lab, lt))
    first = lab[0]
    n = next((i for i in range(1, len(lab)) if lab[i] == first), len(lab))
    total = len(lab) // n
    if max_samples is not None and max_samples < total:
        if max_samples <= 1:
            idx = [0][:max_samples]
        else:
            idx = [0, 1, *random.Random(SAMPLING_SEED).sample(range(2, total), k=max_samples - 2)]
    else:
        idx = list(range(total))
    labels = hyp[:n]
    names = {}
    for i in range(len(lab)):   # class name of a hypothesis = label_text of its entailment rows
        if lab[i] == 1:
            names.setdefault(hyp[i], lt[i])
    label_names = [names.get(h, h) for h in labels]
    assert len(set(label_names)) == len(label_names), (name, "duplicate label names")
    texts, refs = [], []
    for g in idx:
        o = g * n
        texts.append(text[o])
        b = lab[o:o + n]
        refs.append(b.index(1) if 1 in b else 0)
    return {"name": name, "texts": texts, "labels": labels, "label_names": label_names, "refs": refs,
            "sample_idx": idx, "task": TASK[name]}


def request(ds: dict, j: int, render: str = "desc") -> dict:
    crit = {k: (h if render == "desc" else None) for k, h in zip(ds["label_names"], ds["labels"])}
    gold = ds["label_names"][ds["refs"][j]]
    return {"id": f"btzsc/{ds['name']}/{ds['sample_idx'][j]}", "family": f"btzsc/{ds['name']}", "state": ds["texts"][j],
            "questions": {"label": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": crit}},
            "expected": {"label": gold},
            "meta": {"bench": "btzsc", "ds": ds["name"], "task": ds["task"], "ref": ds["refs"][j], "render": render}}


def macro_f1(refs, preds, k):
    from sklearn.metrics import accuracy_score, f1_score
    return float(f1_score(refs, preds, average="macro", zero_division=0)), float(accuracy_score(refs, preds))


def score(req_path: str, res_path: str) -> dict:
    reqs = {}
    for l in open(req_path):
        r = json.loads(l); reqs[r["id"]] = r
    res = {}
    for l in open(res_path):
        r = json.loads(l); res[r["id"]] = r
    per = defaultdict(lambda: ([], [], 0, 0))
    out = {}
    by = defaultdict(list)
    for rid, r in reqs.items():
        ds = r["meta"]["ds"]
        keys = list(r["questions"]["label"]["criteria"])
        x = res.get(rid)
        ok = x is not None and x.get("status") == "ok"
        # unanswered = wrong: predicted as an impossible class (-1), as the board does
        pred = keys.index(x["response"]["answers"]["label"]["choice"]) if ok else -1
        by[ds].append((r["meta"]["ref"], pred, ok))
    for ds in DATASETS:
        if ds not in by:
            continue
        refs = [a for a, _, _ in by[ds]]; preds = [b for _, b, _ in by[ds]]
        f1, acc = macro_f1(refs, preds, None)
        out[ds] = {"n": len(refs), "answered": sum(c for _, _, c in by[ds]), "macro_f1": f1, "accuracy": acc}
    tasks = defaultdict(list)
    for ds, v in out.items():
        tasks[TASK[ds]].append(v["macro_f1"])
    summary = {"overall_macro_f1": sum(v["macro_f1"] for v in out.values()) / max(1, len(out)), "n_datasets": len(out),
               "by_task": {k: sum(v) / len(v) for k, v in tasks.items()}}
    return {"summary": summary, "by_dataset": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["requests", "targets", "score"])
    ap.add_argument("--root"); ap.add_argument("--out")
    ap.add_argument("--max_samples", type=int); ap.add_argument("--datasets")
    ap.add_argument("--render", default="desc", choices=["desc", "names"])
    ap.add_argument("--requests"); ap.add_argument("--results")
    a = ap.parse_args()
    names = a.datasets.split(",") if a.datasets else DATASETS
    if a.cmd == "score":
        s = score(a.requests, a.results)
        if a.out:
            json.dump(s, open(a.out, "w"), indent=1)
        for ds, v in s["by_dataset"].items():
            print(f"{ds:28s} n={v['n']:6d} answered={v['answered']:6d} F1={v['macro_f1']:.4f} acc={v['accuracy']:.4f}")
        print(json.dumps(s["summary"]))
        return
    with open(a.out, "w") as f:
        for name in names:
            ds = load(name, a.root, a.max_samples)
            for j in range(len(ds["texts"])):
                if a.cmd == "requests":
                    f.write(json.dumps(request(ds, j, a.render), ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps({"bench": "btzsc", "ds": name, "id": ds["sample_idx"][j], "text": ds["texts"][j]},
                                       ensure_ascii=False) + "\n")
            print(name, len(ds["texts"]), "classes", len(ds["labels"]), file=sys.stderr)


if __name__ == "__main__":
    main()
