"""Re-score one GPU-1 system with the official scorers' own code paths, memory-light, and compare with its .score.json.

S1-Arena: per domain, the parquet items become s1arena.schema.Item objects and every task goes through
s1arena.metrics.score, exactly as `s1arena/adapter.py score` does on the GPU box (same to_di / to_s1_answers mapping,
last line per id, unanswered = error row). Only one domain's items and results are in memory at a time.
BTZSC: the request file is streamed keeping (id, dataset, ref, label keys); predictions = index of `choice` (-1 when not
ok), then `btzsc/adapter.py::macro_f1` (sklearn) per dataset, as `adapter.py score` does.

    python3 stats/rescore_check.py s1    --data stats/gpu1/ext/s1arena_data --code stats/gpu1/ext/s1arena_code \
        --results results_gpu/s1/crivo-17m.jsonl --score results_gpu/s1/crivo-17m.score.json
    python3 stats/rescore_check.py btzsc --requests stats/gpu1/req/btzsc_full.jsonl \
        --results results_gpu/btzsc/crivo-17m.jsonl --score results_gpu/btzsc/crivo-17m.score.json
"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def s1(a):
    sys.path.insert(0, str(Path(a.code).resolve()))
    from s1arena import metrics
    from s1arena.schema import Item
    ad = load_mod("s1_adapter", EXP / "s1arena" / "adapter.py")
    ref = json.load(open(a.score))
    per_task = {}
    for dom in ad.NON_AGENTIC:
        raw = list(ad.items(a.data, [dom]))
        ids = {it["id"] for it in raw}
        res = {}
        with open(a.results) as f:
            for l in f:
                x = json.loads(l)
                if x["id"] in ids:
                    res[x["id"]] = x                 # last line per id wins
        by_task = defaultdict(list)
        for it in raw:
            r = ad.to_di(it)
            item = Item(id=it["id"], leaf=it["leaf"], task=it["task"], request=json.loads(it["request"]),
                        gold=json.loads(it["gold"]), meta=json.loads(it["meta"]) if it.get("meta") else {})
            x = res.get(it["id"]); answers = {}
            if x and x.get("status") == "ok":
                for k, q in r["questions"].items():
                    if k in x["response"]["answers"]:
                        answers[k] = ad.to_s1_answers(x["response"]["answers"][k], q)
            row = {"item_id": it["id"], "sample": 0,
                   "response": {"answers": answers, "error": None if answers else (x or {}).get("status", "missing"),
                                "latency_ms": (x or {}).get("total_wall_ms")}}
            by_task[it["task"]].append((item, row))
        for task, rows in sorted(by_task.items()):
            m = metrics.score([i for i, _ in rows], [r for _, r in rows])
            per_task[task] = {"domain": dom, **{k: m.get(k) for k in ("n_questions", "accuracy", "accuracy_se", "brier", "ece", "unanswered_rate")}}
        print(dom, len(raw), "items", flush=True)
        del raw, res, by_task
    out = {"tasks": len(per_task), "max_abs_diff": {}, "exact_equal_fields": {}}
    for k in ("n_questions", "accuracy", "accuracy_se", "brier", "ece", "unanswered_rate"):
        d = [abs(per_task[t][k] - ref["tasks"][t][k]) for t in ref["tasks"]]
        out["max_abs_diff"][k] = max(d)
        out["exact_equal_fields"][k] = sum(per_task[t][k] == ref["tasks"][t][k] for t in ref["tasks"])
    acc = sum(v["accuracy"] for v in per_task.values()) / len(per_task)
    out["overall_accuracy"] = acc; out["overall_accuracy_score_json"] = ref["summary"]["accuracy"]
    assert sorted(per_task) == sorted(ref["tasks"])
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def btzsc(a):
    ad = load_mod("btzsc_adapter", EXP / "btzsc" / "adapter.py")
    ref = json.load(open(a.score))
    meta, keys = {}, {}
    with open(a.requests) as f:
        for l in f:
            r = json.loads(l); ds = r["meta"]["ds"]
            k = list(r["questions"]["label"]["criteria"])
            if ds not in keys:
                keys[ds] = k
            assert keys[ds] == k
            meta[r["id"]] = (ds, r["meta"]["ref"])
    pred = {}
    with open(a.results) as f:
        for l in f:
            x = json.loads(l)
            if x["id"] not in meta:
                continue
            ok = x.get("status") == "ok"
            pred[x["id"]] = keys[meta[x["id"]][0]].index(x["response"]["answers"]["label"]["choice"]) if ok else -1
    by = defaultdict(lambda: ([], []))
    for rid, (ds, rf) in meta.items():          # request-file order, as adapter.score
        by[ds][0].append(rf); by[ds][1].append(pred.get(rid, -1))
    out = {"max_abs_diff": {}, "exact_equal": {}}
    md = {"macro_f1": 0.0, "accuracy": 0.0}; ex = {"macro_f1": 0, "accuracy": 0}
    for ds, (rf, pr) in by.items():
        f1, acc = ad.macro_f1(rf, pr, None)
        for k, v in (("macro_f1", f1), ("accuracy", acc)):
            md[k] = max(md[k], abs(v - ref["by_dataset"][ds][k])); ex[k] += v == ref["by_dataset"][ds][k]
        assert len(rf) == ref["by_dataset"][ds]["n"]
    out["max_abs_diff"] = md; out["exact_equal"] = ex; out["datasets"] = len(by)
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["s1", "btzsc"])
    for k in ("data", "code", "requests", "results", "score", "out"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args()
    {"s1": s1, "btzsc": btzsc}[a.cmd](a)
