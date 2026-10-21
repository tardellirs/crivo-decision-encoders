"""System1-Arena (2610.21244) <-> Decision Index wire format, and S1-Arena scoring of our results.

Requests are taken as they are from the HF snapshot (C-Tianyu/S1-Arena, test split). The only mapping is the
question type: S1-Arena `boolean` (criteria null) -> DI `noul` (criteria {}), answered as P(true); `choice` and
`score` pass unchanged (score criteria = ordered list of levels, option ids "0".."n-1"). Nothing is rendered or
tuned per task. Scoring calls s1arena.metrics.task_metrics (the benchmark's own code) per task; a domain is the mean
over its tasks and the overall non-agentic accuracy is the mean over all tasks of the 13 non-agentic domains, as on
the S1-Arena leaderboard.

    python adapter.py requests --data <S1-Arena data dir> --out req.jsonl [--domains basic_cls,know] [--per_task 5]
    python adapter.py targets  --data <S1-Arena data dir> --out targets.jsonl
    python adapter.py score    --data <dir> --code <S1-Arena code dir> --requests req.jsonl --results results.jsonl --out s.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

NON_AGENTIC = ["basic_cls", "know", "social", "pro", "ret", "struct", "code", "fore", "dialog", "batch", "judge",
               "safety", "route"]


def items(data: str, domains):
    import pyarrow.parquet as pq
    for d in domains:
        for f in sorted(glob.glob(os.path.join(data, d, "test-*.parquet"))):
            t = pq.read_table(f).to_pylist()
            for r in t:
                r["domain"] = d
                yield r


def to_di(item: dict) -> dict:
    req = json.loads(item["request"])
    qs = {}
    for k, q in req["questions"].items():
        if q["type"] == "boolean":
            qs[k] = {"type": "noul", "instructions": q["instructions"], "criteria": {}}
        else:
            qs[k] = {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}
    gold = json.loads(item["gold"])
    return {"id": item["id"], "family": f"s1arena/{item['task']}", "state": req["state"], "questions": qs,
            "expected": {k: g.get("label") for k, g in gold.items()},
            "meta": {"bench": "s1arena", "task": item["task"], "leaf": item["leaf"], "domain": item["domain"],
                     "source": item.get("source"), "url": item.get("url")}}


def longest_leaf(o):
    best = ""
    stack = [o]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            if len(x) > len(best):
                best = x
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return best


def to_s1_answers(di_answer: dict, q: dict) -> dict:
    """Our answer for one question -> S1-Arena {"probs": {option_id: p}}."""
    if q["type"] == "noul":
        p = float(di_answer["noul"])
        return {"probs": {"true": p, "false": 1 - p}}
    if q["type"] == "score":
        probs = di_answer.get("probabilities")
        return {"probs": {str(k): float(v) for k, v in probs.items()}}
    return {"probs": {k: float(v) for k, v in di_answer["probabilities"].items()}}


def score(a):
    sys.path.insert(0, a.code)
    from s1arena import metrics
    from s1arena.schema import Item
    reqs = {}
    for l in open(a.requests):
        r = json.loads(l); reqs[r["id"]] = r
    res = {}
    for l in open(a.results):
        r = json.loads(l); res[r["id"]] = r
    by_task = defaultdict(list)
    raw = {it["id"]: it for it in items(a.data, sorted({r["meta"]["domain"] for r in reqs.values()}))}
    for rid, r in reqs.items():
        it = raw[rid]
        item = Item(id=it["id"], leaf=it["leaf"], task=it["task"], request=json.loads(it["request"]),
                    gold=json.loads(it["gold"]), meta=json.loads(it["meta"]) if it.get("meta") else {})
        x = res.get(rid)
        answers = {}
        if x and x.get("status") == "ok":
            for k, q in r["questions"].items():
                if k in x["response"]["answers"]:
                    answers[k] = to_s1_answers(x["response"]["answers"][k], q)
        row = {"item_id": rid, "sample": 0, "response": {"answers": answers, "error": None if answers else (x or {}).get("status", "missing"),
                                                         "latency_ms": (x or {}).get("total_wall_ms")}}
        by_task[it["task"]].append((item, row, r["meta"]["domain"]))
    per_task = {}
    for task, rows in sorted(by_task.items()):
        m = metrics.score([i for i, _, _ in rows], [r for _, r, _ in rows])
        per_task[task] = {"domain": rows[0][2], **{k: m.get(k) for k in ("n_questions", "accuracy", "accuracy_se", "brier", "ece", "unanswered_rate")}}
    dom = defaultdict(list)
    for t, v in per_task.items():
        dom[v["domain"]].append(v)
    domains = {d: {"accuracy": sum(x["accuracy"] for x in v) / len(v), "brier": sum(x["brier"] for x in v) / len(v),
                   "tasks": len(v)} for d, v in dom.items()}
    allv = list(per_task.values())
    summary = {"accuracy": sum(x["accuracy"] for x in allv) / len(allv), "brier": sum(x["brier"] for x in allv) / len(allv),
               "tasks": len(allv), "questions": sum(x["n_questions"] for x in allv)}
    out = {"summary": summary, "domains": domains, "tasks": per_task}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    for d, v in domains.items():
        print(f"{d:10s} tasks={v['tasks']:3d} acc={v['accuracy']:.4f} brier={v['brier']:.4f}")
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["requests", "targets", "score"])
    ap.add_argument("--data"); ap.add_argument("--code"); ap.add_argument("--out")
    ap.add_argument("--domains", default=",".join(NON_AGENTIC))
    ap.add_argument("--per_task", type=int, default=0, help="sample this many items per task (seed 0), for smoke tests")
    ap.add_argument("--requests"); ap.add_argument("--results")
    a = ap.parse_args()
    if a.cmd == "score":
        return score(a)
    doms = a.domains.split(",")
    allit = list(items(a.data, doms))
    if a.per_task:
        by = defaultdict(list)
        for it in allit:
            by[it["task"]].append(it)
        allit = [x for t in sorted(by) for x in random.Random(0).sample(by[t], min(a.per_task, len(by[t])))]
    with open(a.out, "w") as f:
        for it in allit:
            if a.cmd == "requests":
                f.write(json.dumps(to_di(it), ensure_ascii=False) + "\n")
            else:
                req = json.loads(it["request"])
                f.write(json.dumps({"bench": "s1arena", "ds": it["task"], "id": it["id"], "text": longest_leaf(req["state"])},
                                   ensure_ascii=False) + "\n")
    print(len(allit), "items", file=sys.stderr)


if __name__ == "__main__":
    main()
