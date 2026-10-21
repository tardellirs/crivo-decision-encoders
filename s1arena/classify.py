"""S1-Arena contamination table per task (non-agentic, HF snapshot), for the v2 family (mix400_*).

    python classify.py <audit dir> <src_counts.json> <tasks.json> <out.json> <out.md>

tasks.json: {task: {"source": ..., "url": ..., "n": ...}} (written from the parquet files).
Source-level exposure: a training `src` whose alphanumeric form contains the task's dataset token (last path
segment of the task name, or the HF dataset name of its url, >= 4 chars, minus generic words), counted over the
v2-family training files. Item-level overlap from contam.py (longest state leaf; conservative: a shared premise
counts). Classes as for BTZSC: seen-test-overlap (>= 0.5% items), seen-train-split, unseen.
"""
import json
import re
import sys

GENERIC = {"test", "train", "data", "dataset", "datasets", "full", "default", "english", "en", "main", "all", "v1", "v2",
           "prompts", "responses", "eval", "bench", "benchmark", "huggingface", "github", "com", "org", "the"}
ALIAS = {"anli_r3": ["anli"], "winogrande": ["winogrande"], "clinc150": ["clinc"], "massive_en": ["massive"],
         "banking77": ["banking77"], "sst5": ["sst", "rotten"], "esci": ["esci"], "hellaswag": ["hellaswag"],
         "bfcl_live": ["bfcl"], "bfcl_v3": ["bfcl"], "ragtruth": ["ragtruth"], "goemotions": ["goemotions", "go_emotions"],
         "financial_phrasebank": ["financial_phrasebank", "auditor_review"], "strategyqa": ["strategyqa"],
         "cuad": ["cuad"], "sms_spam": ["sms_spam"], "scifact": ["scifact"], "toxicchat_jailbreak": ["toxicchat", "toxic-chat"],
         "chaosnli": ["chaosnli", "chaos_nli"], "fewrel": ["few_rel", "fewrel"], "fewnerd": ["few_nerd", "fewnerd"],
         "squad_v2": ["squad2", "squad"], "dbpedia_l2": ["dbpedia"], "nycc_captions": ["newyorker"], "isarcasmeval": ["isarcasm"],
         "scruples_anecdotes": ["scruples"], "nq_open": ["natural_questions", "bge-hn-nq", "bge_m3__nq"], "ambigqa": ["ambignq", "ambigqa"],
         "bbh_date_understanding": ["bbh"], "mmlu_pro": ["mmlu"], "hotpotqa_comparison": ["hotpotqa"]}
alnum = lambda s: re.sub(r"[^0-9a-z]+", "", s.lower())


def tokens(task, url):
    last = task.split("/")[-1]
    toks = set(ALIAS.get(last, [])) | {last}
    m = re.search(r"huggingface\.co/datasets/[^/]+/([^/?#]+)", url or "")
    if m:
        toks.add(m.group(1))
    out = set()
    for t in toks:
        a = alnum(t)
        if len(a) >= 4 and a not in GENERIC:
            out.add(a)
    return out


def main():
    adir, srcf, tasksf, outj, outm = sys.argv[1:6]
    src = json.load(open(srcf)); tasks = json.load(open(tasksf))
    aud = json.load(open(f"{adir}/s1arena_audit_v2family.json"))["per_dataset"]
    v2 = {}
    for f, c in src.items():
        if f.startswith("mix400_") and ".dev." not in f:
            for s, n in c.items():
                v2[s] = v2.get(s, 0) + n
    srcs_alnum = {s: alnum(s) for s in v2}
    res = {}
    for t, info in sorted(tasks.items()):
        toks = tokens(t, info.get("url"))
        hits = {s: v2[s] for s, a in srcs_alnum.items() if any(k in a for k in toks)}
        a = aud.get(t, {"n": info["n"], "any": 0, "srcs": {}})
        rate = a["any"] / a["n"]
        cls = "seen-test-overlap" if rate >= 0.005 else ("seen-train-split" if hits else "unseen")
        res[t] = {"domain": t.split("/")[0], "n": a["n"], "class": cls, "overlap_items": a["any"], "overlap_rate": round(rate, 4),
                  "overlap_srcs": a.get("srcs", {}), "source_tokens": sorted(toks), "train_rows_same_source": sum(hits.values()),
                  "train_sources": dict(sorted(hits.items(), key=lambda x: -x[1])[:5]), "source": info.get("source"), "url": info.get("url")}
    json.dump(res, open(outj, "w"), indent=1)
    dom = {}
    for t, v in res.items():
        d = dom.setdefault(v["domain"], {"tasks": 0, "unseen": 0, "seen-train-split": 0, "seen-test-overlap": 0, "items": 0, "unseen_items": 0})
        d["tasks"] += 1; d[v["class"]] += 1; d["items"] += v["n"]; d["unseen_items"] += v["n"] if v["class"] == "unseen" else 0
    L = ["| Domain | Tasks | unseen | seen-train-split | seen-test-overlap | items | items in unseen tasks |", "|---|---:|---:|---:|---:|---:|---:|"]
    tot = {k: 0 for k in ("tasks", "unseen", "seen-train-split", "seen-test-overlap", "items", "unseen_items")}
    for d, v in sorted(dom.items()):
        L.append(f"| {d} | {v['tasks']} | {v['unseen']} | {v['seen-train-split']} | {v['seen-test-overlap']} | {v['items']:,} | {v['unseen_items']:,} |")
        for k in tot:
            tot[k] += v[k]
    L.append(f"| **all** | {tot['tasks']} | {tot['unseen']} | {tot['seen-train-split']} | {tot['seen-test-overlap']} | {tot['items']:,} | {tot['unseen_items']:,} |")
    L += ["", "| Task | Items | Class | Overlap | Training sources matched (rows) |", "|---|---:|---|---:|---|"]
    for t, v in res.items():
        if v["class"] != "unseen":
            L.append(f"| {t} | {v['n']} | {v['class']} | {v['overlap_items']} ({v['overlap_rate']*100:.1f}%) | "
                     f"{', '.join(f'{s} ({n})' for s, n in list(v['train_sources'].items())[:3]) or '-'} |")
    open(outm, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
