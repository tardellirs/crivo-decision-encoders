"""Collect the published BTZSC baseline scores (macro-F1 per dataset + overall) from btzsc/btzsc-results
(results/<type>/<model>.json) and the harness's packaged base-encoder rows (_baselines/f1_scores.csv).

    python published.py <btzsc-results dir> <harness dir> <out.json>
"""
import csv, glob, json, os, sys

DATASETS = ["amazonpolarity", "imdb", "appreviews", "yelpreviews", "rottentomatoes", "financialphrasebank",
            "emotiondair", "empathetic", "banking77", "biasframes_intent", "massive", "agnews", "yahootopics",
            "trueteacher", "manifesto", "capsotu", "biasframes_offensive", "biasframes_sex",
            "wikitoxic_toxicaggregated", "wikitoxic_obscene", "wikitoxic_threat", "wikitoxic_insult"]
res_dir, harness, out = sys.argv[1:4]
rows = {}
for f in sorted(glob.glob(os.path.join(res_dir, "results", "*", "*.json"))):
    j = json.load(open(f))
    m = j["model"]
    by = {d: j["results"]["by_dataset"][d]["macro_f1"] for d in DATASETS if d in j["results"]["by_dataset"]}
    rows[m["name"]] = {"type": m["model_type"], "params": m["params"], "source": "btzsc-results",
                       "overall_macro_f1": j["results"]["overall"]["macro_f1"],
                       "overall_acc": j["results"]["overall"]["accuracy"],
                       "by_task": {k: v["macro_f1"] for k, v in j["results"]["by_task"].items()},
                       "by_dataset": by, "n_datasets": len(by)}
import yaml
meta = yaml.safe_load(open(os.path.join(harness, "src/btzsc/_meta/models.yml")))
with open(os.path.join(harness, "src/btzsc/_baselines/f1_scores.csv")) as fh:
    for r in csv.DictReader(fh):
        name = r["mdl"]
        if name in rows or meta["model_to_type"].get(name) != "base_encoder":
            continue
        by = {d: float(r[d]) for d in DATASETS if r.get(d) not in (None, "")}
        rows[name] = {"type": "base_encoder", "params": meta["model_to_params"].get(name), "source": "harness _baselines csv",
                      "overall_macro_f1": sum(by.values()) / len(by), "by_dataset": by, "n_datasets": len(by)}
json.dump(rows, open(out, "w"), indent=1)
for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["overall_macro_f1"]):
    print(f"{k:34s} {v['type']:13s} {str(v['params']):>6s} n={v['n_datasets']} overall={v['overall_macro_f1']:.3f}")
