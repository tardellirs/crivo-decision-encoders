"""BTZSC contamination table: source-level exposure (training rows whose `src` is the dataset or its source corpus)
plus item-level overlap from contam.py, per model family.

    python classify.py <audit dir> <src_counts.json> <out.json> <out.md>

Families: v2 = mix400_* (17M, 32M v2, 68M, 150M v2, 400M, 1B); 32Mv1 = mix32_* + long_v3; 150Mv1 = mix_v1/v2c/v3.
Classes: seen-test-overlap = item overlap (exact on >=5-word text or 8-gram containment >= 0.8) for >= 0.5% of the
test items; seen-train-split = the dataset/source corpus is in training (any rows) but overlap < 0.5%;
unseen = neither. `related` lists training sources of the same task type on a different corpus (not counted as seen).
"""
import json
import re
import sys

SRC = {  # dataset -> regex over training `src` naming the same dataset or its source corpus
    "amazonpolarity": r"amazon_polarity", "imdb": r"(^|/)imdb$|counterfactually-augmented-imdb",
    "appreviews": r"app_reviews", "yelpreviews": r"yelp_review", "rottentomatoes": r"rotten_tomatoes|glue/sst2",
    "financialphrasebank": r"financial_phrasebank|auditor_review", "emotiondair": r"^t3/emotion$",
    "empathetic": r"empathetic", "banking77": r"banking77", "biasframes_intent": r"SBIC", "biasframes_offensive": r"SBIC",
    "biasframes_sex": r"SBIC", "massive": r"(^|/)massive$", "agnews": r"ag_?news", "yahootopics": r"yahoo_answers",
    "trueteacher": r"trueteacher", "manifesto": r"manifesto|Pol_NLI", "capsotu": r"capsotu|comparative_agendas",
    "wikitoxic_toxicaggregated": r"jigsaw", "wikitoxic_obscene": r"jigsaw", "wikitoxic_threat": r"jigsaw",
    "wikitoxic_insult": r"jigsaw"}
RELATED = {"amazonpolarity": r"amazon_counterfactual|amazon_reviews_multi", "emotiondair": r"go_emotions|tweet_eval/emotion|text_emotion",
           "empathetic": r"go_emotions|(^|/)emotion$", "massive": r"clinc|hwu64|mtop|snips", "banking77": r"bitext|minds14",
           "biasframes_intent": r"hate_speech", "wikitoxic_toxicaggregated": r"civil_comments|toxic", "manifesto": r"policy",
           "trueteacher": r"doc-nli|factcc|summac", "yahootopics": r"", "capsotu": r""}
FAM = {"v2": lambda f: f.startswith("mix400_"), "32Mv1": lambda f: f.startswith("mix32_") or f.startswith("long_v3"),
       "150Mv1": lambda f: f.startswith("mix_v")}
AUD = {"v2": "btzsc_audit_v2family.json", "32Mv1": "btzsc_audit_32mv1.json", "150Mv1": "btzsc_audit_150mv1.json"}


def main():
    adir, srcf, outj, outm = sys.argv[1:5]
    src = json.load(open(srcf))
    res = {}
    for fam, keep in FAM.items():
        aud = json.load(open(f"{adir}/{AUD[fam]}"))["per_dataset"]
        for ds, pat in SRC.items():
            rows, rel = {}, {}
            for f, c in src.items():
                if not keep(f) or ".dev." in f:
                    continue
                for s, n in c.items():
                    if re.search(pat, s, re.I):
                        rows[s] = rows.get(s, 0) + n
                    elif RELATED.get(ds) and re.search(RELATED[ds], s, re.I):
                        rel[s] = rel.get(s, 0) + n
            a = aud[ds]
            rate = a["any"] / a["n"]
            cls = "seen-test-overlap" if rate >= 0.005 else ("seen-train-split" if rows else "unseen")
            if cls == "unseen" and a["any"]:
                cls = "unseen (stray overlap)"
            res.setdefault(ds, {})[fam] = {"class": cls, "train_rows_same_source": sum(rows.values()), "sources": rows,
                                           "related_rows": sum(rel.values()), "related": dict(sorted(rel.items(), key=lambda x: -x[1])[:4]),
                                           "test_items": a["n"], "overlap_items": a["any"], "overlap_rate": round(rate, 4),
                                           "exact": a["exact"], "c80": a["c80"], "short_exact": a.get("short_exact", 0),
                                           "overlap_srcs": a["srcs"]}
    json.dump(res, open(outj, "w"), indent=1)
    L = ["| Dataset | Test items | v2 family (17M-1B): class | rows same source | overlap items | 32M v1 | 150M v1 | Overlap sources (v2) |",
         "|---|---:|---|---:|---:|---|---|---|"]
    for ds, v in res.items():
        a = v["v2"]
        L.append(f"| {ds} | {a['test_items']:,} | {a['class']} | {a['train_rows_same_source']:,} | {a['overlap_items']} ({a['overlap_rate']*100:.1f}%) "
                 f"| {v['32Mv1']['class']} ({v['32Mv1']['overlap_items']}) | {v['150Mv1']['class']} ({v['150Mv1']['overlap_items']}) "
                 f"| {', '.join(list(a['overlap_srcs'])[:3])} |")
    cnt = {fam: {} for fam in FAM}
    for ds, v in res.items():
        for fam in FAM:
            c = v[fam]["class"].split(" ")[0]
            cnt[fam][c] = cnt[fam].get(c, 0) + 1
    L += ["", "Counts: " + "; ".join(f"{fam}: " + ", ".join(f"{k} {n}" for k, n in sorted(c.items())) for fam, c in cnt.items())]
    open(outm, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
