"""Question-level score of a run on public-suite rows with gold (`expected`): used for the ablations' pub15 set.

    python score_pub.py --requests pub15.jsonl --results res.jsonl [--out s.json]

Per benchmark: accuracy over judged questions (choice: argmax == gold key; noul: (p >= 0.5) == gold), chance = mean
1/n_options, skill = clip((acc - chance)/(1 - chance), 0, 1); unanswered = wrong. Summary: unweighted mean skill over
benchmarks (NOT the official index, which uses native metrics, coverage and area weights; declared as such), plus
pooled top-label ECE (10 equal-mass bins) and multiclass Brier over choice questions.
"""
import argparse
import json
from collections import defaultdict


def ece(conf, hit, bins=10):
    z = sorted(zip(conf, hit))
    n = len(z)
    if not n:
        return None
    tot = 0.0
    for b in range(bins):
        s = z[b * n // bins:(b + 1) * n // bins]
        if s:
            tot += len(s) / n * abs(sum(c for c, _ in s) / len(s) - sum(h for _, h in s) / len(s))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True); ap.add_argument("--results", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    res = {}
    for l in open(a.results):
        r = json.loads(l)
        if r.get("status") == "ok":
            res[r["id"]] = r["response"]["answers"]
    per = defaultdict(lambda: [0, 0, 0.0])
    conf, hit, brier = [], [], []
    for l in open(a.requests):
        r = json.loads(l)
        ans = res.get(r["id"], {})
        for k, q in r["questions"].items():
            g = (r.get("expected") or {}).get(k)
            if g is None:
                continue
            x = ans.get(k)
            if q["type"] == "noul":
                gold = g if isinstance(g, bool) else str(g).lower() in ("true", "1", "yes")
                ok = x is not None and ((x["noul"] >= 0.5) == gold); ch = 0.5
            else:
                ok = x is not None and x["choice"] == g; ch = 1 / len(q["criteria"])
                if x is not None:
                    p = x["probabilities"]; c = max(p.values())
                    conf.append(c); hit.append(float(ok))
                    brier.append(sum((v - (1.0 if kk == g else 0.0)) ** 2 for kk, v in p.items()))
            d = per[r["family"]]
            d[0] += 1; d[1] += ok; d[2] += ch
    out = {}
    for b, (n, h, ch) in sorted(per.items()):
        acc, c = h / n, ch / n
        out[b] = {"n": n, "acc": acc, "chance": c, "skill": max(0.0, min(1.0, (acc - c) / (1 - c))) if c < 1 else 0.0}
    summ = {"benchmarks": len(out), "mean_skill": sum(v["skill"] for v in out.values()) / max(1, len(out)),
            "mean_acc": sum(v["acc"] for v in out.values()) / max(1, len(out)), "ece_choice": ece(conf, hit),
            "brier_choice": sum(brier) / max(1, len(brier)), "questions": sum(v["n"] for v in out.values())}
    if a.out:
        json.dump({"summary": summ, "per_benchmark": out}, open(a.out, "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
