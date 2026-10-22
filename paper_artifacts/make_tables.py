#!/usr/bin/env python3
"""LaTeX tables and number macros of the paper, generated from the data files in this repository only.

    python3 paper_artifacts/make_tables.py     # from the repository root; writes build/tables/*.tex (OUT_DIR overrides)

Inputs (read-only):
  board/di-index-20261020.json            Decision Index 0.4 board snapshot (official private scores)
  board/di-index-v0.3.2-20261020.json     archived edition 0.3.2 (v1 models)
  board/di-methodology-20261020.json      areas and gold weights
  stats/gpu1/{s1,btzsc,e4,e6}.json       E1/E2/E4/E6 runs with per-item paired bootstrap CIs
  stats/board_stats.json                 public-vs-private correlations (bootstrap over models)
  btzsc/{published,contamination}.json   published BTZSC baselines; contamination classes
  s1arena/contamination.json             S1-Arena task classes (v2 family)
  results_cpu/prumo/*.json               Prumo on S1-Arena / BTZSC / E4 (CPU, full runs)
  results_gpu2/ev/<arm>_s<seed>/         A1/A2 grid, per-seed score files
  results_gpu3/, results_gpu4/           compute-matched control and initialisation 2x2: score files, compare.json
No number in the generated tables is typed by hand, except the static training facts in tab_models
(layers/hidden, teacher, cost), which come from the model cards.
"""
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                       # repository root
REL = EXP / "board"                     # Decision Index board snapshots (2026-10-20)
OUT = Path(os.environ.get("OUT_DIR", EXP / "build" / "tables"))
OUT.mkdir(parents=True, exist_ok=True)

D04 = json.load(open(REL / "di-index-20261020.json"))
D032 = json.load(open(REL / "di-index-v0.3.2-20261020.json"))
METH = json.load(open(REL / "di-methodology-20261020.json"))
AREAS = [a for a in METH["index"]["areas"] if a.get("weight")]
NGROUPS = max(m["v03"]["rank_tie"] for m in D04["models"] if m.get("v03") and m["v03"].get("rank_tie"))
S1 = json.load(open(EXP / "stats/gpu1/s1.json"))
BT = json.load(open(EXP / "stats/gpu1/btzsc.json"))
E4 = json.load(open(EXP / "stats/gpu1/e4.json"))
E6 = json.load(open(EXP / "stats/gpu1/e6.json"))
BOARD = json.load(open(EXP / "stats/board_stats.json"))
PUB = json.load(open(EXP / "btzsc/published.json"))
S1C = json.load(open(EXP / "s1arena/contamination.json"))
BTC = json.load(open(EXP / "btzsc/contamination.json"))
CAL = json.load(open(EXP / "e5/out/calib.json"))

MACROS = {}


def mac(name, value):
    MACROS[name] = value


def model(engine, data=D04):
    return next(m for m in data["models"] if m["engine"] == engine)


def fmt(x, nd=2, bold=False, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return f"\\textbf{{{s}}}" if bold else s


def ci(c, nd=3, sign=False):
    f = (lambda v: f"{v:+.{nd}f}") if sign else (lambda v: f"{v:.{nd}f}")
    return f"[{f(c[0])}, {f(c[1])}]"


def params(p):
    if p is None:
        return "--"
    return f"{p / 1e9:.2f}B" if p >= 1e9 else f"{p / 1e6:.1f}M"


def esc(s):
    return s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def write(name, text):
    (OUT / name).write_text(text)
    print("wrote", OUT / name)


def private_skill(m, bid):
    b = m["benchmarks"].get(str(bid), {})
    x = b.get("private_eq")
    if x is None:
        return None
    r = b.get("random") or 0.0
    return max(0.0, min(1.0, (x - r) / (1 - r)))


def public_skill(m, bid):
    return m["benchmarks"].get(str(bid), {}).get("skill")


# ===================================================================== Decision Index 0.4 (Table: board)
GROUPS = [
    ("$\\leq$20M", ["crivo-17m", "bekko-v0-17m"]),
    ("32M", ["crivo-32m-v2"]),
    ("68M", ["crivo-68m", "mola-68m", "bekko-v0-68m"]),
    ("150M", ["crivo-150m-v2", "mola-150m", "dinah-1", "tasksource-decider-nano-v4"]),
    ("400M", ["crivo-400m", "amber-0.4b", "decisiontune-1.0", "bekko-v0-400m", "laya-1e28ac20"]),
    ("0.6--1B", ["crivo-1b", "onejev1.6-0.8b", "startlux-decision-v1.1-0.8b-instant", "decision-2.0-kai-0.6b"]),
    ("stock checkpoints served by their entrants (no own training)", ["jeff-uncapped", "harsha"]),
    ("no learned parameters", ["prumo", "zev-latest"]),
]


def find(e):
    try:
        return model(e)
    except StopIteration:
        return None


def row_main(m):
    v = m["v03"]
    lat = m.get("latency") or {}
    ours = m["engine"].startswith("crivo") or m["engine"] == "prumo"
    name = esc(m["name"]).replace(" System One v0", " v0").replace("tasksource Decider nano v4 (150M)",
                                                                   "tasksource nano v4")
    if ours:
        name = f"\\textbf{{{name}}}"
    return " & ".join([
        name, params((m.get("meta") or {}).get("served_params")),
        fmt(v["v"], bold=ours), fmt(v["same_skills"]), fmt(v["new_domains"]), fmt(v["pub"]),
        (f"{lat['median']:.1f} / {lat['p95']:.1f}" if lat.get("p95") is not None else f"{lat['median']:.1f}") if lat.get("median") is not None else "--",
        str(v.get("rank_tie", "--")),
    ]) + " \\\\"


missing = []
lines = ["\\begin{tabular}{@{}llrrrrcr@{}}", "\\toprule",
         "Model & Params & \\textbf{Full 0.4} & Same skills $S$ & New domains $O$ & Public & Latency (ms) & Group \\\\",
         "\\midrule"]
for i, (g, engines) in enumerate(GROUPS):
    ms = [find(e) for e in engines]
    missing += [e for e, m in zip(engines, ms) if m is None]
    ms = [m for m in ms if m is not None]
    if not ms:
        continue
    if i:
        lines.append("\\midrule")
    lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{g}}}}} \\\\")
    for m in ms:
        lines.append(row_main(m))
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_di.tex", "\n".join(lines) + "\n")
if missing:
    print("WARNING: engines not in snapshot:", missing)

# archived 0.3.2 (v1 models)
lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
         "Model (0.3.2) & Full 0.3.2 & $S$ & $O$ & Public & Group \\\\", "\\midrule"]
for lab, e in [("Crivo-32M v1", "crivo-32m"), ("Crivo-150M v1", "crivo-150m")]:
    m = model(e, D032)
    v = m["v03"]
    lines.append(" & ".join([lab, fmt(v["v"]), fmt(v["same_skills"]), fmt(v["new_domains"]), fmt(v["pub"]),
                             str(v.get("rank_tie"))]) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_v1_032.tex", "\n".join(lines) + "\n")

# frontier and the 68M claim (macros)
pts = sorted([(m["meta"]["served_params"], m["v03"]["v"], m) for m in D04["models"]
              if (m.get("meta") or {}).get("served_params") and m.get("v03") and m["v03"].get("v") is not None],
             key=lambda t: t[0])
front, best = [], -1
for p, v, m in pts:
    if v > best:
        best = v
        front.append(m["engine"])
mac("NSized", len(pts))
mac("NFrontierCrivo", sum(1 for e in front if e.startswith("crivo")))
c68 = model("crivo-68m")["v03"]["v"]
first_above = min((p, m["name"], v) for p, v, m in pts if not m["engine"].startswith("crivo") and v > c68)
mac("FirstAboveSixtyEightParams", f"{first_above[0] / 1e6:.1f}M")
mac("FirstAboveSixtyEightName", first_above[1])
mac("NGroups", NGROUPS)
mac("NModels", len(D04["models"]))
small = [m for p, v, m in pts if p <= 420e6]
mac("NSmall", len(small))
mac("NSmallCrivo", sum(1 for m in small if m["engine"].startswith("crivo")))
mac("BoardMaxFull", f"{max(m['v03']['v'] for m in D04['models'] if m.get('v03') and m['v03'].get('v') is not None):.2f}")
# The board maps private scores "onto the public scale with untuned stock models" (methodology JSON) and does not
# document the mapping further. Across the snapshot, the mapped field v03.eq.S is an affine function of the displayed
# same-skill score v03.same_skills; we fit it here so that the paper can state that r does not depend on the mapping.
_rows = [(m["v03"]["same_skills"], m["v03"]["eq"]["S"], m["v03"]["new_domains"], m["v03"]["eq"]["O"], m["v03"]["pub"])
         for m in D04["models"] if m.get("v03") and m["v03"].get("eq") and m["v03"].get("same_skills") is not None
         and m["v03"].get("pub") is not None]


def _fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return b, a, max(abs(y - (a + b * x)) for x, y in zip(xs, ys))


_b, _a, _dev = _fit([r[0] for r in _rows], [r[1] for r in _rows])
mac("MapSSlope", f"{_b:.3f}")
mac("MapSIntercept", f"{_a:.2f}")
mac("MapSMaxDev", f"{_dev:.3f}")
_b, _a, _dev = _fit([r[2] for r in _rows], [r[3] for r in _rows])
mac("MapOSlope", f"{_b:.3f}")
mac("MapOIntercept", f"{_a:.2f}")
mac("MapOMaxDev", f"{_dev:.3f}")
mac("MapN", len(_rows))
for lab, e in [("Seventeen", "crivo-17m"), ("ThirtyTwo", "crivo-32m-v2"), ("SixtyEight", "crivo-68m"),
               ("OneFifty", "crivo-150m-v2"), ("FourHundred", "crivo-400m"), ("OneB", "crivo-1b"), ("Prumo", "prumo")]:
    v = model(e)["v03"]
    mac(f"Full{lab}", f"{v['v']:.2f}")
    mac(f"SS{lab}", f"{v['same_skills']:.2f}")
    mac(f"ND{lab}", f"{v['new_domains']:.2f}")
    mac(f"Pub{lab}", f"{v['pub']:.2f}")

# board correlations
a = BOARD["all"]["pub_vs_same_skills"]
mac("RPubSS", f"{a['pearson']:.3f}")
mac("RPubSSci", ci(a["pearson_ci95"]))
mac("RhoPubSS", f"{a['spearman']:.3f}")
mac("RhoPubSSci", ci(a["spearman_ci95"]))
pl = BOARD["partial_logparams"]["pub_vs_same_skills"]
mac("RPartial", f"{pl['partial_pearson']:.3f}")
mac("RPartialci", ci(pl["partial_pearson_ci95"]))
bk = BOARD["buckets"]
rs = [bk[b]["pub_vs_same_skills"]["pearson"] for b in bk]
mac("RBucketMin", f"{min(rs):.2f}")
mac("RBucketMax", f"{max(rs):.2f}")
mac("NBuckets", len(rs))
v12 = BOARD["crivo_v1_vs_v2"]["150M"]
mac("VOneTwoDiff", f"{v12['mean_diff']:+.3f}")
mac("VOneTwoDiffci", ci(v12["mean_diff_ci95"], sign=True))
mac("VOneTwoWilcoxon", f"{v12['wilcoxon_p']:.2f}")
mac("VOneTwoDown", v12["down"])
mac("VOneTwoUp", v12["up"])
mac("VOneTwoN", v12["n_benchmarks"])

# ===================================================================== S1-Arena (E2)
S1SYS = [
    ("Crivo-17M", "crivo-17m", 16.9e6, True), ("Crivo-32M", "crivo-32m", 32.0e6, True),
    ("Crivo-68M", "crivo-68m", 68.4e6, True), ("Crivo-150M", "crivo-150m", 149.6e6, True),
    ("Crivo-400M", "crivo-400m", 395.8e6, True), ("Crivo-1B", "crivo-1b", 1031.3e6, True),
    ("Crivo-150M v1", "crivo-150m-v1", 149.6e6, True),
    ("DeBERTa-v3-large zs-v2", "nli-deberta-v3-large-zs2", 435e6, False),
    ("BART-large-MNLI", "nli-bart-large-mnli", 407e6, False),
    ("Qwen3-0.6B (letter)", "letter-qwen3-0.6b", 596e6, False),
    ("Gemma-3-1B-it (letter)", "letter-gemma-3-1b", 1000e6, False),
    ("Ettin-32M reranker (untrained)", "reranker-32m-raw", 32e6, False),
    ("Uniform", "uniform", None, False),
]
REF = "nli-deberta-v3-large-zs2"


def pair(lst, a, b):
    for p in lst:
        if p["a"] == a and p["b"] == b:
            return p
        if p["a"] == b and p["b"] == a:
            q = dict(p)
            q["diff"] = -p["diff"]
            q["ci95"] = [-p["ci95"][1], -p["ci95"][0]]
            return q
    return None


lines = ["\\begin{tabular}{@{}lrllrll@{}}", "\\toprule",
         " & & \\multicolumn{3}{c}{Task-mean accuracy} & \\multicolumn{2}{c}{$\\Delta$ vs DeBERTa zs-v2 [task CI]} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(l){6-7}",
         "System & Params & All (209) [item CI] & Unseen (145) [item CI] & Seen (64) & All tasks & Unseen tasks \\\\", "\\midrule"]
for i, (lab, key, p, ours) in enumerate(S1SYS):
    if lab == "DeBERTa-v3-large zs-v2":
        lines.append("\\midrule")
    cells = [f"\\textbf{{{lab}}}" if ours else lab, params(p)]
    for sub in ("all", "unseen"):
        o = S1["item_mode"][sub]["macro"]["own"][key]
        cells.append(f"{o['value']:.3f} {ci(o['ci95'])}")
    cells.append(f"{S1['item_mode']['seen']['macro']['own'][key]['value']:.3f}")
    for sub in ("all", "unseen"):
        if key == REF or not ours:
            cells.append("--")
            continue
        pi = pair(S1["item_mode"][sub]["macro"]["pairs"], key, REF)
        pt = pair(S1["score_mode"][sub]["accuracy"]["pairs"], key, REF)
        star = ("$^\\ast$" if pi["ci95"][0] > 0 or pi["ci95"][1] < 0 else "") + \
               ("$^\\dagger$" if pt["ci95"][0] > 0 or pt["ci95"][1] < 0 else "")
        cells.append(f"{pi['diff']:+.3f}{star} {ci(pt['ci95'], sign=True)}")
    lines.append(" & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_s1.tex", "\n".join(lines) + "\n")


def s1v(key, sub="all"):
    return S1["item_mode"][sub]["macro"]["own"][key]


for lab, key in [("Seventeen", "crivo-17m"), ("ThirtyTwo", "crivo-32m"), ("SixtyEight", "crivo-68m"),
                 ("OneFifty", "crivo-150m"), ("FourHundred", "crivo-400m"), ("OneB", "crivo-1b"),
                 ("Deberta", REF), ("Qwen", "letter-qwen3-0.6b"), ("Gemma", "letter-gemma-3-1b"),
                 ("Bart", "nli-bart-large-mnli"), ("Ettin", "reranker-32m-raw"), ("Uniform", "uniform")]:
    mac(f"SOne{lab}", f"{s1v(key)['value']:.3f}")
    mac(f"SOneU{lab}", f"{s1v(key, 'unseen')['value']:.3f}")
    if key.startswith("crivo"):
        for sub, tag in (("all", ""), ("unseen", "U")):
            pi = pair(S1["item_mode"][sub]["macro"]["pairs"], key, REF)
            pt = pair(S1["score_mode"][sub]["accuracy"]["pairs"], key, REF)
            mac(f"SOneD{tag}{lab}", f"{pi['diff']:+.3f}")
            mac(f"SOneD{tag}{lab}item", ci(pi["ci95"], sign=True))
            mac(f"SOneD{tag}{lab}task", ci(pt["ci95"], sign=True))
for lab, key in [("Qwen", "letter-qwen3-0.6b")]:
    for sub, tag in (("all", ""), ("unseen", "U")):
        pi = pair(S1["item_mode"][sub]["macro"]["pairs"], "crivo-1b", key)
        pt = pair(S1["score_mode"][sub]["accuracy"]["pairs"], "crivo-1b", key)
        mac(f"SOneD{tag}OneBvs{lab}", f"{pi['diff']:+.3f}")
        mac(f"SOneD{tag}OneBvs{lab}item", ci(pi["ci95"], sign=True))
        mac(f"SOneD{tag}OneBvs{lab}task", ci(pt["ci95"], sign=True))
# question-pooled view on unseen tasks (not the benchmark statistic), item CI
for lab, key in [("Seventeen", "crivo-17m"), ("ThirtyTwo", "crivo-32m"), ("SixtyEight", "crivo-68m"),
                 ("OneFifty", "crivo-150m"), ("FourHundred", "crivo-400m"), ("OneB", "crivo-1b")]:
    pp = pair(S1["item_mode"]["unseen"]["pooled"]["pairs"], key, REF)
    mac(f"SOneDUPooled{lab}", f"{pp['diff']:+.3f}")
    mac(f"SOneDUPooled{lab}ci", ci(pp["ci95"], sign=True))
    pt = pair(S1["score_mode"]["unseen"]["accuracy"]["pairs"], key, REF)
    a_better = pt["tasks_a_better"] if pt.get("a") == key else pt["tasks_b_better"]
    b_better = pt["tasks_b_better"] if pt.get("a") == key else pt["tasks_a_better"]
    mac(f"SOneWinU{lab}", f"{a_better}")
    mac(f"SOneLossU{lab}", f"{b_better}")
# Holm adjustment over the 12 Crivo(v2)-vs-DeBERTa comparisons (6 models x all/unseen), task-level bootstrap p-values
# (two-sided, B = 2,000; p = 0 is reported as p < 1/B). Item-level p-values are adjusted the same way.
HOLM = {}
for mode, field in (("task", ("score_mode", "accuracy")), ("item", ("item_mode", "macro"))):
    tests = []
    for lab, key in [("Seventeen", "crivo-17m"), ("ThirtyTwo", "crivo-32m"), ("SixtyEight", "crivo-68m"),
                     ("OneFifty", "crivo-150m"), ("FourHundred", "crivo-400m"), ("OneB", "crivo-1b")]:
        for sub, tag in (("all", ""), ("unseen", "U")):
            pr = pair(S1[field[0]][sub][field[1]]["pairs"], key, REF)
            tests.append((max(pr["p_boot_two_sided"], 1 / 2000), f"{tag}{lab}"))
    tests.sort()
    m_, run = len(tests), 0.0
    for i, (pv, name) in enumerate(tests):
        run = max(run, min(1.0, (m_ - i) * pv))
        HOLM[(mode, name)] = (pv, run)
for (mode, name), (pv, adj) in HOLM.items():
    t = "Task" if mode == "task" else "Item"
    mac(f"Holm{t}{name}", "$<$0.01" if adj < 0.01 else f"{adj:.2f}")
    mac(f"Raw{t}{name}", "$<$0.001" if pv < 0.001 else f"{pv:.3f}")
mac("HolmNTaskSig", sum(1 for (mo, _), (_, adj) in HOLM.items() if mo == "task" and adj < 0.05))
mac("HolmNItemSig", sum(1 for (mo, _), (_, adj) in HOLM.items() if mo == "item" and adj < 0.05))
# Crivo-150M v1 vs v2 on S1-Arena (external corroboration of the private-test drop)
for sub, tag in (("all", ""), ("unseen", "U")):
    pi = pair(S1["item_mode"][sub]["macro"]["pairs"], "crivo-150m", "crivo-150m-v1")
    pt = pair(S1["score_mode"][sub]["accuracy"]["pairs"], "crivo-150m", "crivo-150m-v1")
    mac(f"SOneVTwoMinusVOne{tag}", f"{pi['diff']:+.3f}")
    mac(f"SOneVTwoMinusVOne{tag}item", ci(pi["ci95"], sign=True))
    mac(f"SOneVTwoMinusVOne{tag}task", ci(pt["ci95"], sign=True))
    mac(f"SOneOneFiftyVOne{tag}", f"{s1v('crivo-150m-v1', sub)['value']:.3f}")
    mac(f"SOneOneFiftyVTwo{tag}", f"{s1v('crivo-150m', sub)['value']:.3f}")
mac("SOneUnansweredCrivo", S1["unanswered_questions"]["crivo-17m"])
mac("SOneUnansweredDeberta", S1["unanswered_questions"]["nli-deberta-v3-large-zs2"])
mac("SOneNTasks", S1["n_tasks"]["all"])
mac("SOneNUnseen", S1["n_tasks"]["unseen"])
mac("SOneNSeen", S1["n_tasks"]["seen"])

# Prumo on S1 (CPU run; no CIs)
ps1 = json.load(open(EXP / "results_cpu/prumo/s1.score.json"))
un = [v["accuracy"] for t, v in ps1["tasks"].items() if S1C.get(t, {}).get("class") == "unseen"]
mac("SOnePrumo", f"{ps1['summary']['accuracy']:.3f}")
mac("SOneUPrumo", f"{sum(un) / len(un):.3f}")
pbt = json.load(open(EXP / "results_cpu/prumo/btzsc.score.json"))
mac("BTPrumo", f"{pbt['summary']['overall_macro_f1']:.3f}")

# S1 contamination table by domain
dom = {}
for t, v in S1C.items():
    d = dom.setdefault(v["domain"], {"unseen": 0, "seen-train-split": 0, "seen-test-overlap": 0, "n": 0, "nu": 0})
    dom[v["domain"]][v["class"]] += 1
    dom[v["domain"]]["n"] += v["n"]
    if v["class"] == "unseen":
        dom[v["domain"]]["nu"] += v["n"]

# ===================================================================== BTZSC (E1)
BTSYS = [
    ("Crivo-17M", "crivo-17m", 16.9e6), ("Crivo-32M", "crivo-32m", 32.0e6), ("Crivo-68M", "crivo-68m", 68.4e6),
    ("Crivo-150M", "crivo-150m", 149.6e6), ("Crivo-400M", "crivo-400m", 395.8e6), ("Crivo-1B", "crivo-1b", 1031.3e6),
]
BTBASE = [("Qwen3-0.6B (letter)", "letter-qwen3-0.6b", 596e6), ("Gemma-3-1B-it (letter)", "letter-gemma-3-1b", 1000e6),
          ("Ettin-32M reranker (untrained)", "reranker-32m-raw", 32e6)]
BTSUP = [("DeBERTa-v3-large zs-v2$^\\S$", "nli-deberta-v3-large-zs2", 435e6),
         ("ModernBERT-large zs-v2$^\\S$", "nli-modernbert-large-zs2", 395e6)]
U4 = BT["unseen4"]
PUBSEL = ["e5-mistral-7b-instruct", "Qwen3-Reranker-8B", "Qwen3-Embedding-8B", "gte-reranker-modernbert-base",
          "e5-base-v2", "Qwen3-Reranker-0.6B", "bart-large-mnli", "Qwen3-4B", "deberta-v3-large-nli",
          "ms-marco-MiniLM-L6-v2", "gemma-3-1b-it"]


def btv(key, sub):
    return BT["item_mode"][sub]["macro"]["own"][key]


lines = ["\\begin{tabular}{@{}lrlll@{}}", "\\toprule",
         "System & Params & Unseen-4 [item CI] & Seen-18 & All-22 \\\\", "\\midrule",
         "\\multicolumn{5}{@{}l}{\\textit{This work (v2 family; seen-18 = training split seen)}} \\\\"]
for lab, key, p in BTSYS:
    lines.append(" & ".join([f"\\textbf{{{lab}}}", params(p), f"{btv(key, 'unseen4')['value']:.3f} {ci(btv(key, 'unseen4')['ci95'])}",
                             f"{btv(key, 'seen18')['value']:.3f}", f"{btv(key, 'all22')['value']:.3f}"]) + " \\\\")
lines += ["\\midrule", "\\multicolumn{5}{@{}l}{\\textit{Zero-shot baselines run by us (same adapter)}} \\\\"]
for lab, key, p in BTBASE:
    lines.append(" & ".join([lab, params(p), f"{btv(key, 'unseen4')['value']:.3f} {ci(btv(key, 'unseen4')['ci95'])}",
                             f"{btv(key, 'seen18')['value']:.3f}", f"{btv(key, 'all22')['value']:.3f}"]) + " \\\\")
lines += ["\\midrule", "\\multicolumn{5}{@{}l}{\\textit{Published BTZSC zero-shot baselines (no CI)}} \\\\"]
for k in PUBSEL:
    v = PUB[k]
    u = sum(v["by_dataset"][d] for d in U4) / 4
    s18 = [v["by_dataset"][d] for d in v["by_dataset"] if d not in U4]
    pp = v["params"]
    pp = params(pp) if isinstance(pp, (int, float)) else pp
    lines.append(" & ".join([esc(k), pp, f"{u:.3f}", f"{sum(s18) / len(s18):.3f}", f"{v['overall_macro_f1']:.3f}"]) + " \\\\")
lines += ["\\midrule", "\\multicolumn{5}{@{}l}{\\textit{Supervised upper reference (trained on all 22 datasets' train splits)}} \\\\"]
for lab, key, p in BTSUP:
    lines.append(" & ".join([lab, params(p), f"{btv(key, 'unseen4')['value']:.3f} {ci(btv(key, 'unseen4')['ci95'])}",
                             f"{btv(key, 'seen18')['value']:.3f}", f"{btv(key, 'all22')['value']:.3f}"]) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_btzsc.tex", "\n".join(lines) + "\n")

bestpub = max(((sum(PUB[k]["by_dataset"][d] for d in U4) / 4, k) for k in PUB if PUB[k]["type"] != "base_encoder"))
mac("BTBestPubUnseen", f"{bestpub[0]:.3f}")
mac("BTBestPubName", bestpub[1])
for lab, key in [("Seventeen", "crivo-17m"), ("ThirtyTwo", "crivo-32m"), ("SixtyEight", "crivo-68m"),
                 ("OneFifty", "crivo-150m"), ("FourHundred", "crivo-400m"), ("OneB", "crivo-1b"),
                 ("Qwen", "letter-qwen3-0.6b"), ("Gemma", "letter-gemma-3-1b"), ("Ettin", "reranker-32m-raw"),
                 ("Deberta", "nli-deberta-v3-large-zs2"), ("Modernbert", "nli-modernbert-large-zs2")]:
    mac(f"BTU{lab}", f"{btv(key, 'unseen4')['value']:.3f}")
    mac(f"BTA{lab}", f"{btv(key, 'all22')['value']:.3f}")
p = pair(BT["item_mode"]["unseen4"]["macro"]["pairs"], "crivo-17m", "letter-qwen3-0.6b")
mac("BTDUSeventeenQwen", f"{p['diff']:+.3f}")
mac("BTDUSeventeenQwenci", ci(p["ci95"], sign=True))
cnt = {"unseen": 0, "seen-train-split": 0, "seen-test-overlap": 0}
for d, v in BTC.items():
    cnt[v["v2"]["class"]] += 1
mac("BTNUnseen", cnt["unseen"])
mac("BTNSeenSplit", cnt["seen-train-split"])
mac("BTNSeenOverlap", cnt["seen-test-overlap"])

# contamination summary table
lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         "Benchmark (unit) & Unseen & Seen: split & Seen: overlap & Unseen items \\\\", "\\midrule"]
tot = {"unseen": 0, "seen-train-split": 0, "seen-test-overlap": 0, "n": 0, "nu": 0}
for d in sorted(dom):
    for k in tot:
        tot[k] += dom[d][k]
lines.append(f"S1-Arena (tasks) & {tot['unseen']} & {tot['seen-train-split']} & {tot['seen-test-overlap']} & "
             f"{tot['nu']:,} / {tot['n']:,} \\\\")
nu = sum(v["v2"]["test_items"] for v in BTC.values() if v["v2"]["class"] == "unseen")
na = sum(v["v2"]["test_items"] for v in BTC.values())
lines.append(f"BTZSC (datasets) & {cnt['unseen']} & {cnt['seen-train-split']} & {cnt['seen-test-overlap']} & {nu:,} / {na:,} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_contam.tex", "\n".join(lines) + "\n")
mac("SOneUnseenItems", f"{tot['nu']:,}")
mac("BTUnseenItems", f"{nu:,}")

# ===================================================================== E4 robustness
E4SYS = [("Crivo-17M", "crivo-17m"), ("Crivo-32M", "crivo-32m"), ("Crivo-68M", "crivo-68m"),
         ("Crivo-150M", "crivo-150m"), ("Crivo-400M", "crivo-400m"), ("Crivo-1B", "crivo-1b"),
         ("DeBERTa-v3-large zs-v2", "nli-deberta-v3-large-zs2"), ("Qwen3-0.6B (letter)", "letter-qwen3-0.6b"),
         ("Gemma-3-1B-it (letter)", "letter-gemma-3-1b"), ("Ettin-32M (untrained)", "reranker-32m-raw")]
lines = ["\\begin{tabular}{@{}lllll@{}}", "\\toprule",
         "System & Permutation & Reversed & Yes/no swap & Unsupp. \\\\", "\\midrule"]
for lab, key in E4SYS:
    s = E4["systems"][key]
    fp = s["flip_perm_mean"]
    uns = s["errors"]["orig"]["unsupported"]
    lines.append(" & ".join([f"\\textbf{{{lab}}}" if key.startswith("crivo") else lab,
                             f"{fp['value']:.3f} {ci(fp['ci95'])}", f"{s['rev']['flip_rate']['value']:.3f}",
                             f"{s['yn']['flip_rate']['value']:.3f}", str(uns)]) + " \\\\")
    if key == "crivo-1b":
        lines.append("\\midrule")
lines.append("Prumo (rules) & 0.000 & 0.000 & 0.000 & 0 \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_flips.tex", "\n".join(lines) + "\n")
pf = json.load(open(EXP / "results_cpu/prumo/flips.json"))
assert all(abs(pf[v]["flip_rate"]) < 1e-12 for v in ("perm1", "perm2", "perm3", "rev", "yn") if v in pf), pf
for lab, key in [("Seventeen", "crivo-17m"), ("OneB", "crivo-1b"), ("Deberta", "nli-deberta-v3-large-zs2"),
                 ("Qwen", "letter-qwen3-0.6b"), ("Gemma", "letter-gemma-3-1b"), ("Ettin", "reranker-32m-raw")]:
    mac(f"Flip{lab}", f"{100 * E4['systems'][key]['flip_perm_mean']['value']:.0f}")
mac("EFourRequests", f"{E4['requests']:,}" if isinstance(E4.get("requests"), int) else "2,833")
mac("EFourQuestions", f"{E4['systems']['crivo-17m']['perm1']['questions']:,}")

# ===================================================================== calibration
CALSYS = [("Crivo-17M", "crivo-17m", "c17m"), ("Crivo-32M", "crivo-32m", "c32v2"), ("Crivo-68M", "crivo-68m", "c68m"),
          ("Crivo-150M", "crivo-150m", "c150v2"), ("Crivo-400M", "crivo-400m", "c400m"), ("Crivo-1B", "crivo-1b", "c1b"),
          ("DeBERTa zs-v2", "nli-deberta-v3-large-zs2", None), ("Qwen3-0.6B", "letter-qwen3-0.6b", None),
          ("Gemma-3-1B-it", "letter-gemma-3-1b", None), ("Uniform", "uniform", None)]
ENG = {"c17m": "crivo-17m", "c32v2": "crivo-32m-v2", "c68m": "crivo-68m", "c150v2": "crivo-150m-v2",
       "c400m": "crivo-400m", "c1b": "crivo-1b"}
lines = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
         " & Board & \\multicolumn{3}{c}{S1-Arena} & \\multicolumn{3}{c}{BTZSC} \\\\",
         "\\cmidrule(lr){3-5}\\cmidrule(l){6-8}",
         " & & \\multicolumn{2}{c}{all 209} & unseen & all-22 & \\multicolumn{2}{c}{unseen-4} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-6}\\cmidrule(l){7-8}",
         "System & ECE & ECE & Brier & Brier & ECE & ECE & Brier \\\\", "\\midrule"]
for lab, key, ck in CALSYS:
    board = model(ENG[ck])["calibration"]["ece"] if ck else None
    s_e = S1["item_mode"]["all"]["ece10"]["own"][key]["value"]
    s_b = S1["item_mode"]["all"]["brier"]["own"][key]["value"]
    s_bu = S1["item_mode"]["unseen"]["brier"]["own"][key]["value"]
    if key in BT["item_mode"]["all22"]["ece10"]["own"]:
        b_e = BT["item_mode"]["all22"]["ece10"]["own"][key]["value"]
        b_eu = BT["item_mode"]["unseen4"]["ece10"]["own"][key]["value"]
        b_bu = BT["item_mode"]["unseen4"]["brier"]["own"][key]["value"]
    else:
        b_e = b_eu = b_bu = None
    lines.append(" & ".join([f"\\textbf{{{lab}}}" if key.startswith("crivo") else lab, fmt(board, 3), fmt(s_e, 3),
                             fmt(s_b, 3), fmt(s_bu, 3), fmt(b_e, 3), fmt(b_eu, 3), fmt(b_bu, 3)]) + " \\\\")
    if key == "crivo-1b":
        lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_calib.tex", "\n".join(lines) + "\n")
_cr = ["crivo-17m", "crivo-32m", "crivo-68m", "crivo-150m", "crivo-400m", "crivo-1b"]
_bs = ["nli-deberta-v3-large-zs2", "nli-modernbert-large-zs2", "letter-qwen3-0.6b", "letter-gemma-3-1b", "reranker-32m-raw"]
_u4 = BT["item_mode"]["unseen4"]["ece10"]["own"]
mac("BTUEceCrivoMin", f"{min(_u4[k]['value'] for k in _cr):.3f}")
mac("BTUEceCrivoMax", f"{max(_u4[k]['value'] for k in _cr):.3f}")
mac("BTUEceBaseMin", f"{min(_u4[k]['value'] for k in _bs):.3f}")
mac("BTUEceQwen", f"{_u4['letter-qwen3-0.6b']['value']:.3f}")
mac("BTUEceDeberta", f"{_u4['nli-deberta-v3-large-zs2']['value']:.3f}")
mac("NBTUEceBelowBest", sum(1 for k in _cr if _u4[k]["value"] < min(_u4[b]["value"] for b in _bs)))
mac("BoardEceMin", f"{min(model(e)['calibration']['ece'] for e in ENG.values()):.3f}")
mac("BoardEceMax", f"{max(model(e)['calibration']['ece'] for e in ENG.values()):.3f}")
mac("BoardEcePrumo", f"{model('prumo')['calibration']['ece']:.3f}")
mac("SOneBrierUUniform", f"{S1['item_mode']['unseen']['brier']['own']['uniform']['value']:.3f}")
o = CAL["overall"]
mac("PubAurcSeventeen", f"{o['c17m']['bench']['aurc']:.2f}")
mac("PubAurcOneB", f"{o['c1b']['bench']['aurc']:.2f}")
mac("PubEceSeventeen", f"{o['c17m']['bench']['ece15']:.3f}")
mac("PubEceOneB", f"{o['c1b']['bench']['ece15']:.3f}")

# ===================================================================== E6 latency
# GPU-1 rows are valid only if they are in logs/e6_redo.log; the two first-pass GPU rows (32M read per option, Qwen3)
# are NOT shown. When GPU-3 results exist (G3, see the GPU-3 block below), the re-measured GPU per-option row and the
# same-subset CPU numbers are used instead and \haveGpuLat / \haveCpuLat switch the captions and text.
GPU = {g["name"]: g for g in E6["gpu"]}
CPU = {(c["name"], c["threads"]): c for c in E6["cpu"]}
G3 = Path(os.environ.get("GPU3_DIR", EXP / "results_gpu3"))   # expected GPU-3 layout: ev/<arm>_s<seed>/ (as GPU-2), e6gpu/summary.json, e6cpu/summary.json
FIRST_PASS_GPU = {"crivo-32m-peropt", "letter-qwen06"}


def _load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


E6G3 = _load(G3 / "e6gpu" / "summary.json")
E6C3 = _load(G3 / "e6cpu" / "summary.json")


def g3_pick(summary, prefixes):
    """Mean over passes of the common-subset stats of the first system whose name starts with one of the prefixes."""
    if not summary:
        return None
    for pre in prefixes:
        hits = [v for n, v in summary["systems"].items() if n == pre or n.startswith(pre + ".")]
        if hits:
            out = {}
            for k in ("median_ms", "mean_ms", "p95_ms"):
                vals = [h["common_subset"].get(k) for h in hits if h["common_subset"].get(k) is not None]
                out[k] = sum(vals) / len(vals) if vals else None
            out["n"] = hits[0]["common_subset"].get("n")
            return out
    return None


LATSYS = [("Crivo-17M", "crivo-17m", "crivo-17m", ["crivo-17m"]),
          ("Crivo-32M", "crivo-32m", "crivo-32m-v2", ["crivo-32m-v2", "crivo-32m"]),
          ("Crivo-68M", "crivo-68m", "crivo-68m", ["crivo-68m"]),
          ("Crivo-150M", "crivo-150m", "crivo-150m-v2", ["crivo-150m"]),
          ("Crivo-400M", "crivo-400m", "crivo-400m", ["crivo-400m"]),
          ("Crivo-1B", "crivo-1b", "crivo-1b", ["crivo-1b"]),
          ("32M per option (P)", "crivo-32m-peropt", None, ["crivo-32m-per-option", "crivo-32m-v2-read-per-option", "P_s0"]),
          ("DeBERTa zs-v2", "nli-deberta", None, ["nli-deberta-v3-large-zs2", "nli-deberta"]),
          ("Ettin-32M untrained", "reranker-32m", None, ["reranker-32m"]),
          ("Prumo (rules)", "prumo", "prumo", [])]
# GPU-3 names (results_gpu3, 2026-10-21): GPU block crivo-32m-v2.{a,b}, P_s0-request-batched.{a,b} (per-option
# checkpoint P seed 0, all options of a request in shared batches, as on the CPU) and P_s0-per-question.{a,b};
# CPU block crivo-{17m,...,1b}, crivo-32m-per-option (= P_s0 request-batched), reranker-32m, nli-deberta-v3-large-zs2.
GPU_PO = ["P_s0-request-batched", "crivo-32m-v2-read-per-option"]
GPU_PO_Q = ["P_s0-per-question"]
HAVE_GPU_LAT = g3_pick(E6G3, GPU_PO) is not None and g3_pick(E6G3, ["crivo-32m-v2"]) is not None
HAVE_CPU_LAT = (E6C3 is not None and g3_pick(E6C3, ["nli-deberta-v3-large-zs2", "nli-deberta"]) is not None
                and g3_pick(E6C3, ["crivo-17m"]) is not None)


def thou(x):
    return "--" if x is None else f"{x:,.0f}"


def lat(d, k="median_ms"):
    return None if d is None else d.get(k)


if HAVE_CPU_LAT:   # GPU-3: same 80 requests for every system on one CPU; the old 1-thread column (other VM) is dropped
    lines = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{A100 GPU, ms} & \\multicolumn{3}{c}{CPU, 16 threads, ms} & Board \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "System & med. & mean & p95 & med. & mean & p95 & med. (ms) \\\\", "\\midrule"]
else:
    lines = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{A100, ms} & \\multicolumn{2}{c}{CPU 16 thr., ms} & CPU 1 thr. & Board \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
             "System & med. & mean & p95 & med. & mean & med. (ms) & med. (ms) \\\\", "\\midrule"]
for lab, key, eng, g3names in LATSYS:
    g = GPU.get(key)
    if key in FIRST_PASS_GPU:
        g = None
    if key == "crivo-32m-peropt" and HAVE_GPU_LAT:
        g = g3_pick(E6G3, GPU_PO)
    c16, c1 = CPU.get((key, 16)), CPU.get((key, 1))
    if HAVE_CPU_LAT:   # never mix the old (different-subset) CPU runs with the same-subset GPU-3 runs
        c16 = g3_pick(E6C3, g3names) if g3names else None
    bl = (model(eng).get("latency") or {}).get("median") if eng else None
    if HAVE_CPU_LAT:
        cells = [lab, fmt(lat(g), 1), fmt(lat(g, "mean_ms"), 1), fmt(lat(g, "p95_ms"), 1),
                 fmt(lat(c16), 1), thou(lat(c16, "mean_ms")), thou(lat(c16, "p95_ms")), fmt(bl, 1)]
    else:
        cells = [lab, fmt(lat(g), 1), fmt(lat(g, "mean_ms"), 1), fmt(lat(g, "p95_ms"), 1),
                 fmt(lat(c16), 1), fmt(lat(c16, "mean_ms"), 0), fmt(lat(c1), 1), fmt(bl, 1)]
    lines.append(" & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_latency.tex", "\n".join(lines) + "\n")
mac("GpuSeventeen", f"{GPU['crivo-17m']['median_ms']:.1f}")
mac("GpuOneB", f"{GPU['crivo-1b']['median_ms']:.1f}")
mac("GpuFourHundred", f"{GPU['crivo-400m']['median_ms']:.1f}")
mac("GpuDeberta", f"{GPU['nli-deberta']['median_ms']:.1f}")
mac("GpuDebertaMean", f"{GPU['nli-deberta']['mean_ms']:.1f}")
mac("GpuDebertaSkipped", f"{GPU['nli-deberta'].get('unsupported', 0)}")
mac("GpuSeventeenMean", f"{GPU['crivo-17m']['mean_ms']:.1f}")
mac("GpuSeventeenThroughput", f"{GPU['crivo-17m']['throughput_req_s']:.0f}")
mac("GpuDebertaThroughput", f"{GPU['nli-deberta']['throughput_req_s']:.1f}")
mac("CpuPrumo", f"{CPU[('prumo', 1)]['median_ms']:.1f}")
mac("CpuSeventeenOne", f"{CPU[('crivo-17m', 1)]['median_ms']:.1f}")
if HAVE_CPU_LAT:   # every CPU macro below then comes from the GPU-3 same-subset run (e6cpu/summary.json)
    _c17, _c150 = g3_pick(E6C3, ["crivo-17m"]), g3_pick(E6C3, ["crivo-150m"])
    _cd = g3_pick(E6C3, ["nli-deberta-v3-large-zs2", "nli-deberta"])
    _cj = g3_pick(E6C3, ["crivo-32m-v2", "crivo-32m"])
    _cp = g3_pick(E6C3, ["crivo-32m-per-option", "crivo-32m-v2-read-per-option", "P_s0"])
    _cr = g3_pick(E6C3, ["reranker-32m"])
    mac("CpuN", f"{_c17['n']}")
    mac("CpuSkippedTotal", f"{sum(v.get('skipped_unsupported', 0) for v in E6C3['systems'].values())}")
    mac("CpuOneFifty", f"{_c150['median_ms']:.1f}")
    mac("CpuRerankerMedian", f"{_cr['median_ms']:.1f}")
    mac("CpuRerankerMean", thou(_cr['mean_ms']))
else:
    _c17 = CPU[('crivo-17m', 16)]
    _cd = CPU[('nli-deberta', 16)]
    _cj, _cp = CPU[('crivo-32m', 16)], CPU[('crivo-32m-peropt', 16)]
mac("CpuSeventeen", f"{_c17['median_ms']:.1f}")
mac("CpuSeventeenMean", f"{_c17['mean_ms']:.0f}")
mac("CpuDeberta", f"{_cd['median_ms']:.0f}")
mac("CpuDebertaMean", f"{_cd['mean_ms']:,.0f}")
mac("CpuDebertaRatio", f"{_cd['median_ms'] / _c17['median_ms']:.0f}")
mac("CpuJointMean", f"{_cj['mean_ms']:.0f}")
mac("CpuPeroptMean", f"{_cp['mean_ms']:,.0f}")
mac("CpuJointMedian", f"{_cj['median_ms']:.1f}")
mac("CpuPeroptMedian", f"{_cp['median_ms']:.1f}")
mac("CpuPeroptMedianRatio", f"{_cp['median_ms'] / _cj['median_ms']:.1f}")
mac("CpuPeroptMeanRatio", f"{_cp['mean_ms'] / _cj['mean_ms']:.0f}")
mac("CpuPeroptPNineFiveRatio", f"{_cp['p95_ms'] / _cj['p95_ms']:.0f}")

# ===================================================================== A1/A2 grid (per-seed)
G2 = EXP / "results_gpu2"
ARMS = [("J", "J (reference)"), ("Jsteps", "J, steps matched to P"), ("P", "P (per-option)"),
        ("JnoBrier", "J without Brier"), ("JnoKD", "J without distillation"), ("Jraw", "J from raw Ettin encoder")]


def seed_metrics(ev):
    m = {}
    p = json.load(open(ev / "pub15.score.json"))["summary"]
    m["pub15_skill"] = p["mean_skill"]
    m["pub15_ece"] = p["ece_choice"]
    b = json.load(open(ev / "btzsc500.score.json"))
    m["btzsc_f1"] = b["summary"]["overall_macro_f1"]
    u = [b["by_dataset"][d]["macro_f1"] for d in U4]
    m["btzsc_unseen_f1"] = sum(u) / len(u)
    s = json.load(open(ev / "s1_60.score.json"))
    m["s1_acc"] = s["summary"]["accuracy"]
    m["s1_brier"] = s["summary"]["brier"]
    un = [v["accuracy"] for t, v in s["tasks"].items() if S1C.get(t, {}).get("class") == "unseen"]
    m["s1_unseen_acc"] = sum(un) / len(un)
    fp = ev / "e4" / "flips.json"   # GPU-4 runs have no E4 (flips): None
    if fp.exists():
        f = json.load(open(fp))
        m["flip_perm"] = sum(f[v]["flip_rate"] for v in ("perm1", "perm2", "perm3")) / 3
    else:
        m["flip_perm"] = None
    return m


SEEDS = {}
for a, _ in ARMS:
    SEEDS[a] = [seed_metrics(G2 / "ev" / f"{a}_s{s}") for s in range(3)]
# GPU-3 arms (results_gpu3/ev/, same file layout as GPU-2): Jcomp = J at P's optimizer steps, LR schedule and per-step
# token budget (3 seeds); Praw = P from the raw Ettin-32M encoder (1 seed). Added to the table/figure only if present.
jc_dirs = [G3 / "ev" / f"Jcomp_s{k}" for k in range(3)]
HAVE_JCOMP = all((d / "pub15.score.json").exists() and (d / "e4" / "flips.json").exists() for d in jc_dirs)
# GPU-4 (results_gpu4/ev/, README 18.y): the initialisation 2x2. Praw pools GPU-3 s0 with GPU-4 s1, s2; JrawComp =
# J-comp's flags from the raw Ettin-32M encoder (3 seeds). GPU-4 ran no E4, so their flips are None.
G4 = Path(os.environ.get("GPU4_DIR", EXP / "results_gpu4"))
pr_dirs = [G3 / "ev" / "Praw_s0", G4 / "ev" / "Praw_s1", G4 / "ev" / "Praw_s2"]
pr_dirs = [d for d in pr_dirs if (d / "pub15.score.json").exists()]
HAVE_PRAW = bool(pr_dirs)
jrc_dirs = [G4 / "ev" / f"JrawComp_s{k}" for k in range(3)]
HAVE_JRAWCOMP = all((d / "pub15.score.json").exists() for d in jrc_dirs)
HAVE_2X2 = HAVE_JCOMP and HAVE_JRAWCOMP and len(pr_dirs) == 3
if HAVE_JCOMP:
    SEEDS["Jcomp"] = [seed_metrics(d) for d in jc_dirs]
if HAVE_PRAW:
    SEEDS["Praw"] = [seed_metrics(d) for d in pr_dirs]
if HAVE_JRAWCOMP:
    SEEDS["JrawComp"] = [seed_metrics(d) for d in jrc_dirs]
json.dump(SEEDS, open(OUT / "a1a2_seeds.json", "w"), indent=1)
TAB_ARMS = ARMS[:2] + ([("Jcomp", "J, compute matched to P (J-comp)")] if HAVE_JCOMP else []) + ARMS[2:3] + \
    ([("JrawComp", "J-comp from raw encoder (J-comp-raw)")] if HAVE_JRAWCOMP else []) + \
    ([("Praw", "P from raw encoder (P-raw)" + ("" if len(pr_dirs) == 3 else "$^\\dagger$"))] if HAVE_PRAW else []) + \
    ARMS[3:5] + [("Jraw", "J from raw encoder")]


def mean_sd(v):
    n = len(v)
    mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))
    return mu, sd


def welch(a, b):
    """Welch t 95% CI for mean(a) - mean(b)."""
    from scipy import stats as st
    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    na, nb = len(a), len(b)
    se2 = sa ** 2 / na + sb ** 2 / nb
    se = math.sqrt(se2)
    df = se2 ** 2 / ((sa ** 2 / na) ** 2 / (na - 1) + (sb ** 2 / nb) ** 2 / (nb - 1)) if se > 0 else 1e9
    t = st.t.ppf(0.975, df) if se > 0 else 0.0
    d = ma - mb
    return d, (d - t * se, d + t * se)


MET = [("pub15_skill", "pub15 skill", 3), ("s1_acc", "S1 acc.", 3), ("s1_unseen_acc", "S1 unseen", 3),
       ("btzsc_f1", "BTZSC F1", 3), ("btzsc_unseen_f1", "BTZSC unseen-4", 3), ("s1_brier", "S1 Brier", 3),
       ("pub15_ece", "pub15 ECE", 3), ("flip_perm", "Perm. flips", 3)]
NM_ = {"pub15_skill": "Skill", "s1_acc": "SOne", "s1_unseen_acc": "SOneU", "btzsc_f1": "BT", "btzsc_unseen_f1": "BTU",
       "s1_brier": "SOneBrier", "pub15_ece": "Ece", "flip_perm": "Flip"}
SHOW = ["pub15_skill", "s1_acc", "s1_unseen_acc", "btzsc_f1", "btzsc_unseen_f1", "s1_brier", "pub15_ece", "flip_perm"]
lines = ["\\begin{tabular}{@{}l" + "r" * len(MET) + "@{}}", "\\toprule",
         "Arm & " + " & ".join(m[1] for m in MET) + " \\\\", "\\midrule",
         f"\\multicolumn{{{len(MET) + 1}}}{{@{{}}l}}{{\\textit{{Mean $\\pm$ s.d. over 3 seeds}}}} \\\\"]
for a, lab in TAB_ARMS:
    cells = [lab]
    for k, _, nd in MET:
        vv = [s[k] for s in SEEDS[a] if s[k] is not None]
        if not vv:   # not measured (GPU-4 runs: no E4)
            cells.append("--")
            continue
        if len(vv) == 1:   # one seed: value only
            cells.append(f"{vv[0]:.{nd}f}")
            continue
        mu, sd = mean_sd(vv)
        cells.append(f"{mu:.{nd}f}$\\pm${sd:.{nd}f}" if k != "flip_perm" else f"{mu:.3f}$\\pm${sd:.3f}")
    lines.append(" & ".join(cells) + " \\\\")
# Released Crivo-32M (full recipe, joint) on the same evaluation subsets: its GPU-1 S1/BTZSC outputs and its released
# public-suite run restricted to the subset ids, official scorers (exp/a1a2/released_on_subsets.py). pub15 is on the
# 12,833 of 13,358 requests present in the released run ("pub15c"; P on the same requests in the text macros).
RS = json.load(open(G3 / "analysis" / "released_subsets.json")) if (G3 / "analysis" / "released_subsets.json").exists() else None
if RS:
    r32 = RS["models"]["crivo-32m"]
    fl32 = E4["systems"]["crivo-32m"]["flip_perm_mean"]["value"]
    lines.append(" & ".join(["Crivo-32M released (joint, full recipe)$^\\ddagger$", f"{r32['pub15c_skill']:.3f}",
                             f"{r32['s1_acc']:.3f}", f"{r32['s1_unseen_acc']:.3f}", f"{r32['btzsc_f1']:.3f}",
                             f"{r32['btzsc_unseen_f1']:.3f}", f"{r32['s1_brier']:.3f}", "--", f"{fl32:.3f}"]) + " \\\\")
    pc = RS["arms_pub15c_skill"]["P"]
    mac("RelThirtyTwoPubc", f"{r32['pub15c_skill']:.3f}")
    mac("PPubc", f"{sum(pc) / len(pc):.3f}")
    for k, nm in (("s1_acc", "SOne"), ("s1_unseen_acc", "SOneU"), ("btzsc_f1", "BT"), ("btzsc_unseen_f1", "BTU")):
        mac(f"RelThirtyTwo{nm}", f"{r32[k]:.3f}")
        mac(f"RelSixtyEight{nm}", f"{RS['models']['crivo-68m'][k]:.3f}")
        pv = [x[k] for x in SEEDS["P"]]
        mac(f"PMean{nm}", f"{sum(pv) / len(pv):.3f}")
        mac(f"PMin{nm}", f"{min(pv):.3f}"); mac(f"PMax{nm}", f"{max(pv):.3f}")
    mac("RelSixtyEightPubc", f"{RS['models']['crivo-68m']['pub15c_skill']:.3f}")
    mac("PubcN", f"{RS['checks']['pub15_common_requests']:,}")
lines += ["\\midrule", f"\\multicolumn{{{len(MET) + 1}}}{{@{{}}l}}{{\\textit{{Differences, Welch 95\\% CI over seeds (3 vs 3)}}}} \\\\"]
CMP = ([("P", "Jcomp", "P $-$ J-comp (A1, matched compute)")] if HAVE_JCOMP else []) + \
      [("P", "Jsteps", "P $-$ J-steps (A1, matched steps)")] + \
      ([("Jcomp", "Jsteps", "J-comp $-$ J-steps")] if HAVE_JCOMP else []) + \
      [("Jsteps", "J", "J-steps $-$ J"),
       ("JnoBrier", "J", "no Brier $-$ J"), ("JnoKD", "J", "no KD $-$ J"), ("Jraw", "J", "raw init $-$ J")]
for a, b, lab in CMP:
    cells = [lab]
    for k, _, nd in MET:
        d, c = welch([s[k] for s in SEEDS[a]], [s[k] for s in SEEDS[b]])
        sig = c[0] > 0 or c[1] < 0
        cells.append("\\shortstack{" + ("\\textbf{" if sig else "") + f"{d:+.3f}" + ("}" if sig else "") +
                     f"\\\\{{\\tiny[{c[0]:+.3f}, {c[1]:+.3f}]}}}}")
        if a == "P" and b == "Jsteps":
            nm = {"pub15_skill": "Skill", "s1_acc": "SOne", "s1_unseen_acc": "SOneU", "btzsc_f1": "BT",
                  "btzsc_unseen_f1": "BTU", "s1_brier": "SOneBrier", "pub15_ece": "Ece", "flip_perm": "Flip"}[k]
            mac(f"AOne{nm}", f"{d:+.3f}")
            mac(f"AOne{nm}ci", f"[{c[0]:+.3f}, {c[1]:+.3f}]")
        if b == "J" and a in ("JnoBrier", "JnoKD", "Jraw"):
            nm = {"pub15_skill": "Skill", "s1_acc": "SOne", "s1_unseen_acc": "SOneU", "btzsc_f1": "BT",
                  "btzsc_unseen_f1": "BTU", "s1_brier": "SOneBrier", "pub15_ece": "Ece", "flip_perm": "Flip"}[k]
            mac(f"ATwo{a}{nm}", f"{d:+.3f}")
            mac(f"ATwo{a}{nm}ci", f"[{c[0]:+.3f}, {c[1]:+.3f}]")
    if (a, b) in (("P", "Jcomp"), ("P", "Jsteps")):   # J-comp - J-steps and the A2 rows (all null) are in the text only
        lines.append(" & ".join(cells) + " \\\\")


def welch4(a, b, c, d):
    """(mean a - mean b) - (mean c - mean d), Welch-Satterthwaite df over the four cells' variances (README 18.y.1 C4)."""
    from scipy import stats as st
    parts = [mean_sd(x) + (len(x),) for x in (a, b, c, d)]
    diff = (parts[0][0] - parts[1][0]) - (parts[2][0] - parts[3][0])
    v = [sd ** 2 / n for _, sd, n in parts]
    se2 = sum(v)
    df = se2 ** 2 / sum(vi ** 2 / (n - 1) for vi, (_, _, n) in zip(v, parts))
    t = st.t.ppf(0.975, df)
    return diff, (diff - t * math.sqrt(se2), diff + t * math.sqrt(se2))


def ci_cell(d, c):
    sig = c[0] > 0 or c[1] < 0
    return ("\\shortstack{" + ("\\textbf{" if sig else "") + f"{d:+.3f}" + ("}" if sig else "") +
            f"\\\\{{\\tiny[{c[0]:+.3f}, {c[1]:+.3f}]}}}}")


TT = {}   # 2x2 contrasts (README 18.y.1): C1 Praw-JrawComp, C2 P-Praw, C3 Jcomp-JrawComp, C4 interaction
if HAVE_2X2:
    lines += ["\\midrule", f"\\multicolumn{{{len(MET) + 1}}}{{@{{}}l}}{{\\textit{{Initialisation $2\\times2$ (decision rule "
              "fixed before the run), Welch 95\\% CI}" + "} \\\\"]
    for cn, lab, spec in (("Cone", "P-raw $-$ J-comp-raw (C1, raw init)", ("Praw", "JrawComp")),
                          ("Ctwo", "P $-$ P-raw (C2, reranker gain, per option)", ("P", "Praw")),
                          ("Cthree", "J-comp $-$ J-comp-raw (C3, reranker gain, joint)", ("Jcomp", "JrawComp")),
                          ("Cfour", "Interaction C2 $-$ C3 (C4)", ("P", "Praw", "Jcomp", "JrawComp"))):
        cells = [lab]
        for k, _, nd in MET:
            if k == "flip_perm":
                cells.append("--")
                continue
            xs = [[s[k] for s in SEEDS[x]] for x in spec]
            d, c = welch(*xs) if len(spec) == 2 else welch4(*xs)
            TT[f"{cn}{NM_[k]}"] = f"{d:+.3f}"
            TT[f"{cn}{NM_[k]}ci"] = f"[{c[0]:+.3f}, {c[1]:+.3f}]"
            cells.append(ci_cell(d, c))
        lines.append(" & ".join(cells) + " \\\\")
    for k, _, _ in MET:   # secondary: compute effect at raw init (JrawComp - Jraw), descriptive
        if k == "flip_perm":
            continue
        d, c = welch([s[k] for s in SEEDS["JrawComp"]], [s[k] for s in SEEDS["Jraw"]])
        TT[f"JrawCompMinusJraw{NM_[k]}"] = f"{d:+.3f}"
        TT[f"JrawCompMinusJraw{NM_[k]}ci"] = f"[{c[0]:+.3f}, {c[1]:+.3f}]"
    for a_ in ("Praw", "JrawComp"):
        for k, _, _ in MET:
            if k == "flip_perm":
                continue
            mu, sd = mean_sd([s[k] for s in SEEDS[a_]])
            TT[f"{a_}Mean{NM_[k]}"] = f"{mu:.3f}"
    _c4 = _load(G4 / "compare.json") or {}
    _tr4 = _c4.get("training_runs", {})
    TT["OomGpuFour"] = f"{sum(v.get('oom_skips', 0) for v in _tr4.values())}"
    TT["TwoTwoN"] = "3"
for k, v in TT.items():
    mac(k, v)
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_ablation.tex", "\n".join(lines) + "\n")
for a, _ in ARMS:
    mu, _ = mean_sd([s["pub15_skill"] for s in SEEDS[a]])
    mac(f"ASkill{a}", f"{mu:.3f}")

# ===================================================================== GPU-3 (pending): Jcomp, Praw, latency
# Jcomp = joint reader trained to (about) P's compute on the same 150k rows, 3 seeds; Praw = P from the raw Ettin-32M
# encoder, 1 seed. Expected layout: G3/ev/Jcomp_s{0,1,2}/ and G3/ev/Praw_s0/ with the GPU-2 file names
# (pub15/s1_60/btzsc500 score files, e4/flips.json). Until they exist every value macro is \pendingval and the
# \ifhave... switches are false, so the sentences wrapped in \whenJcomp{...} etc. print nothing in final mode.
PEND = {}
PSW = {}
PSW["Jcomp"] = HAVE_JCOMP
NM = {"pub15_skill": "Skill", "s1_acc": "SOne", "s1_unseen_acc": "SOneU", "btzsc_f1": "BT", "btzsc_unseen_f1": "BTU",
      "s1_brier": "SOneBrier", "pub15_ece": "Ece", "flip_perm": "Flip"}
if HAVE_JCOMP:
    for k, _, nd in MET:
        mu, _ = mean_sd([x[k] for x in SEEDS["Jcomp"]])
        PEND[f"Jcomp{NM[k]}"] = f"{mu:.3f}"
        d, c = welch([x[k] for x in SEEDS["P"]], [x[k] for x in SEEDS["Jcomp"]])
        PEND[f"AOneComp{NM[k]}"] = f"{d:+.3f}"
        PEND[f"AOneComp{NM[k]}ci"] = f"[{c[0]:+.3f}, {c[1]:+.3f}]"
        PEND[f"AOneComp{NM[k]}sig"] = "1" if (c[0] > 0 or c[1] < 0) else "0"
        d, c = welch([x[k] for x in SEEDS["Jcomp"]], [x[k] for x in SEEDS["J"]])
        PEND[f"JcompMinusJ{NM[k]}"] = f"{d:+.3f}"
        PEND[f"JcompMinusJ{NM[k]}ci"] = f"[{c[0]:+.3f}, {c[1]:+.3f}]"
else:
    for k, _, _ in MET:
        for f in (f"Jcomp{NM[k]}", f"AOneComp{NM[k]}", f"AOneComp{NM[k]}ci", f"JcompMinusJ{NM[k]}", f"JcompMinusJ{NM[k]}ci"):
            PEND[f] = None
PSW["Praw"] = HAVE_PRAW
if HAVE_PRAW:
    for k, _, _ in MET:
        vv = [x[k] for x in SEEDS["Praw"] if x[k] is not None]
        pmk = sum(vv) / len(vv)
        PEND[f"Praw{NM[k]}"] = f"{pmk:.3f}"
        PEND[f"PrawMinusP{NM[k]}"] = f"{pmk - mean_sd([x[k] for x in SEEDS['P']])[0]:+.3f}"
else:
    for k, _, _ in MET:
        PEND[f"Praw{NM[k]}"] = PEND[f"PrawMinusP{NM[k]}"] = None
# re-measured latency
PSW["GpuLat"] = HAVE_GPU_LAT
PSW["CpuLat"] = HAVE_CPU_LAT
gj, gp, gq = g3_pick(E6G3, ["crivo-32m-v2"]), g3_pick(E6G3, GPU_PO), g3_pick(E6G3, GPU_PO_Q)
for nm_, d_ in (("GpuThreeJoint", gj), ("GpuThreePeropt", gp), ("GpuThreePeroptQ", gq)):
    for k, suf in (("median_ms", "Median"), ("mean_ms", "Mean"), ("p95_ms", "PNineFive")):
        PEND[nm_ + suf] = f"{d_[k]:.1f}" if (HAVE_GPU_LAT and d_ and d_.get(k) is not None) else None
PEND["GpuThreeN"] = f"{gj['n']:,}" if HAVE_GPU_LAT else None
PEND["GpuThreeMedianRatio"] = f"{gp['median_ms'] / gj['median_ms']:.2f}" if HAVE_GPU_LAT else None
PEND["GpuThreeMeanRatio"] = f"{gp['mean_ms'] / gj['mean_ms']:.1f}" if HAVE_GPU_LAT else None
PEND["GpuThreePNineFiveRatio"] = f"{gp['p95_ms'] / gj['p95_ms']:.0f}" if HAVE_GPU_LAT else None
PEND["GpuThreeMeanRatioQ"] = f"{gq['mean_ms'] / gj['mean_ms']:.1f}" if (HAVE_GPU_LAT and gq) else None
PEND["GpuThreePNineFiveRatioQ"] = f"{gq['p95_ms'] / gj['p95_ms']:.0f}" if (HAVE_GPU_LAT and gq) else None
cs, cd = g3_pick(E6C3, ["crivo-17m"]), g3_pick(E6C3, ["nli-deberta-v3-large-zs2", "nli-deberta"])
PEND["CpuSameSeventeenMedian"] = f"{cs['median_ms']:.1f}" if HAVE_CPU_LAT else None
PEND["CpuSameDebertaMedian"] = f"{cd['median_ms']:.0f}" if HAVE_CPU_LAT else None
PEND["CpuSameSeventeenMean"] = f"{cs['mean_ms']:.0f}" if HAVE_CPU_LAT else None
PEND["CpuSameDebertaMean"] = f"{cd['mean_ms']:,.0f}" if HAVE_CPU_LAT else None
PEND["CpuSameRatio"] = f"{cd['median_ms'] / cs['median_ms']:.0f}" if HAVE_CPU_LAT else None
PEND["CpuSameN"] = f"{cs['n']}" if HAVE_CPU_LAT else None
# training tokens logged by train_ablate.py (GPU-3 only; GPU-2's P did not log them), from compare.json
_cmp = _load(G3 / "compare.json") or {}
_tr = _cmp.get("training_runs", {})
_tj = [_tr[f"Jcomp_s{k}"]["tokens_total"] for k in range(3) if f"Jcomp_s{k}" in _tr]
_tp = _tr.get("Praw_s0", {}).get("tokens_total")
PEND["TokJcomp"] = f"{sum(_tj) / len(_tj) / 1e6:.1f}" if _tj else None
PEND["TokPraw"] = f"{_tp / 1e6:.1f}" if _tp else None
PEND["TokRatio"] = f"{_tp / (sum(_tj) / len(_tj)):.2f}" if (_tj and _tp) else None
PEND["OomJcompPraw"] = f"{sum(v.get('oom_skips', 0) for v in _tr.values())}" if _tr else None
# Jcomp - Jsteps and Praw - Jraw (descriptive)
if HAVE_JCOMP:
    for k, _, _ in MET:
        d, c = welch([x[k] for x in SEEDS["Jcomp"]], [x[k] for x in SEEDS["Jsteps"]])
        PEND[f"JcompMinusJsteps{NM[k]}"] = f"{d:+.3f}"
        PEND[f"JcompMinusJsteps{NM[k]}ci"] = f"[{c[0]:+.3f}, {c[1]:+.3f}]"
if HAVE_PRAW:   # reranker-initialisation gain at J's budget (descriptive; the 3-seed 2x2 contrasts are the \C* macros)
    PEND["InitGainJ"] = f"{mean_sd([x['pub15_skill'] for x in SEEDS['J']])[0] - mean_sd([x['pub15_skill'] for x in SEEDS['Jraw']])[0]:.3f}"
ptxt = ["% generated by tables/make_tables.py -- GPU-3 switches and numbers"]
for sw, on in PSW.items():
    ptxt.append(f"\\newif\\ifhave{sw} \\have{sw}{'true' if on else 'false'}")
    # \when<sw>{text}: the text when the result exists; otherwise a \pending marker (nothing in final mode)
    ptxt.append(f"\\newcommand{{\\when{sw}}}[1]{{\\ifhave{sw}#1\\else\\pending{{#1}}\\fi}}")
for k, v in PEND.items():
    ptxt.append(f"\\newcommand{{\\{k}}}{{{v if v is not None else chr(92) + 'pendingval'}}}")
write("pending.tex", "\n".join(ptxt) + "\n")
print("GPU-3 switches:", PSW)

# ===================================================================== v1 vs v2 (selection) and per-benchmark (appendix)
PANELS = {  # internal panels (model cards) -- not board numbers
    "150M v1": (0.459, 0.150), "150M v2": (0.556, 0.224), "32M v1": (0.390, 0.019), "32M v2": (0.453, 0.126)}
cols = [("150M v1", model("crivo-150m", D032), "0.3.2"), ("150M v2", model("crivo-150m-v2"), "0.4"),
        ("32M v1", model("crivo-32m", D032), "0.3.2"), ("32M v2", model("crivo-32m-v2"), "0.4")]
lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", " & " + " & ".join(c[0] for c in cols) + " \\\\", "\\midrule",
         "SS-core panel (internal) & " + " & ".join(f"{PANELS[c[0]][0]:.3f}" for c in cols) + " \\\\",
         "ND-proxy panel (internal) & " + " & ".join(f"{PANELS[c[0]][1]:.3f}" for c in cols) + " \\\\",
         "Public index (self-run) & " + " & ".join(f"{c[1]['v03']['pub']:.2f}" for c in cols) + " \\\\",
         "\\textbf{Private same skills (board)} & " + " & ".join(f"\\textbf{{{c[1]['v03']['same_skills']:.2f}}}" for c in cols) + " \\\\",
         "Private new domains (board) & " + " & ".join(f"{c[1]['v03']['new_domains']:.2f}$^{{{c[2]}}}$" for c in cols) + " \\\\",
         "\\bottomrule", "\\end{tabular}"]
write("tab_v1v2.tex", "\n".join(lines) + "\n")
mac("SSOneFiftyVOne", f"{model('crivo-150m', D032)['v03']['same_skills']:.2f}")
mac("PubOneFiftyVOne", f"{model('crivo-150m', D032)['v03']['pub']:.2f}")
mac("FullOneFiftyVOne", f"{model('crivo-150m', D032)['v03']['v']:.2f}")
mac("FullThirtyTwoVOne", f"{model('crivo-32m', D032)['v03']['v']:.2f}")


def deltas(e1, e2):
    m1, m2 = model(e1, D032), model(e2)
    out = []
    for a in AREAS:
        for b in a["panel"]:
            p1, p2 = private_skill(m1, b["id"]), private_skill(m2, b["id"])
            q1, q2 = public_skill(m1, b["id"]), public_skill(m2, b["id"])
            if None in (p1, p2, q1, q2):
                continue
            out.append((b["short"], p1, p2, q1, q2))
    return out


def corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


d150 = deltas("crivo-150m", "crivo-150m-v2")
mac("RDeltaOneFifty", f"{corr([r[4] - r[3] for r in d150], [r[2] - r[1] for r in d150]):.2f}")
mac("NDeltaOneFifty", len(d150))
d150.sort(key=lambda r: r[2] - r[1])
pick = d150[:7] + d150[-6:][::-1]
lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         "Benchmark & Private v1 & Private v2 & $\\Delta$ private & $\\Delta$ public \\\\", "\\midrule"]
for i, (n, p1, p2, q1, q2) in enumerate(pick):
    if i == 7:
        lines.append("\\midrule")
    lines.append(f"{esc(n)} & {p1:.2f} & {p2:.2f} & {p2 - p1:+.2f} & {q2 - q1:+.2f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_v1v2_bench.tex", "\n".join(lines) + "\n")

# Prumo per-benchmark transfer correlation (board data)
P = model("prumo")
pp = [(public_skill(P, b["id"]), private_skill(P, b["id"])) for a in AREAS for b in a["panel"]]
pp = [t for t in pp if None not in t]
mac("RPrumoTransfer", f"{corr([t[0] for t in pp], [t[1] for t in pp]):.2f}")
mac("NPrumoTransfer", len(pp))

# ===================================================================== macros file
txt = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in MACROS.items())
write("stats.tex", txt)
print(len(MACROS), "macros")
