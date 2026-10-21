#!/usr/bin/env python3
"""Figures of the paper (light: matplotlib only, a few seconds, < 300 MB).

    python3 paper_artifacts/make_figs.py     # from the repository root, after make_tables.py; writes build/figs/

fig_task.pdf      the task: an invented typed-decision request -> encoder -> per-question probabilities (the released
                  Crivo-150M's real output, read from figs/task_example.json, written once by figs/task_example.py),
                  and where the encoder sits in a pipeline.
fig_reader.pdf    joint vs per-option reader, with option-option attention arcs and per-question cost (sequences, tokens).
fig_a1bench.pdf   A1 per benchmark: P - J-comp (compute-matched) skill on the 37 pub15 benchmarks, sorted, paired-bootstrap
                  CIs (results_gpu3/analysis/a1_analysis.json, from a1a2/analyze_a1.py --ref Jcomp).
fig_flips.pdf     option-permutation flip rate vs parameters, Crivo / baselines / A1 per-option arm (stats/gpu1/e4.json).
fig_latency.pdf   PENDING GPU-3: S1-Arena accuracy vs median CPU latency (results_gpu3/e6cpu/summary.json); drawn only
                  when that file exists, otherwise deleted; main.tex shows it only under ifhaveCpuLat.
fig_overview.pdf  (a) S1-Arena task-mean accuracy vs parameters, Crivo vs baselines (our runs, item CIs) -- the lead;
                  (b) Decision Index 0.4 Full vs served parameters with the Pareto frontier (board data) -- supporting.
fig_ablation.pdf  A1/A2 grid at 32M: per-arm means (bars = horizontal ticks; dot plot, axis not at zero) and the three seeds on three benchmarks.
fig_pubpriv.pdf   public index vs private same-skill score (all board models); one panel.
fig_prumo.pdf     Prumo public vs private chance-corrected skill per benchmark (board data).
Colours: Crivo #2a78d6 (blue), baselines #eb6834 (orange); validated with the dataviz palette validator.
"""
import json
from os import environ as _os_env
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                                   # repository root
REL = EXP / "board"
FIG = Path(_os_env.get("FIG_DIR", EXP / "build" / "figs")); FIG.mkdir(parents=True, exist_ok=True)
D04 = json.load(open(REL / "di-index-20261020.json"))
D032 = json.load(open(REL / "di-index-v0.3.2-20261020.json"))
METH = json.load(open(REL / "di-methodology-20261020.json"))
PANEL = {b["id"]: b["short"] for a in METH["index"]["areas"] if a.get("weight") for b in a["panel"]}
S1 = json.load(open(EXP / "stats/gpu1/s1.json"))
SEEDS = json.load(open(Path(_os_env.get("OUT_DIR", EXP / "build" / "tables")) / "a1a2_seeds.json"))

INK, INK2, GRID = "#222222", "#666666", "#E6E6E6"
OTHER, FRONT = "#B8C7DD", "#2F5D9E"
CRIVO, BASE, PRUMO, PUBL = "#2a78d6", "#eb6834", "#B23A48", "#8a8a8a"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "pdf.fonttype": 42, "axes.titlesize": 7.5})


def model(engine, data=D04):
    return next(m for m in data["models"] if m["engine"] == engine)


def priv(m, bid):
    b = m["benchmarks"].get(str(bid), {})
    x = b.get("private_eq")
    if x is None:
        return None
    r = b.get("random") or 0.0
    return max(0.0, min(1.0, (x - r) / (1 - r)))


def pub(m, bid):
    return m["benchmarks"].get(str(bid), {}).get("skill")


# ------------------------------------------------------------------ Figure: overview (a) Pareto, (b) S1-Arena
fig, (bx, ax) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1, 1.15]})
pts = sorted([(m["meta"]["served_params"], m["v03"]["v"], m) for m in D04["models"]
              if (m.get("meta") or {}).get("served_params") and m.get("v03") and m["v03"].get("v") is not None],
             key=lambda t: t[0])
front, best = [], -1
for p, v, m in pts:
    if v > best:
        best = v
        front.append((p, v, m))
ax.set_xscale("log")
ax.grid(True, color=GRID, lw=0.5, zorder=0)
oth = [(p, v) for p, v, m in pts if not m["engine"].startswith("crivo")]
ax.scatter([p for p, _ in oth], [v for _, v in oth], s=8, color=OTHER, edgecolor="none", zorder=2,
           label="other entries")
fx, fy = [p for p, _, _ in front], [v for _, v, _ in front]
ax.step(fx + [4e10], fy + [fy[-1]], where="post", color=FRONT, lw=1.1, zorder=3, label="Pareto frontier")
cr = [(p, v, m) for p, v, m in pts if m["engine"].startswith("crivo")]
ax.scatter([p for p, _, _ in cr], [v for _, v, _ in cr], s=30, marker="D", color=CRIVO, edgecolor="white",
           lw=0.6, zorder=5, label="Crivo (this work)")
lab = {"crivo-17m": ((0, 6), "center"), "crivo-32m-v2": ((0, 6), "center"), "crivo-68m": ((-2, 7), "right"),
       "crivo-150m-v2": ((0, -10), "center"), "crivo-400m": ((2, -10), "center"), "crivo-1b": ((6, -10), "left")}
for p, v, m in cr:
    off, ha = lab[m["engine"]]
    ax.annotate(m["name"].replace("Crivo-", "").replace(" v2", ""), (p, v), xytext=off, textcoords="offset points",
                fontsize=6.3, color=INK, ha=ha, fontweight="bold")
for e, (off, ha) in {"onejev1.6-0.8b": ((-5, 5), "right"), "amber-0.4b": ((-5, 4), "right"),
                     "mammoth-1": ((-5, 3), "right")}.items():
    m = model(e)
    ax.annotate(m["name"], (m["meta"]["served_params"], m["v03"]["v"]), xytext=off, textcoords="offset points",
                fontsize=5.8, color=INK2, ha=ha)
pr = model("prumo")["v03"]["v"]
ax.axhline(pr, color=PRUMO, lw=0.9, ls="--", zorder=1)
ax.text(1.3e9, pr - 4.6, f"Prumo, no learned parameters ({pr:.2f})", color=PRUMO, fontsize=6)
ax.set_xlim(1e7, 4e10)
ax.set_ylim(0, 76)
ax.set_xlabel("Served parameters (log scale)")
ax.set_ylabel("Full score, Decision Index 0.4 (private)")
ax.set_title("(b) Decision Index 0.4 private tests (supporting), %d sized entries" % len(pts), fontsize=7)
ax.legend(loc="upper left", fontsize=6, frameon=False)

SYS = [("crivo-17m", 16.9e6), ("crivo-32m", 32.0e6), ("crivo-68m", 68.4e6), ("crivo-150m", 149.6e6),
       ("crivo-400m", 395.8e6), ("crivo-1b", 1031.3e6)]
BASES = [("nli-deberta-v3-large-zs2", 435e6, "DeBERTa-v3-large zs-v2", (5, -2), "left"),
         ("nli-bart-large-mnli", 407e6, "BART-large-MNLI", (5, -5), "left"),
         ("letter-qwen3-0.6b", 596e6, "Qwen3-0.6B", (5, 1), "left"),
         ("letter-gemma-3-1b", 1000e6, "Gemma-3-1B-it", (5, -3), "left"),
         ("reranker-32m-raw", 32e6, "Ettin-32M untrained", (5, -6), "left")]
bx.set_xscale("log")
bx.grid(True, color=GRID, lw=0.5, zorder=0)
for sub, ls, mfc, lab_ in (("all", "-", CRIVO, "Crivo, all 209 tasks"), ("unseen", "--", "white", "Crivo, 145 unseen tasks")):
    xs = [p for _, p in SYS]
    ys = [S1["item_mode"][sub]["macro"]["own"][k]["value"] for k, _ in SYS]
    lo = [y - S1["item_mode"][sub]["macro"]["own"][k]["ci95"][0] for (k, _), y in zip(SYS, ys)]
    hi = [S1["item_mode"][sub]["macro"]["own"][k]["ci95"][1] - y for (k, _), y in zip(SYS, ys)]
    bx.errorbar(xs, ys, yerr=[lo, hi], color=CRIVO, ls=ls, lw=1.1, marker="D", ms=4, mfc=mfc, mec=CRIVO,
                capsize=0, zorder=4, label=lab_)
for k, p, name, off, ha in BASES:
    o = S1["item_mode"]["all"]["macro"]["own"][k]
    bx.errorbar([p], [o["value"]], yerr=[[o["value"] - o["ci95"][0]], [o["ci95"][1] - o["value"]]], color=BASE,
                marker="o", ms=4, ls="none", zorder=4, label="baselines run by us (all tasks)" if k.startswith("nli-deb") else None)
    bx.annotate(name, (p, o["value"]), xytext=off, textcoords="offset points", fontsize=5.6, color=INK2, ha=ha)
for name, p, v in (("Kai-0.6B (published*)", 600e6, 0.503), ("Laya (published*)", 421e6, 0.445),
                   ("Qwen3-0.6B (published*)", 596e6, 0.392)):
    bx.scatter([p], [v], s=16, marker="s", facecolor="white", edgecolor=PUBL, lw=0.9, zorder=3,
               label="*leaderboard rows, different item set" if name.startswith("Kai") else None)
    bx.annotate(name, (p, v), xytext=(5, 1) if not name.startswith("Qwen") else (5, -8), textcoords="offset points",
                fontsize=5.6, color=PUBL, ha="left")
u = S1["item_mode"]["all"]["macro"]["own"]["uniform"]["value"]
bx.axhline(u, color=PUBL, lw=0.8, ls=":")
bx.text(1.2e7, u + 0.004, "uniform guess", fontsize=5.6, color=PUBL)
bx.set_xlim(1e7, 8e9)
bx.set_ylim(0.31, 0.60)
bx.set_xlabel("Parameters (log scale)")
bx.set_ylabel("Task-mean accuracy, S1-Arena")
bx.set_title("(a) S1-Arena, external audited benchmark (95% item CIs)", fontsize=7)
bx.legend(loc="upper left", fontsize=5.6, frameon=False)
fig.tight_layout(w_pad=1.2)
fig.savefig(FIG / "fig_overview.pdf")
fig.savefig(FIG / "fig_overview.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure: A1/A2 ablation
ARMS = [("J", "J"), ("Jsteps", "J-steps")] + ([("Jcomp", "J-comp")] if "Jcomp" in SEEDS else []) + [("P", "P")] + \
    ([("Praw", "P raw")] if "Praw" in SEEDS else []) + [("JnoBrier", "no Brier"), ("JnoKD", "no KD"),
                                                        ("Jraw", "J raw")]
ARM_COL = {"P": CRIVO, "Praw": "#8fb8ea", "Jcomp": BASE, "Jsteps": "#f2b48f"}
METS = [("pub15_skill", "Public-suite subset\n(chance-corrected skill)"), ("s1_acc", "S1-Arena subset\n(task-mean accuracy)"),
        ("btzsc_f1", "BTZSC subset\n(macro-F1, 22 datasets)")]
fig, axs = plt.subplots(1, 3, figsize=(3.5, 2.15))
rng = np.random.default_rng(0)
for ax_, (k, title) in zip(axs, METS):
    for i, (a, _) in enumerate(ARMS):
        vals = [s[k] for s in SEEDS[a]]
        c = ARM_COL.get(a, "#9aa7b8")
        ax_.hlines(np.mean(vals), i - 0.32, i + 0.32, color=c, lw=2.0, zorder=2)
        if len(vals) > 1:
            ax_.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=9, color=c, edgecolor=INK, lw=0.3, zorder=3)
        else:   # one seed: hollow marker
            ax_.scatter([i], vals, s=10, facecolor="white", edgecolor=c, lw=0.9, zorder=3)
    lo = min(min(s[k] for s in SEEDS[a]) for a, _ in ARMS)
    hi = max(max(s[k] for s in SEEDS[a]) for a, _ in ARMS)
    ax_.set_ylim(max(0, lo - 0.08 * (hi - lo) - 0.03), hi + 0.08 * (hi - lo) + 0.01)
    ax_.set_xticks(range(len(ARMS)))
    ax_.set_xticklabels([l for _, l in ARMS], rotation=90, ha="center", fontsize=5.6)
    ax_.set_title(title, fontsize=5.8)
    ax_.tick_params(axis="y", labelsize=5.6)
    ax_.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
fig.tight_layout(w_pad=0.6)
fig.savefig(FIG / "fig_ablation.pdf")
fig.savefig(FIG / "fig_ablation.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure: public vs private
fig, a1 = plt.subplots(figsize=(3.0, 2.1))
rows = [(m["v03"]["pub"], m["v03"]["same_skills"], m) for m in D04["models"]
        if m.get("v03") and m["v03"].get("pub") is not None and m["v03"].get("same_skills") is not None]
x = np.array([r[0] for r in rows])
y = np.array([r[1] for r in rows])
r = np.corrcoef(x, y)[0, 1]
a1.grid(True, color=GRID, lw=0.5, zorder=0)
a1.scatter(x, y, s=5, color=OTHER, edgecolor="none", zorder=2, label="other entries")
for i, (pv, sv, m) in enumerate([t for t in rows if t[2]["engine"].startswith("crivo") or t[2]["engine"] == "prumo"]):
    isp = m["engine"] == "prumo"
    a1.scatter([pv], [sv], s=14, marker="s" if isp else "D", color=PRUMO if isp else CRIVO, edgecolor="white", lw=0.4,
               zorder=4, label=("Prumo" if isp else None))
a1.scatter([], [], s=14, marker="D", color=CRIVO, label="Crivo")
lim = [0, max(x.max(), y.max()) + 3]
a1.plot(lim, lim, color=INK2, lw=0.6, ls=":")
a1.set_xlim(lim)
a1.set_ylim(lim)
a1.set_xlabel("Public index (self-run kit)", fontsize=6.5)
a1.set_ylabel("Private same-skill score (board)", fontsize=6.5)
a1.set_title(f"{len(rows)} entries, Pearson r = {r:.3f}", fontsize=6.5)
a1.tick_params(labelsize=5.8)
a1.legend(loc="upper left", fontsize=5.8, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig_pubpriv.pdf")
fig.savefig(FIG / "fig_pubpriv.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure: Prumo transfer
P = model("prumo")
cap = {"CRUXEval", "BBH", "ChessBench", "RAGTruth", "Habermas", "CLINC150", "When2Call", "ToolRet", "BANKING77",
       "HoVer", "GSM8K", "BFCL", "API-Bank", "POP909"}
OFF3 = {"ToolRet": ((-3, 3), "right"), "BANKING77": ((3, 3), "left"), "HoVer": ((-3, -7), "right"),
        "API-Bank": ((3, -6), "left"), "GSM8K": ((3, -7), "left"), "BFCL": ((3, -7), "left"),
        "POP909": ((-3, 3), "right")}
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ax.grid(True, color=GRID, lw=0.5, zorder=0)
ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=":")
for bid, name in PANEL.items():
    u_, w = pub(P, bid), priv(P, bid)
    if u_ is None or w is None or (u_ < 0.05 and w < 0.05):
        continue
    ax.scatter([u_], [w], s=12, color=PRUMO, edgecolor="white", lw=0.4, zorder=3)
    if name in cap:
        off, ha = OFF3.get(name, ((3, 3), "left"))
        ax.annotate(name, (u_, w), xytext=off, textcoords="offset points", fontsize=5.6, color=INK2, ha=ha)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 0.75)
ax.set_xlabel("Public skill (chance-corrected)")
ax.set_ylabel("Private skill (chance-corrected)")
fig.tight_layout()
fig.savefig(FIG / "fig_prumo.pdf")
fig.savefig(FIG / "fig_prumo.png", dpi=200)
plt.close(fig)


# ------------------------------------------------------------------ shared helpers for the diagrams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

TINT_J, TINT_P, TINT_S = "#d6e6f8", "#fbe1d5", "#f2f2f2"


def rbox(ax, x, y, w, h, fc, ec=INK2, lw=0.6, ls="-", r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls,
                                zorder=z))


def arrow(ax, a, b, color=INK2, lw=0.7, style="-|>", rad=0.0, ms=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ Figure: the task (typed decision request)
# The request is invented; the probabilities are the real output of the released Crivo-150M on it, from
# figs/task_example.json (written once by figs/task_example.py, CPU; the build only reads the file).
TASK = json.load(open(HERE / "task_example.json"))
FIG_MODEL = "Crivo-150M"
ANS = TASK["runs"][FIG_MODEL]["answers"]
Q = TASK["questions"]
E6 = json.load(open(EXP / "stats/gpu1/e6.json"))
CPU17 = next(c for c in E6["cpu"] if c["name"] == "crivo-17m" and c["threads"] == 16)["median_ms"]
CPU150 = next(c for c in E6["cpu"] if c["name"] == "crivo-150m" and c["threads"] == 16)["median_ms"]
try:   # GPU-3 same-subset CPU run (results_gpu3/e6cpu/summary.json) replaces the older VM run when present
    import os as _os
    _c3 = json.load(open(Path(_os.environ.get("GPU3_DIR", EXP / "results_gpu3")) / "e6cpu" / "summary.json"))["systems"]
    CPU17, CPU150 = _c3["crivo-17m"]["common_subset"]["median_ms"], _c3["crivo-150m"]["common_subset"]["median_ms"]
except (OSError, ValueError, KeyError):
    pass

W_, H_ = 3.5, 3.08
fig = plt.figure(figsize=(W_, H_))
ax = fig.add_axes([0, 0, 1, 1])
U = 10 / W_                       # data units per inch, same on both axes (x: 0..10)
TOP = H_ * U
ax.set_xlim(0, 10)
ax.set_ylim(0, TOP)
ax.axis("off")
FS = 5.7


def fy(y):                        # data y -> figure fraction
    return y / TOP


# --- request box
ry1 = TOP - 0.05
ax.text(0.22, ry1 - 0.14, "Request", fontsize=6.6, fontweight="bold", color=INK, va="top")
ax.text(1.62, ry1 - 0.17, "(invented support ticket)", fontsize=FS, color=INK2, va="top", style="italic")
st = TASK["state"]
import textwrap
lines_ = [f'channel: {st["channel"]},  plan: {st["plan"]}'] + textwrap.wrap(f'message: "{st["message"]}"', 66)
sb0 = ry1 - 0.5 - 0.3 * len(lines_) - 0.14
rbox(ax, 0.22, sb0, 9.56, ry1 - 0.5 - sb0, TINT_S, lw=0.5)
ax.text(0.34, ry1 - 0.58, "state", fontsize=FS, color=INK2, va="top", fontweight="bold")
ax.text(1.12, ry1 - 0.58, "\n".join(lines_), fontsize=FS, color=INK, va="top", linespacing=1.3)
TYPE_TAG = {"choice": "choice", "noul": "yes/no (noul)", "score": "ordinal (score)"}
qy = sb0 - 0.12
for key in ("team", "refund", "urgency"):
    q = Q[key]
    rbox(ax, 0.22, qy - 0.5, 1.75, 0.42, TINT_J, lw=0.5)
    ax.text(1.095, qy - 0.29, TYPE_TAG[q["type"]], fontsize=FS - 0.3, ha="center", va="center", color=INK)
    ax.text(2.12, qy - 0.17, q["instructions"], fontsize=FS, va="center", color=INK)
    if q["type"] == "choice":
        opts = "options: " + " | ".join(q["criteria"])
    elif q["type"] == "noul":
        opts = "options: false | true"
    else:
        opts = "levels: " + " < ".join(q["criteria"].values())
    ax.text(2.12, qy - 0.43, opts, fontsize=FS - 0.3, va="center", color=INK2)
    qy -= 0.6
ry0 = qy - 0.02
rbox(ax, 0.05, ry0, 9.9, ry1 - ry0, "white", lw=0.7, z=0)
# --- encoder
ey = ry0 - 0.78
arrow(ax, (5.0, ry0), (5.0, ey + 0.5))
rbox(ax, 0.6, ey, 8.8, 0.5, CRIVO, ec=CRIVO, lw=0.6)
ax.text(5.0, ey + 0.25, "Crivo encoder  \u00b7  one pass per question  \u00b7  softmax over its options",
        fontsize=FS + 0.1, ha="center", va="center", color="white", fontweight="bold")
arrow(ax, (5.0, ey), (5.0, ey - 0.32))
# --- answers (real model output)
ax.text(0.22, ey - 0.46, f"Answers: {FIG_MODEL}'s actual output on this request", fontsize=6.3, fontweight="bold",
        color=INK, va="center")
panels = [("team", list(Q["team"]["criteria"]), [ANS["team"]["probabilities"][k] for k in Q["team"]["criteria"]]),
          ("refund", ["false", "true"], [1 - ANS["refund"]["noul"], ANS["refund"]["noul"]]),
          ("urgency", list(Q["urgency"]["criteria"].values()),
           [ANS["urgency"]["probabilities"][str(i)] for i in range(len(Q["urgency"]["criteria"]))])]
for (key, labs, ps), x0, wd in zip(panels, [0.155, 0.47, 0.78], [0.16, 0.12, 0.14]):
    pa = fig.add_axes([x0, fy(1.95), wd, fy(ey - 0.93) - fy(1.95)])
    yy = np.arange(len(labs))[::-1]
    best = int(np.argmax(ps))
    pa.barh(yy, ps, height=0.62, color=[CRIVO if i == best else "#9fc0e8" for i in range(len(ps))], lw=0)
    for y_, p_ in zip(yy, ps):
        pa.text(p_ + 0.05, y_, f"{p_:.2f}", va="center", fontsize=FS - 0.4, color=INK)
    pa.set_yticks(yy)
    pa.set_yticklabels(labs, fontsize=FS - 0.3)
    pa.set_xlim(0, 1.45)
    pa.set_ylim(-0.6, len(labs) - 0.4)
    pa.set_xticks([])
    pa.spines["bottom"].set_visible(False)
    pa.tick_params(axis="y", length=0, pad=1.5)
    pa.set_title(key, fontsize=FS, color=INK2, pad=1.5)
# --- where it sits
ax.text(0.22, 1.5, "Where it sits", fontsize=6.3, fontweight="bold", color=INK, va="center")
steps = [(0.22, 2.55, "LLM agent or app\nbuilds the request", TINT_S),
         (3.38, 3.0, f"Crivo on a CPU, median\n{CPU17:.0f} ms (17M), {CPU150:.0f} ms (150M)", TINT_J),
         (7.05, 2.73, "argmax or threshold:\nroute, pick tool, block", TINT_S)]
for x_, w_, t_, fc_ in steps:
    rbox(ax, x_, 0.12, w_, 0.78, fc_, lw=0.5)
    ax.text(x_ + w_ / 2, 0.51, t_, fontsize=FS - (0.5 if fc_ is TINT_J else 0.1), ha="center", va="center", color=INK,
            linespacing=1.3)
arrow(ax, (2.77, 0.51), (3.38, 0.51))
arrow(ax, (6.38, 0.51), (7.05, 0.51))
ax.text(3.07, 1.0, "request", fontsize=FS - 0.6, ha="center", color=INK2)
ax.text(6.72, 1.0, "probabilities", fontsize=FS - 0.6, ha="center", color=INK2)
fig.savefig(FIG / "fig_task.pdf")
fig.savefig(FIG / "fig_task.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure: the two readers (schematic, with cost)
W_, H_ = 3.5, 1.25
fig = plt.figure(figsize=(W_, H_))
ax = fig.add_axes([0, 0, 1, 1])
U = 10 / W_
TOP = H_ * U
ax.set_xlim(0, 10)
ax.set_ylim(0, TOP)
ax.axis("off")
FS = 5.6
BH = 0.34                                   # box height
SX, SW, OW, G = 0.05, 3.55, 1.08, 0.08      # state box x/width, option box width, gap


def seq(y, opts, tint, state_fc=TINT_S, state_ls="-"):
    rbox(ax, SX, y, SW, BH, state_fc, lw=0.6, ls=state_ls)
    ax.text(SX + SW / 2, y + BH / 2, "[CLS] instr [SEP] state [SEP]", ha="center", va="center", fontsize=FS,
            color=INK)
    xs = []
    for j, i in enumerate(opts):
        x = SX + SW + G + j * (OW + G)
        rbox(ax, x, y, OW, BH, tint, lw=0.6)
        ax.text(x + OW / 2, y + BH / 2, f"[OPT] $o_{i}$", ha="center", va="center", fontsize=FS, color=INK)
        xs.append(x + OW / 2)
    return xs


# joint reader
yj = TOP - 1.15
ax.text(SX, TOP - 0.16, "Joint reader (released): 1 sequence per question", fontsize=6.4, fontweight="bold",
        color=INK, va="center")
cx = seq(yj, [1, 2, 3], TINT_J)
top = yj + BH + 0.02
for a_, b_, rad in ((cx[0], cx[1], -0.38), (cx[1], cx[2], -0.38), (cx[0], cx[2], -0.22)):
    arrow(ax, (a_, top), (b_, top), color=CRIVO, lw=0.6, style="<|-|>", rad=rad, ms=4)
xe = SX + SW + G + 3 * (OW + G)
ax.text(xe - 0.02, top + 0.22, "options attend\nto each other", fontsize=FS - 0.3, color=CRIVO, va="center",
        linespacing=1.1)
ax.text(xe - 0.02, yj + BH / 2 - 0.02, "$\\rightarrow$ 3 logits, softmax", fontsize=FS, va="center", color=INK)
ax.text(SX, yj - 0.2, "cost: 1 pass;  tokens $\\approx |x| + \\Sigma_i\\,|o_i|$", fontsize=FS, color=INK2,
        va="center")
# per-option reader
ax.text(SX, yj - 0.56, "Per-option reader (ablation): $n$ sequences per question", fontsize=6.4, fontweight="bold",
        color=INK, va="center")
rows = []
for i in range(3):
    y = yj - 1.06 - i * (BH + 0.1)
    seq(y, [i + 1], TINT_P, state_fc="white" if i else TINT_S, state_ls="--" if i else "-")
    ax.text(SX + SW + G + OW + 0.07, y + BH / 2, f"$\\rightarrow \\ell_{i + 1}$", fontsize=FS, va="center", color=INK)
    rows.append(y)
bx_ = SX + SW + G + OW + 0.55
ax.plot([bx_, bx_ + 0.1, bx_ + 0.1, bx_], [rows[0] + BH - 0.03, rows[0] + BH - 0.03, rows[2] + 0.03, rows[2] + 0.03],
        color=INK2, lw=0.6)
ax.text(bx_ + 0.22, (rows[0] + rows[2] + BH) / 2, "softmax across the $n$ logits;\nthe state is re-read for every\n"
        "option (dashed), and options\nnever see each other", fontsize=FS - 0.2, va="center", color=INK,
        linespacing=1.2)
ax.text(SX, rows[2] - 0.2, "cost: $n$ passes;  tokens $\\approx n\\,|x| + \\Sigma_i\\,|o_i|$", fontsize=FS,
        color=INK2, va="center")
fig.savefig(FIG / "fig_reader.pdf")
fig.savefig(FIG / "fig_reader.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure: A1 per benchmark (P - J-comp)
A1 = json.load(open(EXP / "results_gpu3/analysis/a1_analysis.json"))   # analyze_a1.py --ref Jcomp
assert A1["meta"]["ref"] == "Jcomp"
SHORT = {"HoVer claim verification": "HoVer", "Home appliance simulator": "Home appliance sim.",
         "New Yorker caption matching": "New Yorker captions", "PhishNChips phishing decisions": "PhishNChips",
         "BBH fixed-option tasks": "BBH (fixed options)", "RAGTruth response-level hallucination": "RAGTruth",
         "When2Call MCQ": "When2Call", "Habermas Machine": "Habermas"}
rows = []
for b in A1["pub15"]["per_benchmark"]:
    d = b["d_skill"]["P-Jcomp"]
    rows.append((SHORT.get(b["benchmark"], b["benchmark"]), d["diff"], d["boot_ci95"]))
rows.sort(key=lambda t: -t[1])
n_up = sum(v > 1e-9 for _, v, _ in rows)
n_dn = sum(v < -1e-9 for _, v, _ in rows)
s_up = sum(c[0] > 0 for _, _, c in rows)
s_dn = sum(c[1] < 0 for _, _, c in rows)
assert (len(rows), n_up, n_dn, s_up, s_dn) == (37, 22, 10, 8, 1), (len(rows), n_up, n_dn, s_up, s_dn)  # = VI-B text
NS = "#9aa7b8"
half = (len(rows) + 1) // 2
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.3), sharex=True)
for ax_, part in zip(axs, (rows[:half], rows[half:])):
    yy = np.arange(len(part))[::-1]
    for y_, (name, v, c) in zip(yy, part):
        col = CRIVO if c[0] > 0 else (BASE if c[1] < 0 else NS)
        ax_.barh(y_, v, height=0.6, color=col, lw=0, zorder=2)
        ax_.plot(c, [y_, y_], color=INK, lw=0.6, zorder=3)
    ax_.axvline(0, color=INK2, lw=0.6, zorder=1)
    ax_.set_yticks(yy)
    ax_.set_yticklabels([r[0] for r in part], fontsize=5.4)
    ax_.set_ylim(-0.7, half - 0.3)
    ax_.tick_params(axis="y", length=0, pad=2)
    ax_.tick_params(axis="x", labelsize=5.6)
    ax_.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    ax_.spines["left"].set_visible(False)
axs[0].set_xlim(-0.24, 0.42)
axs[0].set_xticks([-0.2, 0, 0.2, 0.4])
axs[0].set_xticklabels(["$-$0.2", "0", "0.2", "0.4"])
fig.text(0.5, 0.012, "P $-$ J-comp, chance-corrected skill (95% paired bootstrap)", fontsize=6.2,
         ha="center", va="bottom", color=INK)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=CRIVO, label=f"P better ({s_up})"), Patch(color=BASE, label=f"joint better ({s_dn})"),
                    Patch(color=NS, label="interval includes 0")], loc="upper center", ncol=3, fontsize=5.6,
           frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.0)
axs[0].set_position([0.245, 0.14, 0.25, 0.78])
axs[1].set_position([0.735, 0.14, 0.25, 0.78])
fig.savefig(FIG / "fig_a1bench.pdf")
fig.savefig(FIG / "fig_a1bench.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure: option-order flips vs size
E4 = json.load(open(EXP / "stats/gpu1/e4.json"))["systems"]
CRIVO_P = [("crivo-17m", 16.9e6), ("crivo-32m", 32.0e6), ("crivo-68m", 68.4e6), ("crivo-150m", 149.6e6),
           ("crivo-400m", 395.8e6), ("crivo-1b", 1031.3e6)]
BASE_F = [("nli-deberta-v3-large-zs2", 435e6, "DeBERTa-v3-large zs-v2 (NLI)", (0, -6), "center"),
          ("letter-qwen3-0.6b", 596e6, "Qwen3-0.6B", (-6, 0), "right"),
          ("letter-gemma-3-1b", 1000e6, "Gemma-3-1B-it", (-6, 0), "right"),
          ("reranker-32m-raw", 32e6, "Ettin-32M untrained", (6, 0), "left")]
fig, ax = plt.subplots(figsize=(3.4, 2.15))
ax.set_xscale("log")
ax.grid(True, color=GRID, lw=0.5, zorder=0)
xs = [p for _, p in CRIVO_P]
fp = [E4[k]["flip_perm_mean"] for k, _ in CRIVO_P]
ax.errorbar(xs, [f["value"] for f in fp], yerr=[[f["value"] - f["ci95"][0] for f in fp], [f["ci95"][1] - f["value"] for f in fp]],
            color=CRIVO, lw=1.1, marker="D", ms=4, mec=CRIVO, capsize=0, zorder=4, label="Crivo (joint), permuted options")
yn = [E4[k]["yn"]["flip_rate"]["value"] for k, _ in CRIVO_P]
ax.plot(xs, yn, color=CRIVO, ls="--", lw=1.0, marker="D", ms=3.6, mfc="white", mec=CRIVO, zorder=4,
        label="Crivo (joint), yes/no swapped")
for k, p, name, off, ha in BASE_F:
    f = E4[k]["flip_perm_mean"]
    ax.errorbar([p], [f["value"]], yerr=[[f["value"] - f["ci95"][0]], [f["ci95"][1] - f["value"]]], color=BASE, marker="o",
                ms=4, ls="none", zorder=4, label="baselines, permuted options" if k.startswith("nli") else None)
    ax.annotate(name, (p, f["value"]), xytext=off, textcoords="offset points", fontsize=5.6, color=INK2, ha=ha,
                va="center" if off[1] >= 0 else "top", linespacing=1.1)
pf = [s["flip_perm"] for s in SEEDS["P"]]
ax.scatter([32.0e6], [np.mean(pf)], s=20, marker="s", facecolor="white", edgecolor=CRIVO, lw=1.0, zorder=5,
           label="per-option reader (32M, A1 arm P),\n0 flips by construction")
for (k, p), f in zip(CRIVO_P, fp):
    if k in ("crivo-17m", "crivo-1b"):
        ax.annotate(k.replace("crivo-", "").upper(), (p, f["value"]), xytext=(0, 6), textcoords="offset points",
                    fontsize=5.8, color=INK, ha="center", fontweight="bold")
ax.set_xlim(1.2e7, 2e9)
ax.set_ylim(-0.1, 0.85)
ax.set_xlabel("Parameters (log scale)")
ax.set_ylabel("Share of answers that change")
h_, l_ = ax.get_legend_handles_labels()
order = [l_.index(t) for t in ("Crivo (joint), permuted options", "Crivo (joint), yes/no swapped",
                               "per-option reader (32M, A1 arm P),\n0 flips by construction", "baselines, permuted options")]
ax.legend([h_[i] for i in order], [l_[i] for i in order], loc="center left", bbox_to_anchor=(0.17, 0.53), fontsize=5.6,
          frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig_flips.pdf")
fig.savefig(FIG / "fig_flips.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure (pending GPU-3): accuracy vs CPU latency
# Drawn only when paper/exp/results_gpu3/e6cpu/summary.json exists (override with env GPU3_DIR), with the same
# system-name matching as tables/make_tables.py (g3_pick: mean over passes of the common-subset median). main.tex shows
# it only under \ifhaveCpuLat, so nothing appears in the PDF until then; a stale file is removed.
import os
G3 = Path(os.environ.get("GPU3_DIR", EXP / "results_gpu3"))
try:
    C3 = json.load(open(G3 / "e6cpu" / "summary.json"))
except (OSError, ValueError):
    C3 = None


def g3_median(summary, prefixes):
    for pre in prefixes:
        hits = [v for n_, v in summary["systems"].items() if n_ == pre or n_.startswith(pre + ".")]
        vals = [h["common_subset"].get("median_ms") for h in hits if h["common_subset"].get("median_ms") is not None]
        if vals:
            return sum(vals) / len(vals), hits[0]["common_subset"].get("n")
    return None


LATF = [("crivo-17m", ["crivo-17m"], "17M"), ("crivo-32m", ["crivo-32m-v2", "crivo-32m"], "32M"),
        ("crivo-68m", ["crivo-68m"], "68M"), ("crivo-150m", ["crivo-150m"], "150M"),
        ("crivo-400m", ["crivo-400m"], "400M"), ("crivo-1b", ["crivo-1b"], "1B")]
LATB = [("nli-deberta-v3-large-zs2", ["nli-deberta-v3-large-zs2", "nli-deberta"], "DeBERTa-v3-large zs-v2"),
        ("reranker-32m-raw", ["reranker-32m"], "Ettin-32M untrained")]
have_lat = C3 is not None and g3_median(C3, ["crivo-17m"]) and g3_median(C3, ["nli-deberta-v3-large-zs2", "nli-deberta"])
if not have_lat:
    for ext in ("pdf", "png"):
        (FIG / f"fig_latency.{ext}").unlink(missing_ok=True)
else:
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    own = S1["item_mode"]["all"]["macro"]["own"]
    pts_ = [(g3_median(C3, pre), own[k], lab) for k, pre, lab in LATF]
    pts_ = [(m_[0], o, lab) for m_, o, lab in pts_ if m_]
    ax.errorbar([m_ for m_, _, _ in pts_], [o["value"] for _, o, _ in pts_],
                yerr=[[o["value"] - o["ci95"][0] for _, o, _ in pts_], [o["ci95"][1] - o["value"] for _, o, _ in pts_]],
                color=CRIVO, lw=1.1, marker="D", ms=4, capsize=0, zorder=4, label="Crivo (joint reader)")
    for m_, o, lab in pts_:
        ax.annotate(lab, (m_, o["value"]), xytext=(4, -7), textcoords="offset points", fontsize=5.8, ha="left",
                    color=INK, fontweight="bold")
    for i, (k, pre, lab) in enumerate(LATB):
        r_ = g3_median(C3, pre)
        if not r_:
            continue
        o = own[k]
        ax.errorbar([r_[0]], [o["value"]], yerr=[[o["value"] - o["ci95"][0]], [o["ci95"][1] - o["value"]]], color=BASE,
                    marker="o", ms=4, ls="none", zorder=4, label="baselines, one pass per option" if i == 0 else None)
        ax.annotate(lab, (r_[0], o["value"]), xytext=(-5, -2), textcoords="offset points", fontsize=5.6, ha="right",
                    va="top", color=INK2)
    # the per-option arm P has no full S1-Arena run: shown on the grid's S1-60 subset (60 items per task, mean and
    # range of its three seeds), with a distinct hollow marker; the released Crivo-32M scores 0.447 on that subset
    # (results_gpu3/analysis/released_subsets.json) against 0.448 on all items, so the two sets are close but not equal.
    rp = g3_median(C3, ["crivo-32m-per-option"])
    if rp:
        pv = [x["s1_acc"] for x in SEEDS["P"]]
        pm = sum(pv) / len(pv)
        ax.errorbar([rp[0]], [pm], yerr=[[pm - min(pv)], [max(pv) - pm]], color=CRIVO, marker="*", ms=7, mfc="white",
                    mew=0.9, ls="none", zorder=5, label="per-option arm P (S1-60 subset)")
        ax.annotate("P (32M per option, S1-60)", (rp[0], pm), xytext=(-3, -9), textcoords="offset points",
                    fontsize=5.4, ha="left", va="top", color=CRIVO)
    u = own["uniform"]["value"]
    ax.axhline(u, color=PUBL, lw=0.8, ls=":")
    allx = [m_ for m_, _, _ in pts_] + [g3_median(C3, pre)[0] for _, pre, _ in LATB if g3_median(C3, pre)]
    ax.set_xlim(min(allx) / 2, max(allx) * 2)
    ax.text(min(allx) / 1.9, u + 0.004, "uniform guess", fontsize=5.6, color=PUBL)
    ax.set_ylim(u - 0.02, 0.58)
    ax.set_xlabel("Median CPU latency per request (ms, log scale)")
    ax.set_ylabel("Task-mean accuracy, S1-Arena")
    ax.legend(loc="upper left", fontsize=5.6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_latency.pdf")
    fig.savefig(FIG / "fig_latency.png", dpi=200)
    plt.close(fig)

# The author photograph (figs/stekel.jpg, a centred 4:5 crop of the author's photo) is not generated here.
print("figures written; r_pub_ss", round(r, 3), "; latency figure:", "yes" if have_lat else "pending GPU-3")
