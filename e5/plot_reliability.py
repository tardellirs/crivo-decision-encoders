"""Reliability diagrams for E5: one 3x3 multi-panel PDF from e5/out/calib.json (no model data needed).

Bars: our benchmark-weighted reliability on the public suite (15 equal-width bins of top-label confidence; bar height =
accuracy in the bin, bar width spans the bin; bins holding < 0.2% of the weight are hidden). Orange markers: the
board's calibration bins (10 bins, maintainer-computed; 0.4 board, 0.3.2 for the v1 models). Grey diagonal = perfect
calibration. Panel text: our ECE-15 [95% CI] and the board ECE.

    uv run --python 3.12 --with matplotlib --with numpy python plot_reliability.py out/calib.json out/reliability.pdf
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OURS, BOARD, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def main(src, dst):
    r = json.load(open(src))
    order = ["c17m", "c32v1", "c32v2", "c68m", "c150v1", "c150v2", "c400m", "c1b", "prumo"]
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.6), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, order):
        ax.plot([0, 1], [0, 1], color="#9a9893", lw=0.8, ls="--", zorder=1)
        for b in r["reliability"][m]["bench"]:
            if b["w"] < 0.002:
                continue
            ax.bar(b["lo"] + 1 / 30, b["acc"], width=1 / 15 - 0.008, color=OURS, zorder=2, linewidth=0)
        brd = r["board"]["crivo_prumo"][m]
        bx = [x["conf"] for x in brd["rel"] if x["w"] >= 0.002]
        by = [x["acc"] for x in brd["rel"] if x["w"] >= 0.002]
        ax.scatter(bx, by, s=12, color=BOARD, edgecolor="white", linewidth=0.6, zorder=3)
        o = r["overall"][m]["bench"]; c = r["ci"]["bench"][m]["ece15"]
        ax.set_title(r["labels"][m], color=INK, fontsize=8, pad=3)
        ax.text(0.03, 0.97, f"ECE {o['ece15']:.3f} [{c[0]:.3f}, {c[1]:.3f}]\nboard ECE {brd['ece']:.3f}\n"
                f"acc {o['acc']:.2f} · conf {o['conf']:.2f}", transform=ax.transAxes, va="top", ha="left",
                fontsize=6.3, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("confidence (top-label probability)", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy", color=MUTED)
    h1 = plt.Rectangle((0, 0), 1, 1, color=OURS)
    h2 = plt.Line2D([], [], marker="o", ls="", color=BOARD, markeredgecolor="white", markersize=4.5)
    h3 = plt.Line2D([], [], ls="--", color="#9a9893", lw=0.8)
    fig.legend([h1, h2, h3], ["public suite, ours (15 bins, benchmark-weighted)",
                              "board (10 bins, maintainer-computed)", "perfect calibration"],
               loc="lower center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.0, w_pad=0.6)
    fig.savefig(dst)
    if dst.endswith(".pdf"):
        fig.savefig(dst[:-4] + ".png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
