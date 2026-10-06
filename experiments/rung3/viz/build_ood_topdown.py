"""Top-down sketch vs flown paths for Table 1's OOD sketch cells (paper figure, 2026-09-26): our policy's flights (cut
at the end of the sketch, as the scorer judges them) over the sketch they were commanded, gates as their aperture
spans. Plan view per the room convention: +x up, +y left. Writes ood_topdown.pdf/.png (orbit and figure-eight) and
ood_topdown_gates.pdf/.png (the 13 arbitrary gate poses, each at its own scale with a 1 m bar).

  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
      ~/code/openpi/.venv/bin/python build_ood_topdown.py
"""
import glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__)); RD = os.path.dirname(SP); sys.path.insert(0, RD)
import moved_gate_cell as MG  # noqa: E402
import route_contact as RC  # noqa: E402
from rescore_tables import POSES  # noqa: E402

RUN = "/home/dfliu/ctxrun"; OURS, GATE = "#2f6fa8", "#3b3b3b"
CELLS = [("Orbit Gate", "right_gate", "ro25_orbit_wide", "sketch_orbit_wide.json"),
         ("Figure-Eight", "left_and_center", "ro25_fig8_denis3_cx15", "sketch_fig8_denis3_cx15.json")]


def gates(safety):
    s = yaml.safe_load(open(os.path.expanduser(f"~/code/falsify-pi/configs/safety/{safety}.yaml")))
    return [np.asarray(g["corners"], float)[:2, :2] for g in (s["ordered_miss_gate"]["gates"] if "ordered_miss_gate" in s else [s["miss_gate"]])]


def flights(tag, S):
    fs = sorted(glob.glob(f"{RUN}/traj_{tag}_[0-9]*.npy"))
    return [(P := np.load(f)[:, :3].astype(float))[:RC.route_end(P, S) + 1] for f in fs]


def draw(ax, S, F, G, lw=0.9, alpha=0.5):
    # plan view: horizontal = -y (so +y is left), vertical = x
    for P in F:
        ax.plot(-P[:, 1], P[:, 0], color=OURS, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.plot(-S[:, 1], S[:, 0], color="black", lw=lw * 1.1, ls=(0, (4, 2.5)))
    for a, b in G:
        ax.plot([-a[1], -b[1]], [a[0], b[0]], color=GATE, lw=3.2, solid_capstyle="butt")
    ax.plot(0, 0, marker="^", ms=5, color="black", ls="none")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def axes_key(ax):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); L = 0.5; ox, oy = x0 + 0.1, y0 + 0.1
    ax.annotate("", (ox, oy + L), (ox, oy), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#555555", mutation_scale=7))
    ax.annotate("", (ox + L, oy), (ox, oy), arrowprops=dict(arrowstyle="<|-", lw=0.7, color="#555555", mutation_scale=7))
    ax.text(ox, oy + L + 0.05, "+x", fontsize=6.5, ha="center", va="bottom", color="#555555")
    ax.text(ox + L + 0.05, oy, "+y", fontsize=6.5, ha="left", va="center", color="#555555")


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9))
    for ax, (title, safety, tag, sk) in zip(axes, CELLS):
        S = np.asarray(json.load(open(f"{RD}/{sk}"))["points"], float)[:, :3]; F = flights(tag, S)
        draw(ax, S, F, gates(safety)); ax.set_title(f"{title} ({len(F)} flights)", fontsize=9.5, pad=3); axes_key(ax)
    h = [plt.Line2D([], [], color="black", ls=(0, (4, 2.5)), lw=1), plt.Line2D([], [], color=OURS, lw=1.2),
         plt.Line2D([], [], color=GATE, lw=3), plt.Line2D([], [], color="black", marker="^", ls="none", ms=5)]
    fig.legend(h, ["sketch", "flown (ours)", "gate", "start"], loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{SP}/ood_topdown.{ext}", bbox_inches="tight", dpi=300)

    fig, axes = plt.subplots(3, 5, figsize=(5.5, 3.6)); axes = axes.ravel()
    for ax, (spec, seed) in zip(axes, POSES):
        dyaw, dx, dy = map(float, spec.split(","))
        pt = spec.translate(str.maketrans({"-": "m", ",": "_", ".": "_"})) + f"_{seed}"
        S = np.asarray(json.load(open(f"{RD}/sketch_mg_rr25mg{pt}.json"))["points"], float)[:, :3]
        a, b, *_ = MG.moved_geometry(dyaw, dx, dy)
        draw(ax, S, flights(f"rr25mg{pt}", S), [(a, b)], lw=0.7, alpha=0.6)
    for ax in axes[len(POSES):]:
        ax.axis("off")
    for ax in axes[:len(POSES)]:        # each pose at its own scale, with a 1 m bar so sizes still read
        ax.margins(0.08); x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); yb = y0 - 0.1 * (y1 - y0)
        ax.plot([x1 - 1.0, x1], [yb, yb], color="#555555", lw=1.2, solid_capstyle="butt")   # below the data
        ax.text(x1 - 1.08, yb, "1 m", fontsize=6, ha="right", va="center", color="#555555")
        ax.set_ylim(yb - 0.04 * (y1 - y0), y1)
    fig.legend(h, ["sketch", "flown (ours)", "gate", "start"], loc="lower right", ncol=1, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.98, 0.04), handlelength=1.8)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("pdf", "png"):
        fig.savefig(f"{SP}/ood_topdown_gates.{ext}", bbox_inches="tight", dpi=300)
    print(f"{SP}/ood_topdown.pdf\n{SP}/ood_topdown_gates.pdf")


if __name__ == "__main__":
    main()
