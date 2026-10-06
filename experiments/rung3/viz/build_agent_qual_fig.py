"""Qualitative agent-in-the-loop figure (paper, 2026-09-26): one Sonnet flight of "left gate, then find the mannequin"
(simulation, trial 12, completed and clean) seen from above, every decision marked where it was made (approve the
policy's proposal vs override with a primitive), and the forward camera view the agent saw at a few key decisions.

Decision poses come from the flight's command log (clog_*.npy, one row per served chunk: an approve serves one chunk,
an override two -- the policy's proposal, then the agent's command -- at the same pose); the log is checked against
the decision sequence. The figure is laid out as a row of views over a wide map, so the map is turned: +x right,
+y up (a rotation of the room's usual +x up / +y left plan view).

  python3 build_agent_qual_fig.py     # system python: the openpi venv's matplotlib rejects the Ubuntu variable font
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__)); RD = os.path.dirname(SP); sys.path.insert(0, RD)
from agent_judges import MANNEQUIN  # noqa: E402

RUN = "/home/dfliu/ctxrun"; TAG = "left_mannequin_ours_t12"; D = f"{RUN}/agent_sim_{TAG}"
APPROVE, OVERRIDE, PATH, GATE = "#1e9650", "#e07b00", "#2f6fa8", "#3b3b3b"
# decision index -> short caption for the thumbnails (paraphrasing the agent's own "seen | action" note)
KEY = {2: "Override: left post\nlooks closer,\nshift right",
       9: "Approve: gate\ncentered, let the\npolicy fly toward it",
       13: "Override: posts out\nof view, fly on and\nturn to search",
       16: "Override: figure\nright of center,\nturn toward it",
       21: "Override: mannequin\nfills the view,\nhold"}
# label offsets (points) on the map, placed by hand so the crowded start and gate area stays legible
OFF = {2: (11, -1), 9: (0, -9), 13: (-2, 9), 16: (0, 9), 21: (-9, 5)}

def decisions():
    V = [json.load(open(f"{D}/cmds/{k:03d}.json"))["verdict"] for k in range(len(os.listdir(f"{D}/cmds")))]
    C = np.load(f"{RUN}/clog_{TAG}.npy")
    rows, i = [], 0
    for v in V:
        rows.append(C[i, :3]); i += 1 if v == "approve" else 2
    assert i == len(C), (i, len(C))     # the log's row count must match the decision sequence exactly
    return V, np.array(rows)


def person(ax, x, y, h=0.55, color="#b3261e"):
    """A person-shaped marker standing at (x, y) in data units, h tall (seen as an upright icon on the plan)."""
    from matplotlib.patches import Circle, FancyBboxPatch
    r = 0.13 * h
    ax.add_patch(Circle((x, y + h - r), r, color=color, zorder=6))
    ax.add_patch(FancyBboxPatch((x - 0.17 * h, y), 0.34 * h, h - 2.3 * r, boxstyle=f"round,pad=0,rounding_size={0.1 * h}",
                                color=color, zorder=6))


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Ubuntu", "DejaVu Sans"]})
    V, Q = decisions(); P = np.load(f"{RUN}/traj_{TAG}.npy")[:, :3]
    saf = yaml.safe_load(open(os.path.expanduser("~/code/falsify-pi/configs/safety/left_and_center.yaml")))
    gates = [np.asarray(g["corners"], float)[:2, :2] for g in saf["ordered_miss_gate"]["gates"]][:1]   # the left gate only

    fig = plt.figure(figsize=(5.5, 2.72))
    keys = sorted(KEY); w = 1.0 / len(keys)
    for n, k in enumerate(keys):
        tx = fig.add_axes([n * w + 0.008, 0.615, w - 0.016, 0.375])
        im = Image.open(f"{D}/obs/{k:03d}_view.jpg"); W, H = im.size
        tx.imshow(im.crop((0, 0, W // 2, min(H, 236))))   # the forward panel of what the agent saw (some views carry a strip below)
        col = APPROVE if V[k] == "approve" else OVERRIDE
        tx.set_xticks([]); tx.set_yticks([])
        for sp in tx.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(1.6)
        fig.text(n * w + w / 2, 0.6, f"{k + 1}. {KEY[k]}", fontsize=6.6, va="top", ha="center", color="#1b1f24", linespacing=1.15)

    ax = fig.add_axes([0.0, 0.0, 1.0, 0.43])
    # plan view turned for the wide layout: horizontal = x, vertical = y
    for a, b in gates:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=GATE, lw=3.5, solid_capstyle="butt")
    ax.plot(P[:, 0], P[:, 1], color=PATH, lw=1.3)
    for k, (v, q) in enumerate(zip(V, Q)):
        ax.plot(q[0], q[1], marker="o" if v == "approve" else "s", ms=4.2, color=APPROVE if v == "approve" else OVERRIDE,
                mec="white", mew=0.5, zorder=4)
    for k in keys:
        q = Q[k]; col = APPROVE if V[k] == "approve" else OVERRIDE
        ax.annotate(str(k + 1), (q[0], q[1]), xytext=OFF[k], textcoords="offset points", fontsize=7.5, color=col, weight="bold",
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    person(ax, MANNEQUIN[0], MANNEQUIN[1] - 0.25)
    ax.text(MANNEQUIN[0], MANNEQUIN[1] - 0.35, "mannequin", fontsize=7, va="top", ha="center", color="#b3261e")
    ax.plot(0, 0, marker=">", ms=6, color="black"); ax.text(-0.12, 0, "start", fontsize=7, ha="right", va="center")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-1.0, 10.2); ax.set_ylim(-1.05, 1.25)   # room on the right for the legend
    h_ = [plt.Line2D([], [], marker="o", color=APPROVE, ls="none", ms=5), plt.Line2D([], [], marker="s", color=OVERRIDE, ls="none", ms=5)]
    ax.legend(h_, ["approve", "override"], loc="lower right", frameon=True, fontsize=7.5, handletextpad=0.2, borderaxespad=0.1,
              edgecolor="#9aa0a6", fancybox=False, framealpha=1)
    for ext in ("pdf", "png"):
        fig.savefig(f"{SP}/agent_qual.{ext}", dpi=300, bbox_inches="tight")
    print(f"{SP}/agent_qual.pdf")


if __name__ == "__main__":
    main()
