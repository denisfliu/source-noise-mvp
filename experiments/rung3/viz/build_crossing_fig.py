"""Where Table 1's demonstrated-task flights cross the gate plane (paper figure, 2026-09-26).

For every flight of the n=100 simulation cells, the crossing of the aperture plane nearest the aperture centre (within
1.5 m of it), drawn in the gate's own frame: horizontal offset from the centre of the opening and height, equal scale, grey bars at the posts' inner edges. Flights
that never cross the plane near the gate have no point. Dashed lines mark the 90th percentile of |horizontal offset|
over the crossings through the opening (between the posts; passes outside a post are drawn but not counted), the
numbers quoted in Section 5.2. Writes crossing_fig.pdf/.png.

  python3 build_crossing_fig.py
"""
import glob, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); RUN = "/home/dfliu/ctxrun"
PI0, OURS = "#3b3b3b", "#2f6fa8"    # as results_bars
GATES = [("Left Gate", "left_gate", "armscrreal_apc25_left", "armrealonly_apc25_left"),
         ("Right Gate", "right_gate", "armscrreal_apc25_right", "armrealonly_apc25_right")]


def aperture(safety):
    C = np.asarray(yaml.safe_load(open(os.path.expanduser(f"~/code/falsify-pi/configs/safety/{safety}.yaml")))["miss_gate"]["corners"], float)
    c = C.mean(0); u = C[1] - C[0]; v = C[3] - C[0]; n = np.cross(u, v)
    return c, u / np.linalg.norm(u), np.linalg.norm(u) / 2, n / np.linalg.norm(n), C[:, 2].min(), C[:, 2].max()


def crossing(P, c, u, n, hw):
    """The flight's crossing of the gate plane: the first one through the opening (|offset| <= hw, the posts' inner
    edges), or, for a flight that never goes through, the crossing nearest the centre (a pass outside a post)."""
    s = (P - c) @ n; near = None
    for i in np.where(s[:-1] * s[1:] < 0)[0]:
        q = P[i] + s[i] / (s[i] - s[i + 1]) * (P[i + 1] - P[i])
        lat = (q - c) @ u
        if abs(lat) >= 1.5 or abs(q[2] - c[2]) >= 1.5:
            continue
        if abs(lat) <= hw:
            return lat, q[2]
        if near is None or abs(lat) < abs(near[0]):
            near = (lat, q[2])
    return near

def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Ubuntu", "DejaVu Sans"], "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05), sharey=True, gridspec_kw={"wspace": 0.06})
    ZLO, ZHI = 100, 182        # cm; every crossing lies in this band
    for ax, (title, safety, pre_pi0, pre_ours) in zip(axes, GATES):
        c, u, hw, n, zlo, zhi = aperture(safety); hw *= 100
        for sgn in (-1, 1):     # the posts' inner edges, as the judge's aperture has them
            ax.axvspan(sgn * hw, sgn * (hw + 6), color="#c7ccd1", lw=0)
        for k, (name, pre, col, mk) in enumerate(((r"$\pi_0$", pre_pi0, PI0, "x"), ("Ours", pre_ours, OURS, "o"))):
            fs = glob.glob(f"{RUN}/traj_{pre}_[0-9]*.npy")
            X = np.array([x for x in (crossing(np.load(f)[:, :3].astype(float), c, u, n, hw / 100) for f in fs) if x is not None]) * 100
            p90 = np.percentile(np.abs(X[np.abs(X[:, 0]) <= hw, 0]), 90)   # crossings through the opening (between the posts)
            ax.scatter(X[:, 0], X[:, 1], s=10 if mk == "o" else 12, marker=mk, color=col, alpha=0.55 if mk == "o" else 0.75,
                       lw=0.8 if mk == "x" else 0, label=name, zorder=3)
            for sgn in (-1, 1):
                ax.plot([sgn * p90] * 2, [ZLO, ZHI - 15], color=col, lw=0.9, ls=(0, (3, 2)), zorder=2)
            ax.text(0, ZHI - 1 - 6 * k, f"{name}: 90% within {p90:.0f} cm", color=col, ha="center", va="top", fontsize=7.2)
            print(title, name, f"p90 {p90:.1f} cm over {int((np.abs(X[:, 0]) <= hw).sum())} crossings through the opening; {len(X)}/{len(fs)} cross the plane")
        ax.set_title(title, fontsize=9.5, pad=3)
        ax.set_xlim(-70, 70); ax.set_ylim(ZLO, ZHI); ax.set_aspect("equal")
        ax.set_xlabel("Offset from gate center (cm)", fontsize=8.5)
        ax.tick_params(labelsize=7.5, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Height (cm)", fontsize=8.5)
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8, handletextpad=0.3, borderaxespad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{HERE}/crossing_fig.{ext}", bbox_inches="tight", dpi=300)
    print(f"{HERE}/crossing_fig.pdf")


if __name__ == "__main__":
    main()
