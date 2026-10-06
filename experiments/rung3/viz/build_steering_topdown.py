"""Top-down sketch vs flown paths per method for the OOD-steering table (paper figure, 2026-09-26). Rows are the
sketches (orbit, figure-eight, one arbitrary gate pose), columns the methods; every flight of the cell is drawn, cut at
the end of the sketch as the scorer judges it, in the method's colour when it completed the route and in red when it
did not (score_ood_cells' rules). A failed flight carries its reason: a black dot where it comes within the body radius
(0.18 m) of a gate before the end of the route, and a black cross at its last position when the flight's time ran out
before it reached the end of the sketch. Plan view per the room convention: +x up, +y left.

  env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
      ~/code/openpi/.venv/bin/python build_steering_topdown.py      # writes steering_topdown.pdf/.png
  ... build_steering_topdown.py --strip   # figure-eight row only, one strip for the main text: steering_strip.pdf/.png
"""
import glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__)); RD = os.path.dirname(SP); sys.path.insert(0, RD)
import gate_clearance as G  # noqa: E402
import moved_gate_cell as MG  # noqa: E402
import route_contact as RC  # noqa: E402
import score_ood_cells as S  # noqa: E402

RUN = "/home/dfliu/ctxrun"; FAIL, GATE = "#d62728", "#3b3b3b"
METHODS = [(r"$\pi_0$ + injected sketch", "inj25", "#6b6b6b"), (r"SDEdit on $\pi_0$", "sde25", "#2e9e4f"),
           (r"Velocity projection on $\pi_0$", "vproj25", "#7b4fb5"), ("Ours", "ro25", "#2f6fa8")]
POSE = ("107,-1.29,0.24", 63)   # the arbitrary pose where the methods differ most (injected 0/5, SDEdit 3/5, others 5/5)


def gates(safety):
    s = yaml.safe_load(open(os.path.expanduser(f"~/code/falsify-pi/configs/safety/{safety}.yaml")))
    return [np.asarray(g["corners"], float) for g in (s["ordered_miss_gate"]["gates"] if "ordered_miss_gate" in s else [s["miss_gate"]])]


def rows():
    """(row label, sketch, gate segments, gate cloud, {method prefix: [(flight, completed)]})"""
    out = []
    # orbit: completed = reaches the end of the sketch without touching the gate
    S_ = np.asarray(json.load(open(f"{RD}/sketch_orbit_wide.json"))["points"], float)[:, :3]
    cloud = torch.tensor(np.asarray(G.gate_cloud("right"), np.float32))
    cells = {m: [(P, S.reached_clean(P, S_, cloud)) for P in load(f"{m}_orbit_wide")] for _, m, _ in METHODS}
    out.append(("Orbit Gate", S_, [g[:2, :2] for g in gates("right_gate")], cloud, cells))
    # figure-eight: also both gates in order along the sketch's direction, no wrong-way pass
    S_ = np.asarray(json.load(open(f"{RD}/sketch_fig8_denis3_cx15.json"))["points"], float)[:, :3]
    cloud = torch.tensor(np.asarray(G.gate_cloud("left_and_center"), np.float32)); gs = gates("left_and_center")

    def fig8_ok(P):
        (lf, lb, l0), (cf, cb, c0) = S.crossings(P, gs[0]), S.crossings(P, gs[1])
        return S.reached_clean(P, S_, cloud) and lf >= 1 and cf >= 1 and lb == 0 and cb == 0 and l0 < c0
    cells = {m: [(P, fig8_ok(P)) for P in load(f"{m}_fig8_denis3_cx15")] for _, m, _ in METHODS}
    out.append(("Figure-Eight", S_, [g[:2, :2] for g in gs], cloud, cells))
    # one arbitrary gate pose: the moved gate's route check plus reaching the end cleanly
    spec, seed = POSE; dyaw, dx, dy = map(float, spec.split(","))
    pt = spec.translate(str.maketrans({"-": "m", ",": "_", ".": "_"})) + f"_{seed}"
    S_ = np.asarray(json.load(open(f"{RD}/sketch_mg_rr25mg{pt}.json"))["points"], float)[:, :3]
    cloud = torch.tensor(np.asarray(G.moved_gate_cloud(dyaw, (dx, dy)), np.float32)); a, b, *_ = MG.moved_geometry(dyaw, dx, dy)
    cells = {m: [(P, bool(MG.score_traj(P, dyaw, dx, dy)["ok"]) and S.reached_clean(P, S_, cloud))
                 for P in load(f"{'rr25' if m == 'ro25' else m}mg{pt}")] for _, m, _ in METHODS}
    out.append(("Arbitrary Gate", S_, [np.array([a, b])], cloud, cells))
    return out


def load(tag):
    return [np.load(f)[:, :3].astype(float) for f in S.trajs(tag)]


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Ubuntu", "DejaVu Sans"], "mathtext.fontset": "cm"})
    strip = "--strip" in sys.argv
    R = [r for r in rows() if r[0] == "Figure-Eight"] if strip else rows()
    fig, axes = plt.subplots(len(R), len(METHODS), figsize=(5.5, 1.75 if strip else 4.6), squeeze=False,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    for r, (label, S_, segs, cloud, cells) in enumerate(R):
        allpts = np.concatenate([S_[:, :2]] + [s for s in segs])
        lo, hi = allpts.min(0) - 0.35, allpts.max(0) + 0.35
        for c, (name, m, col) in enumerate(METHODS):
            ax = axes[r, c]; flights = cells[m]; n_ok = sum(ok for _, ok in flights)
            marks = []
            for P, ok in sorted(flights, key=lambda x: x[1], reverse=True):     # failures drawn last, on top
                e, dmin = RC.contact(P, S_, cloud); timed_out = e == len(P) - 1; P = P[:e + 1]
                ax.plot(-P[:, 1], P[:, 0], color=col if ok else FAIL, lw=0.8, alpha=0.75 if ok else 0.9)
                if ok:
                    continue
                if dmin < RC.BODY:          # touched a gate: where it came closest
                    d = torch.cdist(torch.from_numpy(P.astype(np.float32)), cloud).min(1).values
                    marks.append(("o", P[int(d.argmin())]))
                if timed_out:               # never reached the end of the sketch before the flight's time ran out
                    marks.append(("x", P[-1]))
            for mk, q in marks:
                ax.plot(-q[1], q[0], marker=mk, ms=3.6 if mk == "o" else 4.2, color="black", mew=1.1, ls="none", zorder=5)
            ax.plot(-S_[:, 1], S_[:, 0], color="black", lw=0.9, ls=(0, (4, 2.5)))
            for s in segs:
                ax.plot(-s[:, 1], s[:, 0], color=GATE, lw=3, solid_capstyle="butt")
            ax.text(0.97, 0.03, f"{n_ok}/{len(flights)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color=col if n_ok == len(flights) else FAIL)
            ax.set_xlim(-hi[1], -lo[1]); ax.set_ylim(lo[0], hi[0]); ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#d6d9dc"); sp.set_linewidth(0.6)
            if r == 0:
                ax.set_title(name, fontsize=7.5, pad=3, color=col)
            if c == 0 and not strip:
                ax.set_ylabel(label, fontsize=8)
    h = [plt.Line2D([], [], color="black", ls=(0, (4, 2.5)), lw=1), plt.Line2D([], [], color=GATE, lw=3),
         plt.Line2D([], [], color=FAIL, lw=1.2), plt.Line2D([], [], marker="o", color="black", ls="none", ms=3.6),
         plt.Line2D([], [], marker="x", color="black", ls="none", ms=4.2, mew=1.1)]
    names = ["sketch", "gate", "not completed", "touches a gate", "out of time"]
    if strip:       # the figure-eight row has no flight that ran out of time
        h, names = h[:4], names[:4]
    fig.legend(h, names, loc="lower center", ncol=len(names), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.1 if strip else 0.01), handlelength=1.8)
    out = "steering_strip" if strip else "steering_topdown"
    for ext in ("pdf", "png"):
        fig.savefig(f"{SP}/{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"{SP}/{out}.pdf")


if __name__ == "__main__":
    main()
