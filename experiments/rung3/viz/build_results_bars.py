"""Table 1 as grouped bars (paper figure draft, 2026-09-25).

Success per task for the plain pi0 fine-tune and ours, simulation and hardware side by side; each bar is labelled
with its count so the different trial numbers stay visible. Writes results_bars.pdf/.png next to this script.

  python3 build_results_bars.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PI0, OURS = "#3b3b3b", "#2f6fa8"

# (task label, pi0 (k, n) or None if not run, ours (k, n)) -- Table 1
SIM = [("Left\nGate", (44, 100), (66, 100)), ("Right\nGate", (45, 100), (80, 100)),
       ("Arbitrary\nGates", (24, 65), (60, 65)), ("Orbit\nGate", (2, 10), (10, 10)),
       ("Figure-\nEight", (1, 10), (10, 10))]
HW = [("Left\nGate", (9, 10), (10, 10)), ("Right\nGate", (9, 10), (7, 10)),
      ("Orbit\nGate", None, (5, 5)), ("Figure-\nEight", None, (5, 5))]
N_DEMO = {"sim": 2, "hw": 2}   # the first N tasks of each panel are demonstrated, the rest OOD sketches


def panel(ax, rows, title, n_demo):
    w = 0.36
    for i, (name, p, o) in enumerate(rows):
        for dx, val, col in ((-w / 2 - 0.02, p, PI0), (w / 2 + 0.02, o, OURS)):
            if val is None:
                ax.text(i + dx, 2, "--", ha="center", va="bottom", fontsize=7, color="#777777")
                continue
            k, n = val
            ax.bar(i + dx, 100 * k / n, w, color=col, linewidth=0)
            ax.text(i + dx, 100 * k / n + 2, f"{k}/{n}", ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.axvline(n_demo - 0.5, color="#bbbbbb", lw=0.6, ls=(0, (3, 2)))
    for x, lab in ((n_demo / 2 - 0.5, "Demonstrated"), ((n_demo + len(rows)) / 2 - 0.5, "OOD Sketches")):
        ax.text(x, 111, lab, ha="center", va="bottom", fontsize=7.5, color="#555555")
    ax.set_title(title, fontsize=9.5, pad=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.95), gridspec_kw={"width_ratios": [len(SIM), len(HW)], "wspace": 0.08})
    panel(axes[0], SIM, "Simulation", N_DEMO["sim"])
    panel(axes[1], HW, "Hardware", N_DEMO["hw"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (PI0, OURS)]
    fig.legend(handles, [r"$\pi_0$", "Ours"], loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.16), handlelength=1.2, columnspacing=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{HERE}/results_bars.{ext}", bbox_inches="tight", dpi=250)
    print(f"{HERE}/results_bars.pdf")


if __name__ == "__main__":
    main()
