"""One-page image of the paper's key results with how each was run (2026-10-08), for saving and sharing.

Numbers are the scored results recorded in docs/RESEARCH_LOG.md and the paper; the agent realism block is read live from
experiments/rung3/agent_eval/realism.md (agent_realism.py). Writes docs/results_sheet.png.

  python3 docs/results_sheet.py          # system python (the openpi venv's matplotlib rejects the Ubuntu variable font)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); RD = os.path.join(os.path.dirname(HERE), "experiments", "rung3")
INK, MUTED, RULE, HEAD, OURS = "#1b1f24", "#5d6570", "#d6d9dc", "#eef1f4", "#e8f0f8"

BLOCKS = [
    ("1. Demonstrated tasks (main result)",
     "Both policies fine-tuned from the same π0 checkpoint for 5,000 steps on 100 real demos (50 left gate, 50 right gate). Sim: 100 flights per "
     "gate, 25-step replan, Gaussian-splat renders (visual shift). Hardware: 10 flights per gate, human-judged. Success = through the opening, "
     "correct direction and order, no contact (18 cm body radius).",
     ["", "Sim left", "Sim right", "Sim total", "HW left", "HW right", "HW total"],
     [["π0", "44/100", "45/100", "89/200", "9/10", "9/10", "18/20"],
      ["Ours", "66/100", "80/100", "146/200", "10/10", "7/10", "17/20"]], {1}),
    ("2. Why ours wins in sim: misses and centring",
     "Same 400 sim flights. Missed opening = no correct-direction transit. Offset = 90th percentile of |horizontal offset from the gate centre| "
     "over crossings through the opening (Figure 5).",
     ["", "Missed openings", "90% within (left)", "90% within (right)"],
     [["π0", "69 / 200", "20 cm", "29 cm"], ["Ours", "19 / 200", "13 cm", "11 cm"]], {1}),
    ("3. Trust input σ (ablation)",
     "Ours with σ fixed instead of set by the head, same flights and seeds as block 1 (sim). σ = 0 follows the head's command exactly; 1.5 is the "
     "training cap. Neither fixed setting beats π0 (89/200).",
     ["σ", "Left", "Right", "Total"],
     [["Learned (head)", "66/100", "80/100", "146/200"], ["Fixed 0", "30/100", "57/100", "87/200"], ["Fixed 1.5", "42/100", "9/100", "51/200"]], {0}),
    ("4. OOD sketched routes",
     "A person sketches a route never demonstrated; it becomes the command. Sim: 5 flights at each of 13 arbitrary gate poses, 10 orbits (right-gate "
     "scene), 10 figure-eights (left + centre gate). Completed = flown to the end of the sketch without contact. Hardware: ours only, 5 each. "
     "Tracking = mean distance to the sketch (median over flights). Baselines impose the same command on the π0 fine-tune.",
     ["", "Arbitrary gates", "Orbit", "Figure-eight", "Track arb.", "Track orbit", "Track fig-8", "HW orbit", "HW fig-8"],
     [["π0 + injected sketch", "24/65", "2/10", "1/10", "26.8 cm", "76.2 cm", "43.7 cm", "–", "–"],
      ["SDEdit on π0 (t0 = 0.5)", "58/65", "7/10", "9/10", "5.4", "4.9", "6.1", "–", "–"],
      ["Velocity projection on π0", "65/65", "10/10", "10/10", "0.9", "0.9", "1.3", "–", "–"],
      ["Ours", "60/65", "10/10", "10/10", "4.9", "4.4", "6.1", "5/5", "5/5"]], {3}),
    ("5. More data (appendix)",
     "Mixed real + simulated corpus covering more routes (incl. centre-gate tasks); two training seeds for demonstrated tasks (20 flights per cell). "
     "OOD sketches injected into π0 for the baseline; ours = gmsig3 (left-then-centre) and xswap (orbit, figure-eight).",
     ["", "Left", "Right", "Centre (from L)", "Centre (from R)", "Total", "Left, then centre", "Orbit", "Fig-8"],
     [["π0 (scratch3)", "17/20", "19/20", "16/20", "7/20", "59/80", "0/5", "0/5", "0/5"],
      ["Ours", "20/20", "20/20", "20/20", "16/20", "76/80", "5/5", "5/5", "5/5"]], {1}),
    ("6. Flawed sketch (appendix)",
     "Mixed-data policies, a left-then-centre sketch that passes 7 cm from the centre gate's post; 10 flights each. Velocity projection keeps a "
     "fraction s of the flow's own velocity along the command. Closest = median closest approach to the post.",
     ["Method", "Setting", "Completed", "Closest to post"],
     [["Ours", "σ = 0", "0/10", "3.4 cm"], ["Ours", "σ = 0.3", "2/10", "8.2 cm"], ["Ours", "σ = 0.5", "8/10", "21.1 cm"],
      ["Velocity projection", "s = 0 / 0.1 / 0.3 / 0.5", "0/10 each", "6.3 / 11.7 / 6.0 / 7.5 cm"], ["SDEdit", "t0 = 0.5", "0/10", "3.2 cm"]], {0, 1, 2}),
    ("7. Agent as command author",
     "Claude pilots in sim, 5 flights per task, same briefs for every cell (commit 485ea86), models pinned (claude-sonnet-5, claude-opus-5). Equal "
     "budget: 14/20/24 decisions per task, ≤ 50 steps (5 s) per decision. Ours: approve the policy's proposal or override with a move the policy "
     "flies. Smooth waypoints: the move flown as its own smooth track, no policy. Decoded U c: the old baseline (the command without the flow). "
     "Success = task done, no contact. Hardware: Sonnet + ours, left gate then mannequin, 2/2.",
     ["Agent / interface", "Find mannequin", "Circle centre gate", "Left gate, then mannequin", "Total", "s per decision"],
     [["Sonnet + smooth waypoints", "2/5", "2/5", "1/5", "5/15", ""], ["Sonnet + ours", "3/5", "2/5", "2/5", "7/15", "15.4"],
      ["Sonnet + decoded U c (old)", "0/5", "4/5", "1/5", "5/15", ""],
      ["Opus + smooth waypoints", "4/5", "3/5", "3/5", "10/15", ""], ["Opus + ours", "4/5", "3/5", "4/5", "11/15", "16.9"],
      ["Opus + decoded U c (old)", "3/5", "5/5", "4/5", "12/15", ""]], {1, 4}),
]


def realism_block():
    lines = [l for l in open(os.path.join(RD, "agent_eval", "realism.md")).read().splitlines() if l.startswith("|") and "---" not in l]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    head = [h.replace("speed median (m/s)", "speed med").replace("speed p95 (m/s)", "speed p95").replace(" (m/s²)", "").replace(" (m/s³)", "")
            .replace(" (°/s)", "").replace("W1 to demos, ", "W1 ") for h in rows[0]]
    ours = {i for i, r in enumerate(rows[1:]) if "ours" in r[0]}
    return ("8. Motion realism of the agent flights vs the 100 real demos",
            "Per-flight 10 Hz statistics, median over flights; W1 = Wasserstein distance of the pooled per-step distribution (while moving) to the "
            "demos' (lower = more pilot-like). Smooth waypoints are smoother than the pilot and cruise too fast; decoded U c is far too jerky; ours "
            "is the closest to the demos on speed, acceleration and jerk for both agents. The simulator is kinematic.",
            head, rows[1:], ours)


def pi(x):
    return x.replace("π0", r"$\pi_0$").replace("π", r"$\pi$") if isinstance(x, str) else x


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Ubuntu", "DejaVu Sans"], "mathtext.fontset": "cm"})
    blocks = BLOCKS + [realism_block()]
    heights = [0.62 + 0.27 * (len(rows) + 1) for *_, rows, _ in blocks]
    W, top = 13.0, 1.0
    H = top + sum(heights) + 0.25 * len(blocks)
    fig = plt.figure(figsize=(W, H))
    fig.text(0.02, 1 - 0.32 / H, "Dynamism of a Trajectory — key results", fontsize=17, color=INK, va="top")
    fig.text(0.02, 1 - 0.68 / H, "Source-noise command channel on a quadrotor; sim = Gaussian-splat flight room, hardware = real drone. "
             "Snapshot 2026-10-08 (repo: docs/RESEARCH_LOG.md).", fontsize=9.5, color=MUTED, va="top")
    y = H - top
    for (title, detail, head, rows, hl), h in zip(blocks, heights):
        fig.text(0.02, y / H, title, fontsize=12.5, color=INK, va="top")
        import textwrap
        fig.text(0.02, (y - 0.25) / H, pi("\n".join(textwrap.wrap(detail, 190))), fontsize=8.3, color=MUTED, va="top", linespacing=1.3)
        nlines = len(textwrap.wrap(detail, 190))
        th = 0.27 * (len(rows) + 1)
        ax = fig.add_axes([0.02, (y - 0.32 - 0.16 * nlines - th) / H, 0.96, th / H]); ax.axis("off")
        first = 0.24 if len(head) > 5 else 0.3
        tb = ax.table(cellText=[[pi(c) for c in r] for r in rows], colLabels=[pi(h) for h in head], cellLoc="center", bbox=[0, 0, 1, 1],
                      colWidths=[first] + [(1 - first) / (len(head) - 1)] * (len(head) - 1))
        tb.auto_set_font_size(False); tb.set_fontsize(9)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor(RULE); cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor(HEAD)
            elif (r - 1) in hl:
                cell.set_facecolor(OURS)
            if c == 0 and r > 0:
                cell._loc = "left"
        y -= h + 0.16 * nlines + 0.1
    fig.savefig(os.path.join(HERE, "results_sheet.png"), dpi=170, bbox_inches="tight", facecolor="white")
    print(os.path.join(HERE, "results_sheet.png"))


if __name__ == "__main__":
    main()
