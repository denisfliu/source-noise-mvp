"""Point-cloud page for the mannequin-search flights (2026-09-21): the four reviewed trajectories over the
left_and_center cloud, the figure's position as a fixed mark (a 1.7 m vertical at (7.4, -0.2) with a
0.5 m ring at its foot -- the viewer's cloud is clipped at |x| <= 6 m, so the figure itself is off the
cloud), the back wall as a line at x = 7.8, and the judge marks and axes as usual.

  python build_mann_page.py --out mannequin.html
"""
import argparse, os, sys, json
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
import cloudviewer                      # noqa: E402
from build_traj_page import marks, axes  # noqa: E402
AF = os.path.join(os.path.dirname(SP), "agentflight")
MANN = np.array([7.4, -0.2, 0.0], np.float32)
FLIGHTS = [("mann1: Sonnet, swept in place (closest 7.3 m)", "mann1", [200, 120, 255]),
           ("mann2: Sonnet, explored the -x bay (closest 7.5 m)", "mann2", [255, 140, 90]),
           ("mann3: Sonnet, sighted at decision 11, mirrored (closest 4.0 m)", "mann3", [255, 90, 90]),
           ("mann4: Sonnet + L/R marks, ended 1.0 m in front (SUCCESS)", "mann4", [80, 220, 120]),
           ("peng1: Sonnet, penguin; 0.11 m off, 1.00 m up, 30-step hold (SUCCESS)", "peng1", [80, 200, 255])]
PENG = np.array([1.525, -0.615, 0.75], np.float32)   # the stuffed penguin on the goal table (goal box centre, table height)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="mannequin.html"); a = ap.parse_args()
    groups = []
    for label, tag, col in FLIGHTS:
        t = np.load(os.path.join(AF, f"traj_{tag}.npy")).astype(np.float32)[:, :3]
        groups.append({"label": label, "color": col, "trajs": [t]})
    th = np.linspace(0, 2 * np.pi, 33, dtype=np.float32)
    ring = np.stack([MANN[0] + 0.5 * np.cos(th), MANN[1] + 0.5 * np.sin(th), np.full_like(th, 0.05)], 1)
    groups.append({"label": "mannequin (7.4, -0.2), 1.7 m tall", "fixed": True, "color": [255, 60, 200],
                   "trajs": [np.stack([MANN, MANN + [0, 0, 1.7]]), ring]})
    pring = np.stack([PENG[0] + 0.3 * np.cos(th), PENG[1] + 0.3 * np.sin(th), np.full_like(th, PENG[2])], 1)
    groups.append({"label": "penguin (1.525, -0.615) on the goal table", "fixed": True, "color": [255, 140, 40],
                   "trajs": [pring, np.stack([PENG, PENG + [0, 0, 0.4]])]})
    groups.append({"label": "back wall (x = 7.8)", "fixed": True, "color": [180, 180, 180],
                   "trajs": [np.array([[7.8, -3, 0.05], [7.8, 3, 0.05]], np.float32),
                             np.array([[7.8, -3, 2.2], [7.8, 3, 2.2]], np.float32)]})
    groups += marks("left_and_center") + axes()
    note = ("Start (0, 0, 1.5) facing -x for all five. Mannequin flights scored by closest approach to the figure; the penguin "
            "flight by the compound judge's goal box (0.6 x 0.6 x 1.0 m about the penguin). mann4 ended 1.02 m from the figure, "
            "14 deg off the line to it; peng1 ended 0.11 m off the penguin in xy at 1.00 m altitude, inside the box for its last 30 steps.")
    body = "<h2>Search flights, agent-reviewed (2026-09-21): mannequin and penguin</h2>" + \
        cloudviewer.viewer_html("left_and_center", groups, elem_id="v3d", max_pts=40000, note=note)
    open(os.path.join(SP, a.out), "w").write("<title>Search Flights</title>\n" + body)
    print("wrote", a.out)
main()
