"""Table 1 flights split by whether they crossed the gate (2026-09-25, for checking the "missed the gate" counts).
Verdicts come from rescore_tables.py (falsify posthoc, normal-direction label, re-measured apertures).

  /home/dfliu/code/tv/bin/python build_miss_page.py
"""
import html, json, os, sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
import cloudviewer  # noqa: E402
from build_traj_page import axes, marks  # noqa: E402

RUN = "/home/dfliu/ctxrun"
V = json.load(open(f"{RUN}/rescore_fixed_judge.json"))
COL = {("pi0", True): [90, 170, 240], ("pi0", False): [240, 80, 80],
       ("ours", True): [96, 235, 160], ("ours", False): [255, 140, 40]}


def section(side, k):
    groups, counts = [], []
    for arm, key in [("pi0", f"T1 pi0 {side}"), ("ours", f"T1 ours {side}")]:
        rows = V[key]
        for crossed in (True, False):
            tr = [np.load(f"{RUN}/{stem}.npy")[:, :3].astype(np.float32) for stem, r in sorted(rows.items()) if bool(r["transit"]) == crossed]
            lab = f"{'π0' if arm == 'pi0' else 'ours'}: {'crossed the gate' if crossed else 'missed the gate'} ({len(tr)})"
            if tr:
                groups.append({"label": lab, "color": COL[(arm, crossed)], "trajs": tr})
            counts.append(lab)
    groups += marks(side) + axes()
    v = cloudviewer.viewer_html(side, groups, elem_id=f"v{k}", max_pts=40000,
                                note="Tick a group to show it. The blue outline is the gate opening the judge uses; yellow is the goal box.")
    return f"<h2>{side.capitalize()} gate</h2><p class='sub'>{html.escape(' · '.join(counts))}</p><div class='vc'>{v}</div>"


def main():
    body = section("left", 0) + section("right", 1)
    src = open(f"{SP}/build_traj_page.py").read()
    css = src[src.index("<style>"):src.index("</style>") + 8].replace("{{", "{").replace("}}", "}")
    css = css.replace("</style>", ".bgbtn{font:12px ui-monospace,Menlo,monospace;padding:3px 9px;border-radius:5px;border:1px solid var(--line);background:transparent;color:inherit;cursor:pointer}</style>")
    page = f"""<title>Missed Gates</title>{css}<main><h1>Missed Gates</h1>
<p class="sub">The 100 simulated flights per policy and gate from Table 1, split by whether the flight crossed the gate opening in the
correct direction. A missed flight never passes through the opening.</p>{body}</main>"""
    open(f"{SP}/missed_gates.html", "w").write(page); print(f"wrote missed_gates.html ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
