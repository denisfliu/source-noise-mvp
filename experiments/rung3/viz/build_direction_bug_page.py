"""Direction-label bug at the left gate (2026-09-25): falsify posthoc labels a crossing by the sign of that one step's
y-velocity; at the 48-degree left gate the drone crosses moving along +x with dy ~ 0, so the label is noise. Each viewer
shows one flight, the aperture, the gate normal, and every in-aperture crossing with its step drawn 30x longer.

  /home/dfliu/code/tv/bin/python build_direction_bug_page.py
"""
import html, os, sys

import numpy as np
import yaml

SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
import cloudviewer  # noqa: E402
from build_traj_page import axes  # noqa: E402

RUN = "/home/dfliu/ctxrun"
CASES = [("armrealonly_apc25_left_48", "ours, left gate, flight 48",
          "Crosses the opening once, in the right direction, and never touches the gate. On the crossing step the drone "
          "moves +x with a slightly negative y, so the current judge calls it a wrong-way pass and fails the flight."),
         ("armscrreal_apc25_left_9", "pi0, left gate, flight 9",
          "Goes through the right way at step 330, backs out through the opening at step 340, and goes through again at "
          "step 351. The backing-out step moves -x with a slightly positive y, so the current judge calls it a correct pass "
          "and the flight succeeds; judged along the gate normal it is a wrong-way pass.")]
SCALE = 30.0


def aperture():
    C = np.asarray(yaml.safe_load(open(os.path.expanduser("~/code/falsify-pi/configs/safety/left_gate.yaml")))["miss_gate"]["corners"], np.float64)
    c = C.mean(0); u = C[1] - C[0]; v = C[3] - C[0]; n = np.cross(u, v); n /= np.linalg.norm(n)
    n = n * np.sign(n[1])            # left task: a correct crossing moves +y, so the normal points to +y
    return C, c, u / np.linalg.norm(u), v / np.linalg.norm(v), n, np.linalg.norm(C[1] - C[0]) / 2, np.linalg.norm(C[3] - C[0]) / 2


def crossings(P, c, u, v, n, hu, hv):
    s = (P - c) @ n; out = []
    for i in np.where(s[:-1] * s[1:] < 0)[0]:
        q = P[i] + s[i] / (s[i] - s[i + 1]) * (P[i + 1] - P[i])
        if abs((q - c) @ u) <= hu and abs((q - c) @ v) <= hv:
            d = P[i + 1] - P[i]
            out.append(dict(i=int(i), q=q, d=d, dy=float(d[1]), normal="correct" if s[i + 1] > s[i] else "wrong way",
                            dylab="correct" if d[1] > 0 else "wrong way"))
    return out


def main():
    C, c, u, v, n, hu, hv = aperture()
    secs = []
    for k, (stem, title, story) in enumerate(CASES):
        P = np.load(f"{RUN}/traj_{stem}.npy")[:, :3].astype(np.float64)
        X = crossings(P, c, u, v, n, hu, hv)
        groups = [{"label": "flight", "color": [96, 235, 160], "trajs": [P.astype(np.float32)]},
                  {"label": "aperture (judge)", "fixed": True, "color": [124, 208, 240], "trajs": [np.vstack([C, C[:1]]).astype(np.float32)]},
                  {"label": "gate normal: the correct crossing direction (0.5 m)", "fixed": True, "color": [255, 255, 255],
                   "trajs": [np.stack([c, c + 0.5 * n]).astype(np.float32)]}]
        for x in X:
            col = [240, 80, 80] if x["normal"] != x["dylab"] else [255, 200, 90]
            groups.append({"label": f"crossing at step {x['i']}: the step, {SCALE:.0f}x longer", "fixed": True, "color": col,
                           "trajs": [np.stack([x["q"], x["q"] + SCALE * x["d"]]).astype(np.float32)]})
        groups += axes()
        rows = "".join(f"<tr class='{'bad' if x['normal'] != x['dylab'] else ''}'><td>{x['i']}</td><td>{x['d'][0]:+.3f}</td><td>{x['dy']:+.3f}</td>"
                       f"<td>{x['dylab']}</td><td>{x['normal']}</td></tr>" for x in X)
        tab = ("<table class='rt'><tr><th>crossing step</th><th>step dx (m)</th><th>step dy (m)</th><th>current judge (sign of dy)</th>"
               f"<th>along the gate normal</th></tr>{rows}</table>")
        v3 = cloudviewer.viewer_html("left", groups, elem_id=f"v{k}", max_pts=40000,
                                     note="Red segments are crossings the two rules label differently. The white arrow is the gate normal.")
        secs.append(f"<h2>{html.escape(title)}</h2><p class='sub'>{html.escape(story)}</p><div class='vc'>{v3}{tab}</div>")
    page = f"""<title>Left-Gate Direction Bug</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}} h1{{font-size:23px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 12px;max-width:92ch}}
h2{{font-size:17px;margin:28px 0 6px;color:var(--acc)}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
table.rt{{border-collapse:collapse;font:12px ui-monospace,Menlo,monospace;margin:10px 0 0;font-variant-numeric:tabular-nums}}
.rt td,.rt th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}} .rt th{{color:var(--acc);font-weight:500}}
tr.bad td{{color:#f07070}}
</style>
<main><h1>Left-Gate Direction Bug</h1>
<p class="sub">The judge decides whether a gate crossing went the right way from the sign of the drone's y-motion on the one
step that crosses the gate plane. The left gate is turned about 48 degrees, so the drone flies through it mostly along +x and
its y-motion on that step is a few millimetres either way. Judging by motion along the gate's normal fixes this.</p>
{''.join(secs)}</main>"""
    open(os.path.join(SP, "direction_bug.html"), "w").write(page); print("wrote direction_bug.html")


if __name__ == "__main__":
    main()
