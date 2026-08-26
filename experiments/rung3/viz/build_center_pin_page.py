"""Center-gate pin, real vs sim: decoded 50-step command paths (chunk = U mu*) from early-flight
frames, over the center scene cloud, next to the actual center demos. The question the page
answers: prompted with the center task on REAL observations (a task never demonstrated in
real), does the pin command the same coarse movement it commands in sim?

  python3 build_center_pin_page.py   (after center_pin_real_probe.py; writes center_pin_real.html)
"""
import os
import sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import cloudviewer

RD = os.path.dirname(SP)
Z = np.load(f"{RD}/center_pin_real.npz")

APER = [[3.156, -0.328, 0.125], [2.356, -0.327, 0.125], [2.356, -0.327, 1.875],
        [3.156, -0.328, 1.875]]
GOAL_C, GOAL_H = np.array([1.525, -0.615, 1.0]), np.array([0.3, 0.3, 0.5])


def box_edges(c, h):
    corners = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32)
    idx = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    return [corners[[a, b]] for a, b in idx]


def demos(eps, n=8):
    return [np.load(f"{RD}/data_gate_synth/ep_{e:04d}.npz", allow_pickle=True)["state"][:, :3]
            .astype(np.float32) for e in list(eps)[:n]]


def paths(key, n=40):
    return [p.astype(np.float32) for p in Z[key][:n]]


def sig_label(key):
    s = Z[key.replace("_paths", "_sig")]
    return f"sigma* {np.median(s):.1f}"


SECS = []
for pk, dem in (("cfl", range(0, 50)), ("cfr", range(50, 100))):
    groups = [
        {"label": f"center demos ({pk.upper()})", "color": [128, 136, 150], "trajs": demos(dem)},
        {"label": f"SIM frames + {pk.upper()} prompt — commanded 50-step path "
                  f"({sig_label(f'synth_{pk}_paths')})",
         "color": [96, 205, 255], "trajs": paths(f"synth_{pk}_paths")},
        {"label": f"REAL frames + {pk.upper()} prompt — commanded 50-step path "
                  f"({sig_label(f'real_{pk}_paths')})",
         "color": [255, 121, 90], "trajs": paths(f"real_{pk}_paths")},
        {"label": "goal box (judge)", "color": [248, 210, 90], "trajs": box_edges(GOAL_C, GOAL_H)},
        {"label": "center aperture (judge)", "color": [124, 208, 240],
         "trajs": [np.array(APER + [APER[0]], np.float32)]},
    ]
    SECS.append((pk.upper(), cloudviewer.viewer_html("center", groups, elem_id=f"v_{pk}",
                                                     max_pts=40000)))

body = "".join(f"<h2>{t} prompt</h2>\n<div class='vc'>{h}</div>\n" for t, h in SECS)
page = f"""<title>Center Pin, Real vs Sim</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;
padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}}
h1{{font-size:23px;margin:0 0 4px}} h2{{font-size:16px;margin:30px 0 8px;color:var(--acc)}}
.sub{{color:var(--mut);margin:0 0 18px;max-width:92ch}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;
font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}}
.sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}}
.v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0}}
</style>
<main>
<h1>Center Pin, Real vs Sim</h1>
<p class="sub">The head (gmsig) prompted with the center-gate task on early-flight frames, its
argmax command mu* decoded to the implied 50-step coarse path (chunk = U mu*) from each frame's
own position. Blue = sim frames, orange = REAL frames (a task with zero real demonstrations).
Grey = actual sim center demos for reference. If the orange fan points where the blue fan
points, the pin's task command transfers to real observations.</p>
{body}
</main>
"""
out = f"{SP}/center_pin_real.html"
open(out, "w").write(page)
print(f"wrote {out} ({len(page)/1e6:.1f} MB)")
