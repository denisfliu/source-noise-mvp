"""Real demonstrations over the simulator's scene clouds (2026-09-15): the 50 real left and 50 real right
mocap flights on their scenes, the simulated center demonstrations on the center scene (no real center
flights exist), and the one real hardware flight on disk (baseline, 2026-09-11) on the left scene.
  python build_real_demos_page.py
"""
import os, sys
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(SP))
import cloudviewer
from build_traj_page import marks
import realism as R

demos = R.load_demos()
real_l, real_r = demos["real"][0], demos["real"][1]
syn_cfl, syn_cfr = demos["synth"][2], demos["synth"][3]
f32 = lambda ps: [p.astype(np.float32) for p in ps]
hw = np.load(os.path.expanduser("~/gate_flights/traj_baseline_left_01.npy"))[:, :3].astype(np.float32)

def section(scene, groups, elem, note):
    return cloudviewer.viewer_html(scene, groups + marks(scene), elem_id=elem, max_pts=40000, note=note)

secs = [
    ("Left gate: 50 real demonstrations (pilot, mocap)", section("left",
        [{"label": "real left-gate demonstrations (50)", "color": [96, 235, 160], "trajs": f32(real_l)},
         {"label": "real hardware flight, pi0 baseline, 2026-09-11 (mocap)", "color": [255, 140, 40], "trajs": [hw]}],
        "v_left", "Mocap trajectories of the 50 pilot-flown left-gate demonstrations, 10 Hz, over the left scene reconstruction; orange is the one hardware flight recorded on this box (baseline arm, 2026-09-11, full mocap rate).")),
    ("Right gate: 50 real demonstrations (pilot, mocap)", section("right",
        [{"label": "real right-gate demonstrations (50)", "color": [90, 170, 240], "trajs": f32(real_r)}],
        "v_right", "Mocap trajectories of the 50 pilot-flown right-gate demonstrations over the right scene reconstruction.")),
    ("Center gate: no real demonstrations exist; the 100 simulated ones", section("center",
        [{"label": "simulated center-from-left demonstrations (50, planner)", "color": [180, 140, 255], "trajs": f32(syn_cfl)},
         {"label": "simulated center-from-right demonstrations (50, planner)", "color": [240, 210, 90], "trajs": f32(syn_cfr)}],
        "v_center", "No real center-gate flights were ever recorded (the 'real_center' collection in falsify/data is the mixed set: its center tasks are the planner's). Shown instead: the 100 simulated center demonstrations the policies learn the center tasks from.")),
]
body = "".join(f"<h2>{t}</h2><div class='vc'>{h}</div>" for t, h in secs)
page = f"""<title>Real Demonstrations in the Scene Clouds</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}} h1{{font-size:23px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px;color:var(--acc)}} .sub{{color:var(--mut);margin:0 0 18px;max-width:92ch}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
</style>
<main><h1>Real Demonstrations in the Scene Clouds</h1>
<p class="sub">The training corpus's real flights drawn over the simulator's reconstructions of the same room: mocap positions at 10 Hz, judge apertures in blue, goal box in yellow. Toggle groups in each legend; drag to orbit, wheel to zoom.</p>
{body}</main>"""
open(f"{SP}/real_demos.html", "w").write(page); print(f"wrote real_demos.html ({len(page)/1e6:.1f} MB)")
