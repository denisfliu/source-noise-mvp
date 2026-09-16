"""Hardware flights over the scene clouds (2026-09-15): the real-only pin's five left and five right
flights, with the 2026-09-11 baseline and xswap left sessions for context.

The flight node's traj_<trial>.npy files written before 2026-09-15 are the SETPOINT stream, not the
drone (policy_node_gate.py appended the integrated target per step; fixed in dronevla2.0 c933466).
Only every apc-th row is a measured mocap pose (the pose the chunk was integrated from). Each flight
is therefore drawn twice: the setpoint stream (thin) and the measured poses joined by chords (thick
markers); the judge lines in the notes come from the chords.

  python build_hw_page.py            -> hw_flights.html
"""
import glob, os, sys
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(SP))
import cloudviewer
from build_traj_page import marks
import realism as R
import gate_success as G

FL = os.path.expanduser("~/gate_flights")
GOOD, MID, BAD, SETP = [96, 235, 160], [255, 200, 70], [255, 95, 95], [150, 160, 175]

def live(P, apc):
    L = P[0::apc]
    D = [a + (b - a) * np.linspace(0, 1, 50, endpoint=False)[:, None] for a, b in zip(L[:-1], L[1:])]
    return L, np.concatenate(D + [L[-1:]]).astype(np.float32)

def cell(tag, side, apc):
    files = sorted(glob.glob(f"{FL}/traj_{tag}_0?.npy"))
    rows, setp, meas = [], [], []
    for f in files:
        P = np.load(f)[:, :3].astype(np.float32); L, D = live(P, apc); v = G.judge(D, side)
        lag = np.linalg.norm(P[apc - 1::apc][:len(L) - 1] - L[1:], axis=1)
        jump = np.linalg.norm(np.diff(L, axis=0), axis=1).max()
        ok = v["transit"] and v["wrong_dir_crossings"] == 0
        col = GOOD if ok and v["goal_after_transit"] else (MID if ok else BAD)
        name = os.path.basename(f)[5:-4]
        rows.append("%s: route-clean transit %s, goal box %s, %d measured poses %.1f s apart, drone trailed the setpoint by %.2f m (max %.2f); end %s%s" % (
            name, "yes" if ok else ("wrong-direction pass" if v["transit"] else "no"),
            "yes" if v["goal_after_transit"] else "no", len(L), apc / 10, np.median(lag), lag.max(),
            np.round(L[-1], 2).tolist(), "; pose jump %.1f m between two samples" % jump if jump > 1.2 else ""))
        setp.append({"label": f"{name} setpoint stream", "color": SETP, "trajs": [P]})
        meas.append({"label": f"{name} measured poses (chords)", "color": col, "trajs": [D]})
    return meas + setp, rows

def section(scene, groups, elem, note):
    return cloudviewer.viewer_html(scene, groups + marks(scene), elem_id=elem, max_pts=40000, note=note)

demos = R.load_demos()
f32 = lambda ps: [p[:, :3].astype(np.float32) for p in ps]
ref_l = {"label": "pilot demonstrations, left (50)", "color": [70, 90, 120], "trajs": f32(demos["real"][0])}
ref_r = {"label": "pilot demonstrations, right (50)", "color": [70, 90, 120], "trajs": f32(demos["real"][1])}
LEG = "Green: route-clean transit and goal box. Amber: route-clean transit, goal box missed (the box sits at the simulator's z = 1.0; the pilot's own flights end at z 1.25-1.6 and hit it only 33/50 left, 25/50 right). Red: no transit. Grey: the setpoint stream the node published; the coloured chords join the measured poses, one per replan."
secs = []
for title, tag, side, apc, ref, elem in [
    ("Real-only pin, left gate, 2026-09-15 (5 flights, apc 50)", "realonly_left", "left", 50, ref_l, "v_rl"),
    ("Real-only pin, right gate, 2026-09-15 (5 flights, apc 50)", "realonly_right", "right", 50, ref_r, "v_rr"),
    ("pi0 baseline, left gate, 2026-09-11 (5 flights, apc 8: poses every 0.8 s)", "baseline_left", "left", 8, ref_l, "v_bl"),
    ("Pin with swap (xswap), left gate, 2026-09-11 (5 flights, apc 8; the session whose setpoints were not followed)", "ours_left", "left", 8, ref_l, "v_ol")]:
    groups, rows = cell(tag, side, apc)
    secs.append((title, section(side, [ref] + groups, elem, LEG), rows))
body = "".join(f"<h2>{t}</h2><div class='vc'>{h}</div><ul class='rows'>" + "".join(f"<li>{r}</li>" for r in rows) + "</ul>" for t, h, rows in secs)
page = f"""<title>Hardware Flights, Real-Only Pin</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}} h1{{font-size:23px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px;color:var(--acc)}} .sub{{color:var(--mut);margin:0 0 18px;max-width:92ch}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}}
.rows{{font:13px/1.5 ui-monospace,Menlo,monospace;color:var(--mut);padding-left:18px}} .rows li{{margin:2px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
</style>
<main>
<h1>Hardware flights over the scene clouds</h1>
<p class="sub">Real drone, mocap frame, over the simulator's reconstructions. The node's trajectory files from these sessions are the published setpoint stream (grey); the drone's measured position is known only at each replan (coloured chords), every 5 s at apc 50. Judge apertures in blue, goal box in yellow, pilot demonstrations in dark blue for reference. Toggle groups in each legend; drag to orbit, wheel to zoom.</p>
{body}
</main>
"""
out = os.path.join(SP, "hw_flights.html"); open(out, "w").write(page); print("wrote", out, len(page) // 1024, "kB")
