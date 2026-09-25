"""Relocated-gate page (2026-09-24): per gate pose, the right-scene cloud with the gate's points moved exactly as
the renderer moves them (about gate_clearance.PIVOT), the sketch, and every arm's flights coloured by verdict.

  /home/dfliu/code/tv/bin/python build_reloc_page.py --arm "pin=rr25mg" --arm "SDEdit=sde25mg" --arm "v-proj=vproj25mg"
Verdicts: route-clean from moved_gate_cell --score (the "OK"/"fail" lines in the score files); contact = closer than
0.18 m to the moved gate cloud, counted up to the route end (first time within 0.25 m of the sketch's last point)
and, separately, over the whole flight (which includes loitering after the route).
"""
import argparse, glob, html, json, math, os, re, sys

import numpy as np
import torch

SP = os.path.dirname(os.path.abspath(__file__)); RD = os.path.dirname(SP)
sys.path.insert(0, SP); sys.path.insert(0, RD)
import cloudviewer  # noqa: E402
import gate_clearance as G  # noqa: E402
from moved_gate_cell import PIVOT  # noqa: E402

RUN = "/home/dfliu/ctxrun"
BODY, END_R = 0.18, 0.25
# spelled exactly as the run scripts pass GATE_TF (the tag is this string with "-" -> "m" and "," "." -> "_")
POSES = ["-45,0,0", "-25,0,0", "25,0,0", "45,0,0", "90,0,0", "0,0.5,-0.3", "30,-0.4,0.4",
         "100,-1.26,1.50", "180,0,0", "-35,1.34,0.75", "90,0.74,1.95", "90,-0.26,0.85"]
SEED0 = 41   # run scripts number the poses 41..52 in this order
COLS = [[96, 235, 160], [90, 170, 240], [180, 140, 255]]
CONTACT, FAIL = [255, 140, 40], [240, 80, 80]
GA, GB = np.array([0.195, -1.348]), np.array([0.924, -0.952])   # original aperture posts (scene right)


def pose_tag(spec, seed):
    return spec.translate(str.maketrans({"-": "m", ",": "_", ".": "_"})) + f"_{seed}"


def route_verdicts():
    v = {}
    for f in glob.glob(f"{RUN}/*scores*.txt"):
        for m in re.finditer(r"(traj_\S+?)\.npy cross=.*?(OK|fail)\s*$", open(f).read(), re.M):
            v[m.group(1)] = m.group(2) == "OK"
    return v


def axes(length=0.5):
    o = np.zeros(3, np.float32)
    return [{"label": f"origin: +{n} axis (0.5 m)", "fixed": True, "color": c, "trajs": [np.stack([o, o + length * e])]}
            for n, c, e in [("x", [240, 80, 80], np.eye(3, dtype=np.float32)[0]),
                            ("y", [80, 220, 80], np.eye(3, dtype=np.float32)[1]),
                            ("z", [90, 140, 255], np.eye(3, dtype=np.float32)[2])]]


def moved_scene(R, t):
    Z = np.load(f"{SP}/scene_cloud_right.npz"); pts, rgb = Z["pts"].astype(np.float32), Z["rgb"]
    tv = (GB - GA) / np.linalg.norm(GB - GA); nv = np.array([tv[1], -tv[0]]); rel = pts[:, :2] - GA
    m = (np.abs(rel @ nv) < 0.25) & ((rel @ tv) > -0.35) & ((rel @ tv) < 1.15) & (pts[:, 2] > 0.1)
    pts[m, :2] = (R @ pts[m, :2].T).T + t
    np.savez(f"{SP}/scene_cloud_reloctmp.npz", pts=pts, rgb=rgb)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", action="append", required=True, help="label=tag-prefix")
    ap.add_argument("--out", default="reloc_gates.html"); a = ap.parse_args()
    arms = [s.rsplit("=", 1) for s in a.arm]; RV = route_verdicts()
    secs, tot = [], {lab: [0, 0, 0, 0] for lab, _ in arms}   # flights, route-clean, contact-free to route end, whole flight
    for i, spec in enumerate(POSES):
        dyaw, dx, dy = map(float, spec.split(",")); pt = pose_tag(spec, SEED0 + i)
        th = math.radians(dyaw); R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        t = PIVOT - R @ PIVOT + np.array([dx, dy])
        moved_scene(R, t)
        C = torch.tensor(np.asarray(G.moved_gate_cloud(dyaw, (dx, dy)), np.float32))
        sk = np.asarray(json.load(open(f"{RD}/sketch_mg_{arms[0][1]}{pt}.json"))["points"], np.float32)[:, :3]
        a2, b2 = R @ GA + t, R @ GB + t
        apt = np.array([[a2[0], a2[1], 0.2], [b2[0], b2[1], 0.2], [b2[0], b2[1], 1.95], [a2[0], a2[1], 1.95], [a2[0], a2[1], 0.2]], np.float32)
        groups, rows = [], []
        for gi, (lab, pre) in enumerate(arms):
            ok, ct, bad = [], [], []
            for f in sorted(glob.glob(f"{RUN}/traj_{pre}{pt}_[0-9]*.npy")):
                P = np.load(f)[:, :3].astype(np.float32); name = os.path.basename(f)[:-4]
                d = torch.cdist(torch.from_numpy(P), C).min(1).values.numpy()
                k = np.where(np.linalg.norm(P - sk[-1], axis=1) < END_R)[0]; e = k[0] if len(k) else len(P)
                route, dr, dw = RV.get(name, False), float(d[:e + 1].min()), float(d.min())
                rows.append((lab, name.rsplit("_", 1)[1], route, dr, dw))
                s = tot[lab]; s[0] += 1; s[1] += route; s[2] += route and dr >= BODY; s[3] += route and dw >= BODY
                (ok if route and dr >= BODY else ct if route else bad).append(P)
            if ok: groups.append({"label": f"{lab}: no contact ({len(ok)})", "color": COLS[gi % 3], "trajs": ok})
            if ct: groups.append({"label": f"{lab}: contact before route end ({len(ct)})", "color": CONTACT, "trajs": ct})
            if bad: groups.append({"label": f"{lab}: route failure ({len(bad)})", "color": FAIL, "trajs": bad})
        groups += [{"label": "the sketch", "fixed": True, "color": [255, 200, 90], "trajs": [sk]},
                   {"label": "moved aperture (judge)", "fixed": True, "color": [124, 208, 240], "trajs": [apt]}] + axes()
        tab = "<table class='rt'><tr><th>arm</th><th>trial</th><th>route-clean</th><th>closest to gate, to route end</th><th>whole flight</th></tr>" + "".join(
            f"<tr class='{'ok' if r and dr >= BODY else 'gz' if r else 'bad'}'><td>{html.escape(l)}</td><td>{tr}</td><td>{'yes' if r else 'no'}</td>"
            f"<td>{dr:.2f} m</td><td>{dw:.2f} m</td></tr>" for l, tr, r, dr, dw in rows) + "</table>"
        v = cloudviewer.viewer_html("reloctmp", groups, elem_id=f"v{i}", max_pts=40000,
                                    note="Orange: completed the route but came within 0.18 m of the gate before the route end. Red: route failure.")
        secs.append((f"gate turned {dyaw:+g}°, moved ({dx:g}, {dy:g}) m", v + tab))
    os.remove(f"{SP}/scene_cloud_reloctmp.npz")
    summ = "".join(f"<li><b>{html.escape(l)}</b>: {s[1]}/{s[0]} route-clean, {s[2]}/{s[0]} without touching the gate before the route end, "
                   f"{s[3]}/{s[0]} over the whole flight</li>" for l, s in tot.items())
    body = "".join(f"<h2>{html.escape(h)}</h2><div class='vc'>{x}</div>" for h, x in secs)
    page = f"""<title>Relocated Gates</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5;--acc:#7cd0f0}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}} h1{{font-size:23px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 18px;max-width:92ch}}
h2{{font-size:17px;margin:28px 0 8px;color:var(--acc)}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px;margin:12px 0}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} .hint{{color:var(--mut);margin-left:auto}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
table.rt{{border-collapse:collapse;font:12px ui-monospace,Menlo,monospace;margin:10px 0 0;font-variant-numeric:tabular-nums}}
.rt td,.rt th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}} .rt th{{color:var(--acc);font-weight:500}}
tr.gz td{{color:#ffb070}} tr.bad td{{color:#f07070}}
</style>
<main><h1>Relocated Gates</h1>
<p class="sub">The right gate moved to 12 poses it never occupied in the demonstrations; each pose gets an automatic sketch
through the new opening, flown 5 times per arm with a 25-step replan. The gate points in each cloud are moved exactly as
the renderer moves them.</p>
<ul class="sub">{summ}</ul>
{body}</main>"""
    open(os.path.join(SP, a.out), "w").write(page); print(f"wrote {a.out} ({len(page) / 1e6:.1f} MB)")
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
