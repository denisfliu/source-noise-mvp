"""All relocated-gate poses in one room cloud (2026-09-25). The right gate's own points are removed from the scene and
redrawn at every pose (moved exactly as the renderer moves them), each with its aperture outline and sketch.

  /home/dfliu/code/tv/bin/python build_gate_poses_page.py
"""
import colorsys, json, math, os, sys

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__)); RD = os.path.dirname(SP)
sys.path.insert(0, SP); sys.path.insert(0, RD)
import cloudviewer  # noqa: E402
from build_reloc_page import POSES, moved_scene, pose_tag  # noqa: E402
from moved_gate_cell import GA, GB, PIVOT, ZHI, ZLO  # noqa: E402
from gsplat_scene_edit import _mask_mocap, load_duplicate_edit  # noqa: E402
from extract_scene_cloud import near_gates, table_mask  # noqa: E402

PER_GATE = 2500
BAD = ("-86,2.1,0.87", 60)   # our flights touched post A on the automatic sketch; now flown on a wider detour (2026-09-25)
ARMS = [("ours", "rr25mg", [96, 235, 160]), ("SDEdit on pi0", "sde25mg", [90, 170, 240]),
        ("velocity projection on pi0", "vproj25mg", [180, 140, 255])]


def bad_pose_viewer():
    """One pose, all three methods: the gate moved as the renderer moves it, the sketch, every flight (orange if it
    touches the gate before the route ends), and a red tick at each touching flight's closest approach."""
    import glob, torch
    import gate_clearance as G
    from route_contact import BODY, route_end
    spec, seed = BAD
    dyaw, dx, dy = map(float, spec.split(",")); pt = pose_tag(spec, seed)
    R, t = se2(spec); moved_scene(R, t)
    cloud = torch.tensor(np.asarray(G.moved_gate_cloud(dyaw, (dx, dy)), np.float32))
    sk = np.asarray(json.load(open(f"{RD}/sketch_mg_rr25mg{pt}.json"))["points"], np.float64)[:, :3]
    a, b = R @ GA + t, R @ GB + t
    ap = np.array([[a[0], a[1], ZLO], [b[0], b[1], ZLO], [b[0], b[1], ZHI], [a[0], a[1], ZHI], [a[0], a[1], ZLO]], np.float32)
    groups, ticks, rows = [], [], []
    for lab, pre, col in ARMS:
        ok, bad = [], []
        for f in sorted(glob.glob(f"/home/dfliu/ctxrun/traj_{pre}{pt}_[0-9].npy")):
            P = np.load(f)[:, :3].astype(np.float64); e = route_end(P, sk)
            d = torch.cdist(torch.from_numpy(P[:e + 1].astype(np.float32)), cloud).min(1).values.numpy(); j = int(d.argmin())
            (ok if d[j] >= BODY else bad).append(P.astype(np.float32)); rows.append((lab, f.rsplit("_", 1)[1][:-4], float(d[j]), P[j]))
            if d[j] < BODY:
                ticks.append(np.stack([P[j] - [0, 0, 0.12], P[j] + [0, 0, 0.12]]).astype(np.float32))
        if ok: groups.append({"label": f"{lab}: no contact ({len(ok)})", "color": col, "trajs": ok})
        if bad: groups.append({"label": f"{lab}: touches the gate ({len(bad)})", "color": [255, 140, 40], "trajs": bad})
    groups += [{"label": "closest approach of each touching flight", "fixed": True, "color": [240, 80, 80], "trajs": ticks},
               {"label": "the sketch", "fixed": True, "color": [255, 200, 90], "trajs": [sk.astype(np.float32)]},
               {"label": "moved aperture (judge)", "fixed": True, "color": [124, 208, 240], "trajs": [ap]}]
    v = cloudviewer.viewer_html("reloctmp", groups, elem_id="v1", max_pts=None,
                                note="Orange flights touch the gate before the route ends; red ticks mark where. Yellow is the sketch.")
    os.remove(f"{SP}/scene_cloud_reloctmp.npz")
    tab = ("<table class='rt'><tr><th>method</th><th>flight</th><th>closest to the gate before the route end</th><th>where (x, y, z)</th></tr>"
           + "".join(f"<tr class='{'' if dm >= BODY else 'gz'}'><td>{l}</td><td>{fl}</td><td>{dm:.2f} m</td><td>({q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f})</td></tr>"
                     for l, fl, dm, q in rows) + "</table>")
    return v + tab



def se2(spec):
    dyaw, dx, dy = map(float, spec.split(","))
    th = math.radians(dyaw); R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return R, PIVOT - R @ PIVOT + np.array([dx, dy])


def main():
    Z = np.load(f"{SP}/scene_cloud_right_full.npz"); pts, rgb = Z["pts"].astype(np.float32), Z["rgb"]
    # the renderer's own gate selection (a hand box here used to catch the goal table and move it with each gate)
    gm = _mask_mocap(pts.astype(np.float64), load_duplicate_edit("right_and_center"))
    gm &= pts[:, 2] > 0.1   # leave the floor patch under the original gate in place
    gp, gc = pts[gm], rgb[gm]
    k = np.random.default_rng(0).permutation(len(gp))[:PER_GATE]; gp, gc = gp[k], gc[k]
    P, Cc = [pts[~gm]], [rgb[~gm]]
    groups = []
    for j, (spec, seed) in enumerate(POSES):
        R, t = se2(spec)
        q = gp.copy(); q[:, :2] = (R @ gp[:, :2].T).T + t
        col = (np.array(colorsys.hsv_to_rgb(j / len(POSES), 0.75, 1.0)) * 255).astype(np.uint8)
        P.append(q); Cc.append((0.35 * gc + 0.65 * col).astype(np.uint8))
        a, b = R @ GA + t, R @ GB + t
        ap = np.array([[a[0], a[1], ZLO], [b[0], b[1], ZLO], [b[0], b[1], ZHI], [a[0], a[1], ZHI], [a[0], a[1], ZLO]], np.float32)
        name = f"gate turned {float(spec.split(',')[0]):+g}°, moved ({spec.split(',')[1]}, {spec.split(',')[2]}) m"
        sk = np.asarray(json.load(open(f"{RD}/sketch_mg_rr25mg{pose_tag(spec, seed)}.json"))["points"], np.float32)[:, :3]
        groups.append({"label": name, "color": col.tolist(), "trajs": [ap, sk]})
    P, Cc = np.concatenate(P), np.concatenate(Cc)
    moved = P[-len(POSES) * len(gp):]                  # the moved gates are appended last
    m = near_gates(P, moved, keep=np.r_[np.zeros(len(P) - len(moved), bool), np.ones(len(moved), bool)] | table_mask(P))
    np.savez(f"{SP}/scene_cloud_posestmp.npz", pts=P[m], rgb=Cc[m])
    v = cloudviewer.viewer_html("posestmp", groups, elem_id="v0", max_pts=None,
                                note="Each gate is drawn at its pose in its own colour; tick a pose to show its opening and sketch. "
                                     "Grey gates are the first pose set, clustered at the original gate and replaced on 2026-09-25.")
    os.remove(f"{SP}/scene_cloud_posestmp.npz")
    page = f"""<title>Relocated Gate Poses</title>
<style>
:root{{--bg:#0f1216;--card:#151a21;--line:#28303c;--ink:#e4e9f1;--mut:#8b94a5}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;padding:28px 18px 70px}}
main{{max-width:1100px;margin:0 auto}} h1{{font-size:23px;margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 14px;max-width:92ch}}
.vc{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:10px}}
.v3dwrap canvas{{width:100%;border-radius:6px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace}}
.lg{{display:inline-flex;align-items:center;gap:5px;cursor:pointer}} .sw{{width:11px;height:11px;border-radius:3px;display:inline-block}}
.ct{{color:var(--mut)}} h2{{font-size:17px;margin:28px 0 6px;color:#7cd0f0}} table.rt{{border-collapse:collapse;font:12px ui-monospace,Menlo,monospace;margin:10px 0 0;font-variant-numeric:tabular-nums}} .rt td,.rt th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}} tr.gz td{{color:#ffb070}} .bgbtn{{font:12px ui-monospace,Menlo,monospace;padding:3px 9px;border-radius:5px;border:1px solid var(--line);background:transparent;color:inherit;cursor:pointer}} .hint{{color:var(--mut);margin-left:auto}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:95ch}}
</style>
<main><h1>Relocated Gate Poses</h1>
<p class="sub">The right gate at each of the 13 poses in the relocated-gate test, all in one room; three sit behind the starting point. The original right gate is
removed; every pose is drawn where the renderer puts it.</p>
<div class="vc">{v}</div>
<h2>Gate turned -86°, moved (2.1, 0.87) m, with a wider path around the first post</h2>
<p class="sub">On the automatic sketch every one of our flights cut the corner at post A and touched it (0.11-0.15 m). The sketch
now swings 0.8 m past that post and 0.8 m out before turning in behind the gate; all three methods fly it clean.</p>
<div class="vc">{bad_pose_viewer()}</div></main>"""
    open(f"{SP}/gate_poses.html", "w").write(page); print("wrote gate_poses.html")


if __name__ == "__main__":
    main()
