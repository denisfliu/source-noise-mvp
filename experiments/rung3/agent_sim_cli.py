#!/usr/bin/env python3
"""Reviewer side of the in-the-loop simulator flight (2026-09-20).

gate_rollout_batch.py with AGENT_DIR set decodes the policy's own command at every replan and blocks
until this CLI answers. The reviewer sees the two camera views and the coarse plan the policy intends
to execute, and either approves it or replaces it with a movement primitive, giving a reason either way.
The reason is burned into the video.

  python agent_sim_cli.py wait                      block until the flight asks; print pose + proposal
  python agent_sim_cli.py approve --k K --why "..."
  python agent_sim_cli.py override --k K --why "..." [--forward m] [--left m] [--up m] [--yaw deg] [--sigma s]
  python agent_sim_cli.py stop --k K --why "..."
  python agent_sim_cli.py prims                     print the movement primitives (--prim NAME [NAME ...] on override)
"""
import argparse, glob, json, math, os, sys, time

DEF = os.path.expanduser("~/ctxrun/agent_sim")

# Movement primitives (Denis, 2026-09-23): a fixed vocabulary in the drone's own frame, the same for every arm.
# Several may be given in one override; they are summed into one move (the yaw and the translation run together).
PRIMS = {}
for _n, _m in ((0.5, "05"), (1.0, "1"), (2.0, "2")):
    PRIMS[f"forward_{_m}"] = {"forward": _n}; PRIMS[f"left_{_m}"] = {"left": _n}; PRIMS[f"right_{_m}"] = {"left": -_n}
for _n, _m in ((0.5, "05"), (1.0, "1")):
    PRIMS[f"back_{_m}"] = {"forward": -_n}
for _n, _m in ((0.5, "05"), (1.0, "1"), (2.0, "2")):   # 45-degree diagonals, heading unchanged (Denis, 2026-09-23)
    PRIMS[f"diag_left_{_m}"] = {"forward": _n * 0.7071, "left": _n * 0.7071}; PRIMS[f"diag_right_{_m}"] = {"forward": _n * 0.7071, "left": -_n * 0.7071}
PRIMS["up_05"] = {"up": 0.5}; PRIMS["down_05"] = {"up": -0.5}
for _d in (15, 30, 45):
    PRIMS[f"turn_left_{_d}"] = {"yaw_deg": float(_d)}; PRIMS[f"turn_right_{_d}"] = {"yaw_deg": -float(_d)}
PRIMS["hold"] = {}


def prims_text():
    return ("\n".join([
        "  forward_05  forward_1  forward_2     move along your heading 0.5 / 1 / 2 m",
        "  back_05  back_1                      move against your heading 0.5 / 1 m",
        "  left_05  left_1  left_2              sidestep to your left 0.5 / 1 / 2 m (heading unchanged)",
        "  right_05  right_1  right_2           sidestep to your right 0.5 / 1 / 2 m",
        "  diag_left_05/1/2  diag_right_05/1/2  move 0.5 / 1 / 2 m at 45 degrees forward-left / forward-right (heading unchanged)",
        "  up_05  down_05                       climb / descend 0.5 m",
        "  turn_left_15/30/45  turn_right_15/30/45   turn in place by that many degrees",
        "  hold                                 stay where you are",
        "Combine up to two, e.g. --prim turn_left_30 forward_1 (the turn and the move run together)."]))


def latest(d):
    p = os.path.join(d, "latest.json")
    for _ in range(6):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except json.JSONDecodeError:
                time.sleep(0.05)
        else:
            return None
    return None


def show(o, d):
    p = o["proposal"]
    print(f"decision k={o['k']}  pose x={o['pose'][0]:.2f} y={o['pose'][1]:.2f} z={o['pose'][2]:.2f} "
          f"heading={o['pose'][3]:.2f} rad ({o['pose'][3]*57.3:+.0f} deg; --yaw is in this same sense, positive turns left)")
    print(f"the policy intends, over the next {o['seconds_per_decision']} s (the part of its plan that will "
          f"actually execute before you are asked again):")
    print(f"  net move  dx {p['net_xyz'][0]:+.2f}  dy {p['net_xyz'][1]:+.2f}  dz {p['net_xyz'][2]:+.2f} m,  yaw {p['net_yaw_deg']:+.1f} deg")
    print(f"  ending at x {p['path_end'][0]:.2f}  y {p['path_end'][1]:.2f}  z {p['path_end'][2]:.2f}   (peak speed {p['speed_max_mps']:.2f} m/s, trust {p['sigma_serve']:.2f})")
    if o.get("last_execution"):
        print(f"  your last command: {o['last_execution']}")
    print(f"  view: {o['view']}")
    print("    top left = forward camera now, top right = downward camera now"
          + (", strip below = what happened during your last move, in time order" if o.get("during_last_move") else ""))
    # The reviewer's own decision log, replayed to it every time (Denis, 2026-09-21): with two dozen
    # decisions and an image each, its earliest reasoning is buried far up its context and it starts
    # repeating itself. These are its own words, read back from what it wrote.
    past = sorted(glob.glob(os.path.join(d, "cmds", "*.json")))[-8:]
    if past:
        print("what you have decided so far (your own words):")
        for f in past:
            try:
                c = json.load(open(f))
            except json.JSONDecodeError:
                continue
            mv = c.get("move") or {}
            terse = " ".join(f"{k.replace('_deg','')} {v:+g}" for k, v in mv.items() if v)
            why = (c.get("why") or "")[:160]
            print(f"    k={c.get('k')} {c.get('verdict','?'):<8} {terse:<28} {why}")
    h=o.get("history") or []
    if len(h)>1:
        print("where you have been (room coordinates, one row per decision):")
        for r in h:
            print(f"    k={r['k']}  x {r['x']:+.2f}  y {r['y']:+.2f}  z {r['z']:.2f}  heading {r['heading_deg']:+.0f} deg")


def write(d, cmd):
    tmp = os.path.join(d, "cmd.json.tmp")
    json.dump(cmd, open(tmp, "w")); os.replace(tmp, os.path.join(d, "cmd.json"))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd", choices=["wait", "look", "approve", "override", "stop", "prims"])
    a.add_argument("--dir", default=DEF); a.add_argument("--k", type=int); a.add_argument("--why", default="")
    a.add_argument("--forward", type=float, default=0.0); a.add_argument("--left", type=float, default=0.0)
    a.add_argument("--up", type=float, default=0.0); a.add_argument("--yaw", type=float, default=0.0)
    # Room-frame movement (2026-09-21, Denis): reviewers build their map in room coordinates, so let them
    # command in room coordinates and do the rotation here. --dx/--dy are metres along the room axes,
    # independent of where the drone is pointing; --dz is the same as --up.
    a.add_argument("--dx", type=float); a.add_argument("--dy", type=float); a.add_argument("--dz", type=float)
    a.add_argument("--prim", nargs="+", choices=sorted(PRIMS), metavar="NAME", help="movement primitives (see `prims`)")
    a.add_argument("--sigma", type=float, default=0.0); a.add_argument("--timeout", type=float, default=900.0)
    # --then-wait (2026-09-21): submit the verdict and block for the next decision in the same call, so a
    # decision costs two round trips (this call + reading the image) instead of three.
    a.add_argument("--then-wait", action="store_true", help="after answering, wait for and print the next decision")
    g = a.parse_args(); d = g.dir
    if g.cmd == "prims":
        print(prims_text()); return
    if g.cmd in ("wait", "look"):
        t0 = time.time()
        while True:
            o = latest(d)
            pend = os.path.exists(os.path.join(d, "cmd.json"))
            done = o and os.path.exists(os.path.join(d, "cmds", f"{o['k']:03d}.json"))
            if o and not pend and not done:
                show(o, d); return
            if os.path.exists(os.path.join(d, "done")):
                print("flight ended"); return
            if g.cmd == "look":
                sys.exit("no pending decision")
            if time.time() - t0 > g.timeout:
                sys.exit("timed out waiting for the flight")
            time.sleep(0.3)
    o = latest(d)
    if o is None or g.k is None or g.k != o["k"]:
        sys.exit(f"--k must be {o['k'] if o else '?'} (the decision you were shown)")
    if not g.why:
        sys.exit("--why is required: it is recorded and shown in the video")
    if g.cmd == "stop":
        write(d, {"k": g.k, "stop": True, "verdict": "approve", "why": g.why})
    elif g.cmd == "approve":
        armf = os.path.join(d, "arm")
        if os.path.exists(armf) and open(armf).read().strip() == "waypoints":
            sys.exit("this flight is the WAYPOINTS arm: nothing is proposed, so there is nothing to approve -- give a move with override")
        write(d, {"k": g.k, "verdict": "approve", "why": g.why})
    else:
        fwd, lft, up = g.forward, g.left, g.up
        if g.prim:
            if len(g.prim) > 2:
                sys.exit("at most two primitives per decision")
            if g.dx is not None or g.dy is not None or g.forward or g.left:
                sys.exit("give either --prim or --dx/--dy/--forward/--left, not both")
            mv = {}
            for n in g.prim:
                for k, v in PRIMS[n].items():
                    mv[k] = mv.get(k, 0.0) + v
            fwd, lft, up = mv.get("forward", 0.0), mv.get("left", 0.0), up + mv.get("up", 0.0)
            g.yaw = g.yaw + mv.get("yaw_deg", 0.0)
        if g.dx is not None or g.dy is not None or g.dz is not None:
            if g.forward or g.left:
                sys.exit("give either --forward/--left (drone frame) or --dx/--dy (room frame), not both")
            dx, dy = (g.dx or 0.0), (g.dy or 0.0)
            psi = float(o["pose"][3])
            fwd = math.cos(psi) * dx + math.sin(psi) * dy
            lft = -math.sin(psi) * dx + math.cos(psi) * dy
            if g.dz is not None:
                up = g.dz
            print(f"room-frame dx {dx:+.2f} dy {dy:+.2f} at heading {psi:+.2f} rad "
                  f"-> forward {fwd:+.2f}, left {lft:+.2f}")
        write(d, {"k": g.k, "verdict": "override", "why": g.why,
                  "move": {"forward": fwd, "left": lft, "up": up, "yaw_deg": g.yaw, "sigma": g.sigma}})
    print(f"k={g.k} {g.cmd}: {g.why}")
    if g.then_wait and g.cmd != "stop":
        t0 = time.time(); k_done = g.k
        while time.time() - t0 < g.timeout:
            o = latest(d)
            if (o and o["status"] == "stopped") or os.path.exists(os.path.join(d, "done")):
                print("flight ended"); return
            if o and o["status"] == "waiting" and o["k"] > k_done \
                    and not os.path.exists(os.path.join(d, "cmds", f"{o['k']:03d}.json")):
                print(); show(o, d); return
            time.sleep(0.25)
        sys.exit("timed out waiting for the next decision")


if __name__ == "__main__":
    main()
