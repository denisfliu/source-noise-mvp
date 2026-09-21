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
"""
import argparse, json, os, sys, time

DEF = os.path.expanduser("~/ctxrun/agent_sim")


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


def show(o):
    p = o["proposal"]
    print(f"decision k={o['k']}  pose x={o['pose'][0]:.2f} y={o['pose'][1]:.2f} z={o['pose'][2]:.2f} "
          f"heading={o['pose'][3]:.2f} rad ({o['pose'][3]*57.3:+.0f} deg; --yaw is in this same sense, positive turns left)")
    print(f"the policy intends, over the next {o['seconds_per_decision']} s:")
    print(f"  net move  dx {p['net_xyz'][0]:+.2f}  dy {p['net_xyz'][1]:+.2f}  dz {p['net_xyz'][2]:+.2f} m,  yaw {p['net_yaw_deg']:+.1f} deg")
    print(f"  ending at x {p['path_end'][0]:.2f}  y {p['path_end'][1]:.2f}  z {p['path_end'][2]:.2f}   (peak speed {p['speed_max_mps']:.2f} m/s, trust {p['sigma_serve']:.2f})")
    print(f"  view: {o['view']}   (left half = forward camera, right half = downward camera)")
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
    a.add_argument("cmd", choices=["wait", "look", "approve", "override", "stop"])
    a.add_argument("--dir", default=DEF); a.add_argument("--k", type=int); a.add_argument("--why", default="")
    a.add_argument("--forward", type=float, default=0.0); a.add_argument("--left", type=float, default=0.0)
    a.add_argument("--up", type=float, default=0.0); a.add_argument("--yaw", type=float, default=0.0)
    a.add_argument("--sigma", type=float, default=0.0); a.add_argument("--timeout", type=float, default=900.0)
    g = a.parse_args(); d = g.dir
    if g.cmd in ("wait", "look"):
        t0 = time.time()
        while True:
            o = latest(d)
            pend = os.path.exists(os.path.join(d, "cmd.json"))
            done = o and os.path.exists(os.path.join(d, "cmds", f"{o['k']:03d}.json"))
            if o and not pend and not done:
                show(o); return
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
        write(d, {"k": g.k, "verdict": "approve", "why": g.why})
    else:
        write(d, {"k": g.k, "verdict": "override", "why": g.why,
                  "move": {"forward": g.forward, "left": g.left, "up": g.up, "yaw_deg": g.yaw, "sigma": g.sigma}})
    print(f"k={g.k} {g.cmd}: {g.why}")


if __name__ == "__main__":
    main()
