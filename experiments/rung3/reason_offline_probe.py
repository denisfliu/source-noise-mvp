"""Offline sanity probe for the VLM movement reasoner (vlmenv): show demo frames from the atomic
tasks to the reasoner with the task instruction, convert its words to the world frame with the
frame's own heading, and compare with what the demonstrator actually did over the next 50 steps.
Prints every trace so the reasoning can be read.

  ~/code/vlmenv/bin/python reason_offline_probe.py [--eps 0 5 55 60] [--fracs 0.2 0.5 0.8]
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
from vlm_reason_server import Reasoner  # noqa: E402

PROMPTS = {"left": "go through the gate on the left and hover over the stuffed animal",
           "right": "go through the gate on the right and hover over the stuffed animal",
           "cfl": "go through the center gate from the left and hover over the stuffed animal",
           "cfr": "go through the center gate from the right and hover over the stuffed animal"}


def task_of(ep):  # data_gate_synth3 layout: 50 episodes per atomic task in this order
    return ["left", "right", "cfl", "cfr"][min(ep // 50, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="+", default=[0, 5, 55, 60, 105, 155])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    a = ap.parse_args()
    R = Reasoner(a.model)
    cos_all = []
    for e in a.eps:
        d = np.load(f"{RD}/data_gate_synth3/ep_{e:04d}.npz", allow_pickle=True)
        st, img = d["state"], d["image"]
        keys = [k for k in d.files if k not in ("image", "wrist", "state", "action")]
        task = task_of(e); instr = PROMPTS[task]
        print(f"\n=== ep {e} ({task}); extra keys {keys}; T={len(st)}")
        prev = []
        for fr in a.fracs:
            t = int(fr * (len(st) - 51)); pos, psi = st[t, :3], float(st[t, 3])
            truth = st[t + 50, :3] - pos
            fwd = np.array([np.cos(psi), np.sin(psi)]); rgt = np.array([np.sin(psi), -np.cos(psi)])
            d0 = pos - st[0, :3]
            body_off = [float(d0[:2] @ fwd), float(d0[:2] @ rgt), float(d0[2])]
            flown = float(np.linalg.norm(np.diff(st[:t + 1, :3], axis=0), axis=1).sum())
            ans, raw = R(Image.fromarray(img[t]), instr, flown, body_off, prev)
            prev.append(ans)
            f, r, u = ans["forward_m"], ans["right_m"], ans["up_m"]
            world = np.array([f * np.cos(psi) + r * np.sin(psi), f * np.sin(psi) - r * np.cos(psi), u])
            cs = float(world[:2] @ truth[:2] / (np.linalg.norm(world[:2]) * np.linalg.norm(truth[:2]) + 1e-9))
            truth_body = [float(truth[:2] @ fwd), float(truth[:2] @ rgt), float(truth[2])]
            cos_all.append(cs)
            print(f"  t={t:3d} pos={pos.round(2)} hdg={np.degrees(psi):6.1f}  demo next 50 (body): fwd {truth_body[0]:+.2f} right {truth_body[1]:+.2f} up {truth_body[2]:+.2f}"
                  f" | VLM: fwd {f:+.2f} right {r:+.2f} up {u:+.2f} turn {ans['turn_deg']:+.0f}  cos_xy={cs:+.2f}  ({ans.get('ms', '?')} ms)")
            print(f"        seen: {ans['seen'][:110]}\n        done: {ans['done'][:60]} | next: {ans['next'][:60]}")
    cos_all = np.array(cos_all)
    print(f"\nmean cos(xy) VLM vs demo over {len(cos_all)} frames: {cos_all.mean():+.2f}; fraction > 0.5: {(cos_all > 0.5).mean():.2f}")


if __name__ == "__main__":
    main()
