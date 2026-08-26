#!/usr/bin/env python3
"""Empirically calibrate commanded invariant (normalized units) -> realized EE
displacement (meters), using a pinned arm C policy as the probe instrument.

For N trials: reset a LIBERO env, settle, record EE pos, request ONE action
chunk with a commanded invariant via the snmvp policy server, execute the
whole chunk open-loop, record EE pos again. Fit a per-dim linear map
  delta_ee[d] = slope[d] * command[d] + offset[d]     (d = x, y, z)
and report R^2. The inverse map converts a desired physical displacement into
a normalized-unit command — the core of the sim-state oracle.

Requires serve_snmvp_policy.py running (it consumes `snmvp_invariant`).
Run in the LIBERO client venv with the usual PYTHONPATH/LIBERO_CONFIG_PATH/
MUJOCO_GL env; see eval_checkpoint.sh for the pattern.
"""

import argparse
import dataclasses
import json
import math
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--command-scale", type=float, default=60.0,
                    help="stddev of random xyz commands in normalized units "
                         "(dataset per-dim invariant std is ~22-77)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    pairs = []
    for i in range(args.n_trials):
        env.reset()
        obs = env.set_init_state(init_states[i % len(init_states)])
        for _ in range(10):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        cmd = np.zeros(7, dtype=np.float32)
        cmd[:3] = rng.normal(0.0, args.command_scale, size=3)
        cmd[6] = rng.normal(0.0, 20.0)  # exercise gripper dim too

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        element = {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, 224, 224)),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist, 224, 224)),
            "observation/state": np.concatenate(
                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                 obs["robot0_gripper_qpos"])),
            "prompt": str(task.language),
            "snmvp_invariant": cmd.tolist(),
        }
        ee_before = np.array(obs["robot0_eef_pos"])
        chunk = client.infer(element)["actions"]
        for a in chunk:  # execute the FULL chunk open-loop
            obs, _, done, _ = env.step(np.asarray(a).tolist())
        ee_after = np.array(obs["robot0_eef_pos"])
        pairs.append({"command": cmd.tolist(),
                      "delta_ee": (ee_after - ee_before).tolist()})
        print(f"trial {i:2d}: cmd_xyz={np.round(cmd[:3],1).tolist()} "
              f"dee={np.round(ee_after-ee_before,4).tolist()}", flush=True)
    env.close()

    C = np.array([p["command"][:3] for p in pairs])
    E = np.array([p["delta_ee"] for p in pairs])
    fit = {}
    for d, name in enumerate("xyz"):
        A = np.stack([C[:, d], np.ones(len(C))], axis=1)
        (slope, offset), res, *_ = np.linalg.lstsq(A, E[:, d], rcond=None)
        pred = A @ np.array([slope, offset])
        ss_res = float(((E[:, d] - pred) ** 2).sum())
        ss_tot = float(((E[:, d] - E[:, d].mean()) ** 2).sum())
        fit[name] = {"slope_m_per_unit": float(slope), "offset_m": float(offset),
                     "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None}

    out = {"n_trials": len(pairs), "command_scale": args.command_scale,
           "task_id": args.task_id, "seed": args.seed,
           "fit_xyz": fit, "pairs": pairs}
    p = pathlib.Path(args.out).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)
    print("CALIB_SUMMARY:", json.dumps({k: {kk: round(vv, 5) if vv is not None else None
                                            for kk, vv in v.items()} for k, v in fit.items()}))


if __name__ == "__main__":
    main()
