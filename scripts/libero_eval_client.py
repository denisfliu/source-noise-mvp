#!/usr/bin/env python3
"""LIBERO eval client for Phase 1 — extends openpi's examples/libero/main.py with:

  --args.init-states-dir   directory of task{NN}.npy held-out init states
                           (from make_heldout_split.py); default: canonical
                           benchmark init states (standard protocol)
  --args.invariant-json    optional JSON {"invariant": [floats]} commanded on
                           EVERY policy call via the `snmvp_invariant` obs key
                           (requires serve_snmvp_policy.py server); used by the
                           wrong-invariant sim probe
  --args.out-json          write machine-readable per-task results

Runs in the LIBERO client venv (py3.8-compatible). Core rollout logic is kept
line-for-line from openpi's main.py so numbers stay comparable.
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    init_states_dir: Optional[str] = None  # held-out split dir (task{NN}.npy)
    invariant_json: Optional[str] = None  # commanded invariant for snmvp server
    # D2 minimal oracle: per-replan invariant = displacement toward the task's
    # first object-of-interest (from the BDDL goal — the same information the
    # language instruction carries), converted to normalized units via the
    # empirical calibration map. Reads ONLY <obj>_pos and robot0_eef_pos from
    # sim state; no task-phase logic. Rotation/gripper dims are commanded at
    # their dataset-mean invariant values (neutral).
    # "minimal2" (D5-b): 6-dim command (xyz + rot-means) leaving the GRIPPER
    # dim UNPINNED — vision owns grasping, the pin owns gross motion — and a
    # post-grasp target switch: bowl until it rises 3 cm off its start height,
    # then the goal object + a fixed 0.10 m hover offset. Documented sim-state
    # reads: robot0_eef_pos, <obj>_pos for the two objects in the BDDL goal.
    oracle: Optional[str] = None  # None | "minimal" | "minimal2"
    calibration_json: Optional[str] = None  # invariant_calibration.json
    stats_json: Optional[str] = None  # invariant_stats.json (dims 3..6 means)
    oracle_clip_m: float = 0.25  # clip |displacement| per dim (calibration range)
    out_json: Optional[str] = None

    video_out_path: str = "data/libero/videos"
    save_videos: bool = True
    seed: int = 7


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name}")
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    max_steps = MAX_STEPS[args.task_suite_name]

    invariant = None
    if args.invariant_json:
        with open(pathlib.Path(args.invariant_json).expanduser()) as f:
            invariant = [float(x) for x in json.load(f)["invariant"]]
        logging.info(f"Commanding invariant on every call: {invariant}")

    oracle_fn = None
    if args.oracle == "minimal2":
        with open(pathlib.Path(args.calibration_json).expanduser()) as f:
            fit = json.load(f)["fit_xyz"]
        slopes = np.array([fit[d]["slope_m_per_unit"] for d in "xyz"])
        offsets = np.array([fit[d]["offset_m"] for d in "xyz"])
        with open(pathlib.Path(args.stats_json).expanduser()) as f:
            rot_mean = json.load(f)["mean"][3:6]  # rot dims only; gripper UNPINNED

        def oracle_fn(obs, ctx):
            manip_key, goal_key, lift0 = ctx["manip"], ctx["goal"], ctx["lift0"]
            ee = np.asarray(obs["robot0_eef_pos"])
            lifted = obs[manip_key][2] > lift0 + 0.03
            if lifted:
                target = np.asarray(obs[goal_key]) + np.array([0.0, 0.0, 0.10])
                delta = target - ee
            else:
                delta = np.asarray(obs[manip_key]) - ee
                # deadband: within reach of the manipuland there is no gross
                # motion to command — a descend command would fight the grasp
                # maneuver (instrumented finding: bowl lifted 3 cm, then was
                # dropped under a continuing descend command). Command a
                # gentle lift bias instead until the lift threshold fires.
                if np.linalg.norm(delta) < 0.07:
                    delta = np.array([0.0, 0.0, 0.05])
            delta = np.clip(delta, -args.oracle_clip_m, args.oracle_clip_m)
            # slope-only inversion: the calibration OFFSETS encode the
            # policy's command-zero drift at episode-start states and are
            # state-dependent — applying them near objects mis-translates.
            # Vision keeps its natural drift; the pin commands the residual.
            cmd_xyz = delta / slopes
            return [float(x) for x in cmd_xyz] + [float(x) for x in rot_mean]

        logging.info("Minimal2 oracle active (bowl->goal switch on 3cm lift; "
                     "gripper dim unpinned; 6-dim commands)")
    elif args.oracle == "minimal":
        with open(pathlib.Path(args.calibration_json).expanduser()) as f:
            fit = json.load(f)["fit_xyz"]
        slopes = np.array([fit[d]["slope_m_per_unit"] for d in "xyz"])
        offsets = np.array([fit[d]["offset_m"] for d in "xyz"])
        with open(pathlib.Path(args.stats_json).expanduser()) as f:
            rest_mean = json.load(f)["mean"][3:7]  # rot dims + gripper, neutral

        def oracle_fn(obs, target_key):
            delta = np.clip(np.asarray(obs[target_key]) - np.asarray(obs["robot0_eef_pos"]),
                            -args.oracle_clip_m, args.oracle_clip_m)
            cmd_xyz = (delta - offsets) / slopes
            return [float(x) for x in cmd_xyz] + [float(x) for x in rest_mean]

        logging.info("Minimal oracle active (displacement toward first "
                     "obj_of_interest; calibration-inverted xyz + dataset-mean rest)")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    results = {"task_suite": args.task_suite_name, "seed": args.seed,
               "init_states": args.init_states_dir or "canonical",
               "invariant": invariant, "tasks": []}
    total_episodes, total_successes = 0, 0
    import os as _os
    _only = _os.environ.get("SNMVP_TASK_ID")
    _ids = [int(_only)] if _only is not None else list(range(task_suite.n_tasks))
    for task_id in tqdm.tqdm(_ids):
        task = task_suite.get_task(task_id)

        if args.init_states_dir:
            path = pathlib.Path(args.init_states_dir).expanduser() / f"task{task_id:02d}.npy"
            initial_states = np.load(path)
            logging.info(f"task {task_id}: {len(initial_states)} held-out init states from {path}")
        else:
            initial_states = task_suite.get_task_init_states(task_id)

        n_trials = min(args.num_trials_per_task, len(initial_states))
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        target_key = None
        oracle_ctx = None
        if oracle_fn is not None:
            target_key = f"{env.obj_of_interest[0]}_pos"
            logging.info(f"task {task_id}: oracle objects = {env.obj_of_interest}")
            if args.oracle == "minimal2":
                goal_name = env.obj_of_interest[1] if len(env.obj_of_interest) > 1 \
                    else env.obj_of_interest[0]
                oracle_ctx = {"manip": target_key, "goal": f"{goal_name}_pos",
                              "lift0": None}

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(n_trials)):
            logging.info(f"\nTask: {task_description}")
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            if oracle_ctx is not None:
                oracle_ctx["lift0"] = None  # re-baseline manipuland height

            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        if invariant is not None:
                            element["snmvp_invariant"] = invariant
                        elif oracle_fn is not None:
                            if oracle_ctx is not None:
                                if oracle_ctx["lift0"] is None:
                                    oracle_ctx["lift0"] = float(obs[oracle_ctx["manip"]][2])
                                element["snmvp_invariant"] = oracle_fn(obs, oracle_ctx)
                            else:
                                element["snmvp_invariant"] = oracle_fn(obs, target_key)

                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_t{task_id:02d}_e{episode_idx:02d}_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            logging.info(f"Success: {done}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        env.close()
        results["tasks"].append({
            "task_id": task_id, "language": task_description,
            "episodes": task_episodes, "successes": task_successes,
            "success_rate": task_successes / task_episodes if task_episodes else None,
        })
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    results["total_episodes"] = total_episodes
    results["total_successes"] = total_successes
    results["total_success_rate"] = total_successes / total_episodes if total_episodes else None
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    if args.out_json:
        out = pathlib.Path(args.out_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        logging.info(f"wrote {out}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
