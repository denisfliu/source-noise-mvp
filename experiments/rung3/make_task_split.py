"""Build a source / held-out TASK split for the few-shot study. Hold out 8 of the 40
LIBERO tasks as few-shot targets; the source model trains only on the other 32.
Writes source_episodes.json (episode indices for source training) and
heldout_tasks.json ({task_index: [episode indices]}) for few-shot adaptation + eval."""
import json, os
import numpy as np

META = os.path.expanduser("~/.cache/huggingface/lerobot/physical-intelligence/libero/meta")
OUT = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
N_HELDOUT = 8

tasks = {}
for line in open(os.path.join(META, "tasks.jsonl")):
    d = json.loads(line); tasks[d["task"]] = d["task_index"]

ep_task = {}
for line in open(os.path.join(META, "episodes.jsonl")):
    d = json.loads(line); ep_task[d["episode_index"]] = tasks[d["tasks"][0]]

all_tasks = sorted(set(ep_task.values()))
rng = np.random.default_rng(0)
held = set(int(t) for t in rng.choice(all_tasks, N_HELDOUT, replace=False))
source_eps = sorted([e for e, t in ep_task.items() if t not in held])
held_map = {}
for e, t in ep_task.items():
    if t in held:
        held_map.setdefault(int(t), []).append(e)
held_map = {t: sorted(v) for t, v in held_map.items()}

json.dump(source_eps, open(os.path.join(OUT, "source_episodes.json"), "w"))
json.dump(held_map, open(os.path.join(OUT, "heldout_tasks.json"), "w"))
print(f"total tasks {len(all_tasks)}; held-out {sorted(held)}; "
      f"source episodes {len(source_eps)}; held-out episodes {sum(len(v) for v in held_map.values())}")
print("held-out episodes per task:", {t: len(v) for t, v in held_map.items()})
