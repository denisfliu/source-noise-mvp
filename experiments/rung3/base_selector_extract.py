"""Extract BASE-pi0 (pre-gate-finetune) lang_pool features for the selector rows and
BOTH paraphrase sets. The paraphrase ladder's conclusion: gate fine-tuning collapsed
paraphrase semantics; the base tower should retain them. GPU."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
from train_paraphrases import TRAIN_PARAPHRASES
from gate_b_paraphrase import PARAPHRASES as EVAL_PARAPHRASES

RUN = os.path.expanduser("~/ctxrun")
BASE = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi0_base")
TASKS = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]
rf = np.load(f"{RUN}/rendered_frames.npz")
fwd224, wrist224 = rf["fwd224"], rf["wrist224"]
row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rf["si"], rf["fidx"]))}
src = gc.load_eps(with_images=False)
rng = np.random.default_rng(0); idx = rng.permutation(len(src))
def frames_from(ep_ids, per_scene):
    fr = {"left": [], "right": []}
    for si in ep_ids:
        scene = "right" if src[si]["lang"] == gc.PROMPT_R else "left"
        if len(fr[scene]) >= per_scene: continue
        for t in (12, 96):
            if (si, t) in row and len(fr[scene]) < per_scene:
                fr[scene].append((si, t))
    return fr["left"] + fr["right"]
train_frames = frames_from([int(i) for i in idx[:160]], 15)
eval_frames = frames_from([int(i) for i in idx[160:]], 12)
policy = gc.make_policy(BASE, config="pi0_libero")  # non-LoRA config matches the base checkpoint structure
def dump(frames, prompts_by_task, tag):
    X, y = [], []
    for task, plist in prompts_by_task.items():
        k = TASKS.index(task)
        for p in plist:
            obs = [{"observation/image": fwd224[row[f]], "observation/wrist_image": wrist224[row[f]],
                    "observation/state": src[f[0]]["state"][f[1]], "prompt": p} for f in frames]
            for i in range(0, len(obs), gc.BS):
                X.append(gc.lang_pool(policy, obs[i:i + gc.BS]))
            y += [k] * len(obs)
        print(tag, task[:40], flush=True)
    np.save(f"{RUN}/Xbase_{tag}.npy", np.concatenate(X, 0))
    np.save(f"{RUN}/ybase_{tag}.npy", np.array(y))
train_prompts = {t: [t] + TRAIN_PARAPHRASES[t] for t in TASKS}
dump(train_frames, train_prompts, "train")
dump(eval_frames, EVAL_PARAPHRASES, "eval")
print("BASE_EXTRACT_DONE", flush=True)
