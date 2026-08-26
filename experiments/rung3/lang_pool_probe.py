"""Decisive probe: does LANGUAGE-position pooling fix paraphrase selection?

Train the same 4-way head on lang_pool features of (canonical prompts + the 12/task
training paraphrases) over train-episode rendered frames; evaluate on the UNTOUCHED
gate-b eval paraphrase set (held frames). Single script, GPU; ~3.9k forwards.
Bar unchanged: >=0.90 per task on unseen phrasings.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
from train_paraphrases import TRAIN_PARAPHRASES
from gate_b_paraphrase import PARAPHRASES as EVAL_PARAPHRASES

RUN = os.path.expanduser("~/ctxrun")
TASKS = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]
torch.manual_seed(0); np.random.seed(0)

rf = np.load(f"{RUN}/rendered_frames.npz")
fwd224, wrist224 = rf["fwd224"], rf["wrist224"]  # materialize (npz trap)
row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rf["si"], rf["fidx"]))}
src = gc.load_eps(with_images=False)
rng = np.random.default_rng(0)
idx = rng.permutation(len(src))
def frames_from(ep_ids, per_scene):
    fr = {"left": [], "right": []}
    for si in ep_ids:
        scene = "right" if src[si]["lang"] == gc.PROMPT_R else "left"
        if len(fr[scene]) >= per_scene:
            continue
        for t in (12, 96):
            if (si, t) in row and len(fr[scene]) < per_scene:
                fr[scene].append((si, t))
    return fr["left"] + fr["right"]
train_frames = frames_from([int(i) for i in idx[:160]], 15)
eval_frames = frames_from([int(i) for i in idx[160:]], 12)
print("train frames %d, eval frames %d" % (len(train_frames), len(eval_frames)), flush=True)

policy = gc.make_policy()
def feats_for(frames, prompts_by_task):
    X, y = [], []
    for task, plist in prompts_by_task.items():
        k = TASKS.index(task)
        for p in plist:
            obs = [{"observation/image": fwd224[row[f]], "observation/wrist_image": wrist224[row[f]],
                    "observation/state": src[f[0]]["state"][f[1]], "prompt": p} for f in frames]
            for i in range(0, len(obs), gc.BS):
                X.append(gc.lang_pool(policy, obs[i:i + gc.BS]))
            y += [k] * len(obs)
    return np.concatenate(X, 0), np.array(y)

train_prompts = {t: [t] + TRAIN_PARAPHRASES[t] for t in TASKS}
Xtr, ytr = feats_for(train_frames, train_prompts)
print("train rows:", len(Xtr), flush=True)
mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-6
net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.GELU(approximate="tanh"), nn.Linear(128, 4))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
Xn = torch.tensor((Xtr - mu) / sg, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
for ep in range(60):
    perm = np.random.permutation(len(Xn))
    for i in range(0, len(perm), 256):
        b = perm[i:i + 256]; opt.zero_grad()
        nn.functional.cross_entropy(net(Xn[b]), yt[b]).backward(); opt.step()
net.eval()

Xev, yev = feats_for(eval_frames, EVAL_PARAPHRASES)
with torch.no_grad():
    pred = net(torch.tensor((Xev - mu) / sg, dtype=torch.float32)).argmax(1).numpy()
ok = True
for k, t in enumerate(TASKS):
    m = yev == k
    acc = float((pred[m] == k).mean())
    ok &= acc >= 0.90
    print("LANGPOOL GATE-b %-70s acc %.3f" % (t[:66], acc), flush=True)
print("LANGPOOL VERDICT (bar >=0.90/task): %s" % ("PASS" if ok else "FAIL"), flush=True)
L = [m for m in net if isinstance(m, nn.Linear)]
np.savez(os.path.join(gc.RD, "task_selector_lang.npz"), mu=mu.astype(np.float32), sg=sg.astype(np.float32),
         W1=L[0].weight.detach().numpy().T, b1=L[0].bias.detach().numpy(),
         W2=L[1].weight.detach().numpy().T, b2=L[1].bias.detach().numpy(), tasks=np.array(TASKS))
print("LANGPOOL_PROBE_DONE", flush=True)
