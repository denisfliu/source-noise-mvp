"""Zero-training retrieval task selection: a paraphrase selects the canonical prompt
whose lang-pool feature (same frame) it is nearest to (cosine). No trained head —
tests the representation directly. Eval on the untouched gate-b paraphrase set."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
from gate_b_paraphrase import PARAPHRASES, TASKS

RUN = os.path.expanduser("~/ctxrun")
rf = np.load(f"{RUN}/rendered_frames.npz")
fwd224, wrist224 = rf["fwd224"], rf["wrist224"]
row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rf["si"], rf["fidx"]))}
src = gc.load_eps(with_images=False)
rng = np.random.default_rng(0)
idx = rng.permutation(len(src))
frames = []
for si in [int(i) for i in idx[160:]]:
    for t in (12, 96):
        if (si, t) in row and len(frames) < 24:
            frames.append((si, t))
policy = gc.make_policy()
def lp(prompt):
    obs = [{"observation/image": fwd224[row[f]], "observation/wrist_image": wrist224[row[f]],
            "observation/state": src[f[0]]["state"][f[1]], "prompt": prompt} for f in frames]
    out = []
    for i in range(0, len(obs), gc.BS):
        out.append(gc.lang_pool(policy, obs[i:i + gc.BS]))
    return np.concatenate(out, 0)   # (n_frames, 2048)
anchors = np.stack([lp(t) for t in TASKS])          # (4, F, D)
an = anchors / np.linalg.norm(anchors, axis=-1, keepdims=True)
ok_all = True
for task, plist in PARAPHRASES.items():
    k = TASKS.index(task); correct = total = 0
    for p in plist:
        q = lp(p); qn = q / np.linalg.norm(q, axis=-1, keepdims=True)
        sims = np.einsum("kfd,fd->kf", an, qn)      # cos vs each canonical, per frame
        pred = sims.argmax(0)
        correct += int((pred == k).sum()); total += len(pred)
    acc = correct / total; ok_all &= acc >= 0.90
    print("RETRIEVAL %-70s acc %.3f" % (task[:66], acc), flush=True)
print("RETRIEVAL VERDICT (bar >=0.90/task): %s" % ("PASS" if ok_all else "FAIL"), flush=True)
print("RETRIEVAL_DONE", flush=True)
