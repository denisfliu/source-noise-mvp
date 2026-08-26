"""Gate b: ZERO-SHOT paraphrase robustness of the VLM task selector.

Hand-authored paraphrases (never seen in training — the selector trained only on
the 4 canonical strings) x held-episode frames from BOTH scenes (rendered obs,
serving domain). The label is the PROMPT's task regardless of scene (selection
follows language; scene-crossed pairs included by construction).
Pre-registered bar: >=90% per task. This is the exact failure mode that killed
frozen text encoders (MiniLM cos(L,R)=0.998) and keyword one-hots.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc

PARAPHRASES = {
    gc.PROMPT_L: [
        "fly through the left gate and hover above the plush toy",
        "take the left-hand gate, then hold position above the stuffed animal",
        "pass through the gate on your left and stop over the toy",
        "go left through the gate and hover over the plushie",
        "head through the leftmost gate and wait above the stuffed animal",
        "use the left gate; finish by hovering over the toy animal",
        "through the left opening, then hover at the stuffed animal",
        "enter via the gate on the left side and hover over the animal"],
    gc.PROMPT_R: [
        "fly through the right gate and hover above the plush toy",
        "take the right-hand gate, then hold position above the stuffed animal",
        "pass through the gate on your right and stop over the toy",
        "go right through the gate and hover over the plushie",
        "head through the rightmost gate and wait above the stuffed animal",
        "use the right gate; finish by hovering over the toy animal",
        "through the right opening, then hover at the stuffed animal",
        "enter via the gate on the right side and hover over the animal"],
    gc.PROMPT_CFL: [
        "go through the middle gate coming from the left and hover over the stuffed animal",
        "take the center gate from the left side, then hold above the toy",
        "approach the central gate from the left and hover over the plush toy",
        "from the left, pass through the middle gate and stop over the animal",
        "center gate, approaching on the left; hover above the stuffed toy",
        "fly through the middle opening from the left side and hover over the plushie",
        "come from the left through the central gate and wait above the toy",
        "via the left approach, go through the center gate and hover over the animal"],
    gc.PROMPT_CFR: [
        "go through the middle gate coming from the right and hover over the stuffed animal",
        "take the center gate from the right side, then hold above the toy",
        "approach the central gate from the right and hover over the plush toy",
        "from the right, pass through the middle gate and stop over the animal",
        "center gate, approaching on the right; hover above the stuffed toy",
        "fly through the middle opening from the right side and hover over the plushie",
        "come from the right through the central gate and wait above the toy",
        "via the right approach, go through the center gate and hover over the animal"],
}
TASKS = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]

if __name__ == "__main__":
    RUN = os.path.expanduser("~/ctxrun")
    rf = np.load(f"{RUN}/rendered_frames.npz")
    fwd224, wrist224 = rf["fwd224"], rf["wrist224"]  # materialize (npz trap)
    row = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rf["si"], rf["fidx"]))}
    
    src = gc.load_eps(with_images=False)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(src)); held = [i for i in idx[160:].tolist()]
    # 24 held frames: spread across scenes/tasks and route phase
    frames = []
    for si in held:
        if len(frames) >= 24:
            break
        n = len(src[si]["state"]) - 1
        for t in (12, 96):
            key = (si, t)
            if key in row:
                frames.append(key)
    frames = frames[:24]
    print("probe frames:", len(frames), flush=True)
    
    sel = np.load(os.path.join(gc.RD, "task_selector.npz"), allow_pickle=True)
    mu, sg, W1, b1, W2, b2 = sel["mu"], sel["sg"], sel["W1"], sel["b1"], sel["W2"], sel["b2"]
    def _gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    def classify(phi):
        return (_gelu(((phi - mu) / sg) @ W1 + b1) @ W2 + b2).argmax(1)
    
    policy = gc.make_policy()
    results = {}
    for task, plist in PARAPHRASES.items():
        k = TASKS.index(task)
        correct = total = 0
        for p in plist:
            obs = [{"observation/image": fwd224[row[f]], "observation/wrist_image": wrist224[row[f]],
                    "observation/state": src[f[0]]["state"][f[1]], "prompt": p} for f in frames]
            phi = gc.feats(policy, obs)
            pred = classify(phi)
            correct += int((pred == k).sum()); total += len(pred)
        results[task] = correct / total
        print("GATE-b %-70s acc %.3f" % (task[:66], results[task]), flush=True)
    bar = all(v >= 0.90 for v in results.values())
    print("GATE-b VERDICT (bar >=0.90/task): %s" % ("PASS" if bar else "FAIL"), flush=True)
    print("PARAPHRASE_DONE", flush=True)
