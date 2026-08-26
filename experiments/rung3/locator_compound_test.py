"""Gate-locator generalization: TWO-GATE scene frames (never in locator training).
The same frame, queried with different prompts, must localize DIFFERENT gates:
'left' prompt -> left-gate anchor; 'center' prompts -> the duplicated gate's anchor.
Renders 24 corridor poses (tv-env renders written by render stage below are read here),
extracts fused features (openpi env), reports per-prompt localization error. GPU."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc

RUN = os.path.expanduser("~/ctxrun")
rf = np.load(f"{RUN}/compound_frames.npz")
fwd224, wrist224, states = rf["fwd224"], rf["wrist224"], rf["states"]
sel = np.load(os.path.join(gc.RD, "gate_locator.npz"))
mu, sg, W1, b1, W2, b2 = sel["mu"], sel["sg"], sel["W1"], sel["b1"], sel["W2"], sel["b2"]
def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
def locate(phi):
    return _gelu(((phi - mu) / sg) @ W1 + b1) @ W2 + b2
ANCHORS = {gc.PROMPT_L: np.array([0.861, 0.694, 1.075]),
           gc.PROMPT_CFL: np.array([2.756, -0.3275, 1.0])}
policy = gc.make_policy()
for prompt, anchor in ANCHORS.items():
    obs = [{"observation/image": fwd224[i], "observation/wrist_image": wrist224[i],
            "observation/state": states[i], "prompt": prompt} for i in range(len(states))]
    phis = []
    for i in range(0, len(obs), gc.BS):
        phis.append(gc.ctx_pool(policy, obs[i:i + gc.BS]))
    P = locate(np.concatenate(phis, 0))
    err = np.linalg.norm(P - anchor, axis=1)
    print("COMPOUND-LOCATOR %-14s err %.3f±%.3f m  (mean pred %s)" % (
        "LEFT" if prompt == gc.PROMPT_L else "CENTER", err.mean(), err.std(), np.round(P.mean(0), 2)), flush=True)
print("COMPOUND_LOCATOR_DONE", flush=True)
