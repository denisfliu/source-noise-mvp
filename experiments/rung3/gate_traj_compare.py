"""Is the domain gap in the TRAJECTORY or only in the ACTION representation? c is built from the action
chunk, so if sim and real fly the SAME path (states match over progress) but with different control style
(smooth MPC vs jerky teleop), c picks up style, not task -- and a trajectory-derived c would be domain-
invariant. Tests: (1) mean STATE (pose x,y,z,yaw) vs normalized progress, sim vs real per gate -> do the
paths overlap? (2) relationship action[t] vs state-delta state[t+1]-state[t] per domain -> is action the
pose derivative, and at the same scale? (3) recompute a c from FUTURE STATE DISPLACEMENT (pose[t+k]-pose[t])
instead of actions and check its sim-vs-real gap vs left-right separation."""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = "go through the gate on the right and hover over the stuffed animal"
DIMS = [0, 1, 2, 3]  # x, y, z, yaw (state dims 4-6 are dead)


def load(raw, lang):
    meta = json.load(open(os.path.join(raw, "meta.json")))
    eps = []
    for k in sorted(meta):
        if meta[k]["lang"] != lang:
            continue
        d = np.load(os.path.join(raw, k + ".npz"))
        eps.append((d["state"].astype(np.float32), d["action"].astype(np.float32)))
    return eps


def prog_mean(eps, sel):  # mean of sel(state,action) over progress deciles
    bins = [[] for _ in range(10)]
    for states, acts in eps:
        T = len(acts)
        for t in range(T):
            v = sel(states, acts, t, T)
            if v is not None:
                bins[min(9, int(10 * t / max(T - 1, 1)))].append(v)
    return np.array([np.mean(b, 0) if b else np.full(len(DIMS), np.nan) for b in bins])


def main():
    RR, SS = os.path.join(RD, "data_gate_real"), os.path.join(RD, "data_gate_synth")
    np.set_printoptions(precision=3, suppress=True)

    # (1) mean POSE trajectory over progress, sim vs real, per gate
    print("=== (1) mean POSE (x,y,z,yaw) over progress deciles: sim vs real ===")
    pose = lambda s, a, t, T: s[t, DIMS]
    for lang, tag in ((LEFT, "LEFT"), (RIGHT, "RIGHT")):
        r = prog_mean(load(RR, lang), pose); s = prog_mean(load(SS, lang), pose)
        gap = np.nanmean(np.linalg.norm(r - s, axis=1))
        print(f"[{tag}] mean |pose_real - pose_sim| over deciles = {gap:.3f}")
        print(f"   real decile0/5/9 pose: {r[0]} {r[5]} {r[9]}")
        print(f"   sim  decile0/5/9 pose: {s[0]} {s[5]} {s[9]}")
    # left-right pose separation (real) for reference
    rl = prog_mean(load(RR, LEFT), pose); rr = prog_mean(load(RR, RIGHT), pose)
    print(f"reference: real LEFT-vs-RIGHT pose separation over deciles = {np.nanmean(np.linalg.norm(rl-rr,axis=1)):.3f}")

    # (2) action vs state-delta relationship per domain (scale + correlation), dims 0-3
    print("\n=== (2) action[t] vs pose-delta (state[t+1]-state[t]) per domain ===")
    for raw, tag in ((RR, "real"), (SS, "sim")):
        A, D = [], []
        for states, acts in load(raw, LEFT) + load(raw, RIGHT):
            if len(states) < 2:
                continue
            A.append(acts[:-1][:, DIMS]); D.append(np.diff(states[:, DIMS], axis=0))
        A, D = np.concatenate(A), np.concatenate(D)
        for i, dn in enumerate(["x", "y", "z", "yaw"]):
            c = np.corrcoef(A[:, i], D[:, i])[0, 1]
            sc = A[:, i].std() / (D[:, i].std() + 1e-9)
            print(f"  [{tag}] {dn:3s}: corr(action, pose-delta)={c:+.3f}  std(action)/std(pose-delta)={sc:.3f}")

    # (3) c from FUTURE POSE DISPLACEMENT instead of actions: domain-invariant?
    print("\n=== (3) trajectory-derived signal: future pose displacement pose[t+K]-pose[t] ===")
    K = 20
    def disp(s, a, t, T):
        j = min(t + K, T - 1)
        return s[j, DIMS] - s[t, DIMS]
    for lang, tag in ((LEFT, "LEFT"), (RIGHT, "RIGHT")):
        r = prog_mean(load(RR, lang), disp); s = prog_mean(load(SS, lang), disp)
        print(f"[{tag}] |disp_real - disp_sim| over deciles = {np.nanmean(np.linalg.norm(r-s,axis=1)):.3f}")
    dl_r = prog_mean(load(RR, LEFT), disp); dr_r = prog_mean(load(RR, RIGHT), disp)
    print(f"reference: real LEFT-vs-RIGHT displacement separation = {np.nanmean(np.linalg.norm(dl_r-dr_r,axis=1)):.3f}")
    print("TRAJ_DONE")


if __name__ == "__main__":
    main()
