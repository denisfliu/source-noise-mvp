"""Command source for the LIBERO pin arms: c = MLP([model_state, task_onehot40]).

The analogue of the drone's clockless state prior, trained ONCE on the full 40-task LIBERO
set and reused at every ladder rung, so the ladder varies the flow's data and not the
command quality. Task encoding is a 40-way one-hot — a SCAFFOLD, as on the drone; the
grounded language-embedding version is the successor once the ladder itself is measured.

Targets: c = U^T normalize(action chunk), U = pin_U_rrr_k5_shared.npy (RRR basis fit on
LIBERO). Reports held-out c-R2 overall and per phase (early/transit/tail).

    python experiments/rung3/libero_prior.py            # writes libero_prior.pt
"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pin_basis

RD = os.path.dirname(os.path.abspath(__file__))
UPATH = os.environ.get("UPATH", f"{RD}/pin_U_rrr_k5_shared.npy")
SRC = os.path.expanduser("~/.cache/huggingface/lerobot/physical-intelligence/libero")
H, AD = 50, 32
STRIDE = 4
TAILW = float(os.environ.get("TAILW", "4"))     # goal-phase weighting (2026-08-08 finding)


def main():
    import openpi.policies.policy_config as PC
    import openpi.shared.normalize as NZ
    import openpi.training.config as C
    from openpi import transforms as T

    from openpi.transforms import NormStats
    cfg = C.get_config("pi0_libero_shared")
    ns = NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_libero_shared/physical-intelligence/libero"))

    def pads(nsd, dim):  # chunks are padded to the model action dim; stats must match
        out = {}
        for k, s in nsd.items():
            n = np.asarray(s.mean).shape[-1]
            if n >= dim:
                out[k] = s; continue
            p = dim - n
            ext = lambda a, f: None if a is None else np.concatenate(
                [np.asarray(a, np.float32), np.full(p, f, np.float32)])
            out[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1),
                               q01=ext(s.q01, 0), q99=ext(s.q99, 1))
        return out
    nsp = pads(ns, cfg.model.action_dim)
    U = np.load(UPATH).astype(np.float32)
    K = U.shape[1]   # pin dimension follows the basis (K=5 under-provisions LIBERO's 7 active dims)
    nrm = T.Normalize(nsp, use_quantiles=False)
    tasks = [json.loads(l) for l in open(f"{SRC}/meta/tasks.jsonl")]
    task_text = {t["task_index"]: t["task"] for t in tasks}
    ntask = len(tasks)
    print(f"{ntask} tasks; basis {U.shape}", flush=True)

    # the policy is loaded only to run _input_transform (model state); images are dummies
    policy = PC.create_trained_policy(cfg, os.path.expanduser(
        "~/code/openpi/checkpoints/pi0_libero_shared/hh_pin_t11/1999"), norm_stats=ns)
    _D = np.zeros((224, 224, 3), np.uint8)

    def c_of(chunk):
        L = len(chunk)
        ch = np.zeros((H, AD), np.float32)
        m = min(L, H)
        ch[:m, :chunk.shape[1]] = chunk[:m]
        if m < H:
            ch[m:, :chunk.shape[1]] = chunk[m - 1]
        return (nrm({"actions": ch})["actions"].reshape(-1)) @ U

    files = sorted(glob.glob(f"{SRC}/data/chunk-*/episode_*.parquet"))
    rng = np.random.default_rng(0)
    held = set(rng.permutation(len(files))[int(0.85 * len(files)):].tolist())
    X, Y, W, FR, TR = [], [], [], [], []
    for ei, f in enumerate(files):
        tb = pq.read_table(f, columns=["state", "actions", "task_index"])
        st = np.asarray(tb.column("state").to_pylist(), np.float32)
        ac = np.asarray(tb.column("actions").to_pylist(), np.float32)
        ti = int(tb.column("task_index")[0].as_py())
        oh = np.zeros(ntask, np.float32); oh[ti] = 1.0
        Tn = len(st)
        for t in range(0, max(Tn - H, 1), STRIDE):
            ms = np.asarray(policy._input_transform(
                {"observation/image": _D, "observation/wrist_image": _D,
                 "observation/state": st[t], "prompt": task_text[ti]})["state"]).reshape(-1)
            frac = t / max(Tn - 1, 1)
            X.append(np.concatenate([ms, oh]).astype(np.float32))
            Y.append(c_of(ac[t:t + H]).astype(np.float32))
            W.append(np.float32(TAILW if frac >= 0.75 else 1.0))
            FR.append(np.float32(frac)); TR.append(ei not in held)
        if ei % 200 == 0:
            print(f"  ep {ei}/{len(files)} rows {len(X)}", flush=True)
    X = np.array(X, np.float32); Y = np.array(Y, np.float32)
    W = np.array(W, np.float32); FR = np.array(FR, np.float32); TR = np.array(TR, bool)
    print(f"rows {len(X)} (train {int(TR.sum())}), in_dim {X.shape[1]}", flush=True)

    mu, sd = X[TR].mean(0), X[TR].std(0) + 1e-6
    Xt = torch.tensor((X[TR] - mu) / sd); Yt = torch.tensor(Y[TR]); Wt = torch.tensor(W[TR])
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                        nn.Linear(256, K))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    nstate = X.shape[1] - ntask
    for ep in range(120):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            j = p[i:i + 1024]
            xb = Xt[j].clone()
            xb[:, :nstate] += 0.1 * torch.randn_like(xb[:, :nstate])
            opt.zero_grad()
            (Wt[j] * ((net(xb) - Yt[j]) ** 2).mean(1)).sum().div(Wt[j].sum()).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor((X - mu) / sd)).numpy()

    def r2(m):
        return 1 - ((Y[m] - pred[m]) ** 2).sum() / (((Y[m] - Y[m].mean(0)) ** 2).sum() + 1e-9)
    print(f"held c-R2 {r2(~TR):+.4f} (train {r2(TR):+.4f})", flush=True)
    for name, lo, hi in (("early", 0.0, 0.5), ("transit", 0.5, 0.75), ("tail", 0.75, 1.01)):
        m = (~TR) & (FR >= lo) & (FR < hi)
        print(f"  phase {name:8s} held c-R2 {r2(m):+.4f} n={int(m.sum())}", flush=True)
    torch.save({"kind": "libero_prior", "in_dim": X.shape[1], "hidden": [256, 256], "K": K,
                "state_dim": nstate, "ntask": ntask,
                "tasks": [task_text[i] for i in range(ntask)],
                "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                "state_dict": net.state_dict(),
                **pin_basis.stamp(UPATH)}, os.environ.get("OUT", f"{RD}/libero_prior.pt"))
    print("LIBERO_PRIOR_DONE", flush=True)


if __name__ == "__main__":
    main()
