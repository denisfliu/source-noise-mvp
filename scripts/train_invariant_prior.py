#!/usr/bin/env python3
"""D6: learned invariant prior p(invariant | base image, wrist image, state).

Trains a small CNN+MLP to predict the NORMALIZED next-H-step chunk invariant
(z-score action space, the exact quantity the arm C pin carries) from raw
observations, matching the serving client's element spec exactly:
224x224 RGB images + raw 8-dim proprio state. Trained directly on the
LeRobot dataset — no openpi transform stack, hence no train/serve mismatch.

This is the toy_frame Step-3 prior at LIBERO scale (the configuration that
passed G3 there) and the miniature of Phase 2's prior head.

Run in the openpi venv:
    cd ~/code/openpi
    UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=1 uv run python \
        ~/code/source-noise-mvp/scripts/train_invariant_prior.py \
        --norm-stats checkpoints/pi0_libero/phase1_C_s42/14999/assets/physical-intelligence/libero/norm_stats.json \
        --out ~/code/source-noise-mvp/experiments/phase1/invariant_prior.pt
"""

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H_CHUNK = 50
REAL_DIMS = 7
IMG = 224


class InvariantPrior(nn.Module):
    def __init__(self):
        super().__init__()

        def tower():
            return nn.Sequential(
                nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())

        self.base = tower()
        self.wrist = tower()
        self.state = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + 128 + 64, 256), nn.ReLU(),
                                  nn.Linear(256, 128), nn.ReLU(),
                                  nn.Linear(128, REAL_DIMS))

    def forward(self, base, wrist, state):
        """base/wrist: (B,3,224,224) float 0..1; state: (B,8) raw."""
        z = torch.cat([self.base(base), self.wrist(wrist), self.state(state)], dim=1)
        return self.head(z)  # standardized invariant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--inv-stats", default=str(pathlib.Path(
        "~/code/source-noise-mvp/experiments/phase1/invariant_stats.json").expanduser()))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ns = json.load(open(pathlib.Path(args.norm_stats).expanduser()))["norm_stats"]["actions"]
    a_mean = np.asarray(ns["mean"][:REAL_DIMS], dtype=np.float32)
    a_std = np.asarray(ns["std"][:REAL_DIMS], dtype=np.float32)
    ivs = json.load(open(pathlib.Path(args.inv_stats).expanduser()))
    i_mean = np.asarray(ivs["mean"], dtype=np.float32)
    i_std = np.asarray(ivs["std"], dtype=np.float32)

    import lerobot.common.datasets.lerobot_dataset as lds
    meta = lds.LeRobotDatasetMetadata("physical-intelligence/libero")
    fps = meta.fps
    ds = lds.LeRobotDataset(
        "physical-intelligence/libero",
        delta_timestamps={"actions": [t / fps for t in range(H_CHUNK)]})

    n_ep = meta.total_episodes
    ep_perm = rng.permutation(n_ep)
    val_eps = set(ep_perm[: int(n_ep * args.val_frac)].tolist())
    frame_ep = np.zeros(len(ds), dtype=np.int64)
    for ep in range(n_ep):
        frame_ep[ds.episode_data_index["from"][ep]:ds.episode_data_index["to"][ep]] = ep
    train_idx = np.where(~np.isin(frame_ep, list(val_eps)))[0]
    val_idx = np.where(np.isin(frame_ep, list(val_eps)))[0]
    print(f"frames: train {len(train_idx)}, val {len(val_idx)} "
          f"({len(val_eps)}/{n_ep} episodes held out)", flush=True)

    class FrameDataset(torch.utils.data.Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, k):
            item = ds[int(self.indices[k])]
            base = F.interpolate(item["image"][None], size=IMG, mode="bilinear",
                                 align_corners=False)[0]
            wrist = F.interpolate(item["wrist_image"][None], size=IMG, mode="bilinear",
                                  align_corners=False)[0]
            acts = item["actions"].numpy()[:, :REAL_DIMS].copy()
            # replicate the pipeline's extra delta transform (mask 6,-1):
            # dims 0-5 become deltas w.r.t. the CURRENT state; gripper absolute
            acts[:, :6] -= item["state"].numpy()[:6]
            inv = ((acts - a_mean) / (a_std + 1e-6)).sum(0)
            tgt = torch.from_numpy((inv - i_mean) / i_std).float()
            return base, wrist, item["state"][:8].float(), tgt

    def make_loader(indices, shuffle):
        return torch.utils.data.DataLoader(
            FrameDataset(indices), batch_size=args.batch, shuffle=shuffle,
            num_workers=8, drop_last=True, persistent_workers=True,
            prefetch_factor=4)

    def to_dev(batch):
        return tuple(t.to(device) for t in batch)

    model = InvariantPrior().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx[:1024], shuffle=False)
    step = 0
    while step < args.steps:
        for batch in train_loader:
            b, w, st, y = to_dev(batch)
            pred = model(b, w, st)
            loss = F.huber_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % 250 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    errs = []
                    for vb_ in val_loader:
                        vb, vw, vs, vy = to_dev(vb_)
                        vpred = model(vb, vw, vs)
                        errs.append(((vpred - vy).abs()
                                     * torch.from_numpy(i_std).to(device)).mean(0))
                        if len(errs) >= 3:
                            break
                    err = torch.stack(errs).mean(0)
                print(f"step {step}: train_loss {loss.item():.4f} "
                      f"val_MAE_rawunits {np.round(err.cpu().numpy(), 1).tolist()} "
                      f"(dataset std {np.round(i_std, 1).tolist()})", flush=True)
            step += 1
            if step >= args.steps:
                break

    out = pathlib.Path(args.out).expanduser()
    torch.save({"state_dict": model.state_dict(),
                "action_mean": a_mean, "action_std": a_std,
                "inv_mean": i_mean, "inv_std": i_std,
                "img_size": IMG, "real_dims": REAL_DIMS, "h_chunk": H_CHUNK,
                "val_episodes": sorted(val_eps)}, out)
    print("saved", out)
    print("PRIOR_TRAIN_FINAL=ok")


if __name__ == "__main__":
    main()
