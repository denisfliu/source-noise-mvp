#!/usr/bin/env python3
"""Mode-averaging ablation server: per inference, sample K flow noises, run the
policy K times, and AVERAGE the action chunks. The mean of flow samples is
E[action | obs] -- exactly what an MSE (BC-regression) policy converges to. If
the pi0 action distribution is genuinely multimodal, this averaged action is an
invalid between-modes trajectory and success collapses, showing flow matching
does real mode-resolution work at scale (i.e. we CANNOT just predict/regress).

Run in the openpi venv (like serve_snmvp_policy.py):
    CUDA_VISIBLE_DEVICES=1 uv run python .../serve_mean.py \
        --config pi0_libero --dir checkpoints/pi0_libero/phase1_A_s42/14999 \
        --port 8000 --mean_k 8
mean_k=1 reproduces the stock single-sample baseline (arm A).
"""
import dataclasses
import logging

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str = "pi0_libero"
    dir: str = ""
    port: int = 8000
    mean_k: int = 8
    noise_seed: int = 0
    default_prompt: str | None = None


class MeanPolicy:
    def __init__(self, policy, k, action_horizon, action_dim, seed):
        self._p = policy
        self._k = k
        self._h = action_horizon
        self._d = action_dim
        self._rng = np.random.default_rng(seed)

    def infer(self, obs: dict) -> dict:
        if self._k <= 1:
            return self._p.infer(obs)
        samples = []
        base = None
        for _ in range(self._k):
            noise = self._rng.normal(size=(self._h, self._d)).astype(np.float32)
            r = self._p.infer(dict(obs), noise=noise)
            samples.append(np.asarray(r["actions"], dtype=np.float64))
            base = r
        S = np.stack(samples)                       # (K, H, Da)
        mean = S.mean(0)
        div = float(np.abs(S - mean).mean())
        scale = float(np.abs(S).mean())
        # MODALITY: 2-means on flattened samples; between/within ratio.
        # ratio~1 => unimodal blob; ratio>>1 => two separated modes (the mean
        # then lands in the empty middle = the mode-averaging failure).
        X = S.reshape(S.shape[0], -1)
        km_ratio, split = self._twomeans(X)
        # also: per-step branching — where along the chunk does spread peak?
        step_mad = np.abs(S - mean).mean(axis=(0, 2))    # (H,)
        peak_step = int(np.argmax(step_mad))
        logging.info(f"SAMPLE_DIVERSITY mad={div:.4f} scale={scale:.4f} "
                     f"ratio={div/(scale+1e-9):.3f} kmeans_bimodality={km_ratio:.2f} "
                     f"split={split} peak_step={peak_step}/{S.shape[1]}")
        base = dict(base)
        base["actions"] = mean.astype(np.float32)
        return base

    @staticmethod
    def _twomeans(X, iters=10):
        """2-means; return (between/within distance ratio, cluster sizes)."""
        n = X.shape[0]
        if n < 4:
            return 1.0, [n, 0]
        # init: two most distant points
        d0 = np.linalg.norm(X - X[0], axis=1)
        a = int(np.argmax(d0))
        da = np.linalg.norm(X - X[a], axis=1)
        b = int(np.argmax(da))
        c = np.stack([X[a], X[b]])
        for _ in range(iters):
            dist = np.stack([np.linalg.norm(X - c[0], axis=1),
                             np.linalg.norm(X - c[1], axis=1)])
            lab = dist.argmin(0)
            for j in (0, 1):
                if (lab == j).any():
                    c[j] = X[lab == j].mean(0)
        between = float(np.linalg.norm(c[0] - c[1]))
        within = float(np.mean([np.linalg.norm(X[i] - c[lab[i]]) for i in range(n)]))
        return between / (within + 1e-9), [int((lab == 0).sum()), int((lab == 1).sum())]

    def __getattr__(self, name):
        return getattr(self._p, name)


def main(args: Args) -> None:
    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, args.dir, default_prompt=args.default_prompt)
    wrapped = MeanPolicy(policy, args.mean_k, cfg.model.action_horizon,
                         cfg.model.action_dim, args.noise_seed)
    logging.info(f"MEAN server: k={args.mean_k}, dir={args.dir}, port={args.port}")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="0.0.0.0", port=args.port,
        metadata=getattr(policy, "metadata", None))
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
