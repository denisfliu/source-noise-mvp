#!/usr/bin/env python3
"""Policy server with a source-noise invariant channel (arm C sim eval).

Identical to openpi's scripts/serve_policy.py checkpoint mode, except: if the
incoming observation dict contains an `snmvp_invariant` key (list of floats in
NORMALIZED action units, length <= real action dims), it is popped and turned
into calibrated source noise via snmvp.openpi_adapter.make_calibrated_noise,
passed to policy.infer(obs, noise=...). Observations without the key behave
exactly as the stock server (fresh Gaussian noise sampled by the model).

Run in the openpi venv:

    cd ~/code/openpi
    UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=1 uv run python \
        ~/code/source-noise-mvp/scripts/serve_snmvp_policy.py \
        --config pi0_libero --dir checkpoints/pi0_libero/<exp>/<step> --port 8000
"""

import dataclasses
import logging

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from snmvp.openpi_adapter import make_calibrated_noise


@dataclasses.dataclass
class Args:
    config: str = "pi0_libero"
    dir: str = ""
    port: int = 8000
    noise_seed: int = 0
    default_prompt: str | None = None
    # D6: path to invariant_prior.pt — when set and the obs carries no
    # explicit snmvp_invariant, the prior computes the pin per call from the
    # obs itself (base image, wrist image, raw state). Arm C becomes a
    # self-contained policy.
    prior: str | None = None
    # arm="B": deliver the invariant through the CONDITIONING state token
    # (z-scored into trailing padding dims after the input transforms, plain
    # noise) instead of the source noise — mirrors the arm B training patch.
    arm: str = "C"
    cond_stats: str | None = None  # invariant_stats.json (required for arm B)


class SnmvpNoisePolicy:
    """Duck-typed BasePolicy wrapper adding the snmvp_invariant channel."""

    def __init__(self, policy, action_horizon, action_dim, noise_seed,
                 prior_path=None, arm="C", cond_stats_path=None):
        self._policy = policy
        self._h = action_horizon
        self._d = action_dim
        self._rng = np.random.default_rng(noise_seed)
        self._arm = arm
        self._prior = None
        if arm == "B":
            from snmvp.conditioning import inject_invariant_state, load_invariant_stats
            stats = load_invariant_stats(cond_stats_path)
            orig_tf = policy._input_transform

            def wrapped(data):
                inv = data.pop("__snmvp_inv", None)
                out = orig_tf(data)
                if inv is not None:
                    inject_invariant_state(
                        out["state"], np.asarray(inv, dtype=np.float32), *stats)
                return out

            policy._input_transform = wrapped
            logging.info("Arm B serving: invariant delivered via state token")
        if prior_path:
            import torch
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).parent))
            from train_invariant_prior import InvariantPrior
            ckpt = torch.load(pathlib.Path(prior_path).expanduser(),
                              map_location="cpu", weights_only=False)
            net = InvariantPrior()
            net.load_state_dict(ckpt["state_dict"])
            self._prior = net.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            self._prior_stats = (np.asarray(ckpt["inv_mean"]), np.asarray(ckpt["inv_std"]))
            self._torch = torch
            logging.info(f"Invariant prior loaded from {prior_path}")

    def _prior_invariant(self, obs):
        t = self._torch
        dev = next(self._prior.parameters()).device
        def img(key):
            x = np.asarray(obs[key], dtype=np.float32) / 255.0
            return t.from_numpy(x).permute(2, 0, 1)[None].to(dev)
        state = t.from_numpy(
            np.asarray(obs["observation/state"], dtype=np.float32)[:8])[None].to(dev)
        with t.no_grad():
            z = self._prior(img("observation/image"),
                            img("observation/wrist_image"), state)[0].cpu().numpy()
        mean, std = self._prior_stats
        return z * std + mean  # back to raw normalized-invariant units

    def infer(self, obs: dict) -> dict:
        inv = obs.pop("snmvp_invariant", None)
        if inv is None and self._prior is not None:
            inv = self._prior_invariant(obs)
        if inv is None:
            return self._policy.infer(obs)
        if self._arm == "B":
            obs["__snmvp_inv"] = np.asarray(inv, dtype=np.float32)
            return self._policy.infer(obs)
        noise = make_calibrated_noise(
            np.asarray(inv, dtype=np.float32), self._h, self._d, self._rng
        )
        return self._policy.infer(obs, noise=noise)

    def __getattr__(self, name):
        return getattr(self._policy, name)


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        train_config, args.dir, default_prompt=args.default_prompt
    )
    wrapped = SnmvpNoisePolicy(
        policy, train_config.model.action_horizon, train_config.model.action_dim,
        args.noise_seed, prior_path=args.prior, arm=args.arm,
        cond_stats_path=args.cond_stats,
    )
    logging.info(f"Serving {args.dir} with snmvp invariant channel on port {args.port}")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="0.0.0.0", port=args.port,
        metadata=getattr(policy, "metadata", None),
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
