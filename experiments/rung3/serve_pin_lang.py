"""Policy server with the source-noise pin driven by a (state[, +language])->c prior, for closed-
loop LIBERO eval. Generalizes serve_pca_pin: --prior is an .npz with key W of shape (feat+1, K)
(bias-augmented least-squares weights). If --langfeat is given (comma-separated onehot), it is
appended to the normalized state before applying the prior; otherwise the prior is state-only.
No --prior => unpinned (baseline). Builds noise via pca_pin.build_pca_noise and calls
policy.infer(obs, noise=...)."""
import dataclasses
import logging
import os
import sys

import numpy as np
import tyro

sys.path.insert(0, os.path.expanduser("~/code/source-noise-mvp/experiments/rung3"))
import pca_pin as PP
import openpi.training.config as _config
import openpi.policies.policy_config as _policy_config
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Args:
    dir: str
    U: str
    prior: str | None = None
    langfeat: str = ""            # comma-separated floats appended to state (empty = state-only)
    config: str = "pi0_libero_shared"
    default_prompt: str | None = None
    port: int = 8000
    noise_seed: int = 0


class PinLangPolicy:
    def __init__(self, policy, U, prior_path, langfeat, ah, ad, seed):
        self._p = policy
        self.U = PP.load_U(U)
        self.H, self.ad = ah, ad
        self.rng = np.random.default_rng(seed)
        self.lf = np.array([float(x) for x in langfeat.split(",")], np.float32) if langfeat else None
        if prior_path:
            self.W = np.load(prior_path)["W"].astype(np.float32)
            logging.info(f"pin ON: W{self.W.shape} langfeat={None if self.lf is None else self.lf.shape}")
        else:
            self.W = None
            logging.info("pin OFF (unpinned baseline)")

    def _state(self, obs):
        return np.asarray(self._p._input_transform(dict(obs))["state"]).reshape(-1)

    def infer(self, obs: dict) -> dict:
        if self.W is None:
            return self._p.infer(obs)
        st = self._state(obs)
        feat = st if self.lf is None else np.concatenate([st, self.lf])
        feat = np.concatenate([feat, [1.0]]).astype(np.float32)
        c = (feat @ self.W).astype(np.float32)
        noise = PP.build_pca_noise(c, self.U, self.rng, self.H, self.ad).astype(np.float32)
        return self._p.infer(obs, noise=noise)

    def __getattr__(self, name):
        return getattr(self._p, name)


def main(a: Args) -> None:
    tc = _config.get_config(a.config)
    pol = _policy_config.create_trained_policy(tc, a.dir, default_prompt=a.default_prompt)
    wrapped = PinLangPolicy(pol, a.U, a.prior, a.langfeat,
                            tc.model.action_horizon, tc.model.action_dim, a.noise_seed)
    logging.info(f"Serving {a.dir} on port {a.port}")
    websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped, host="0.0.0.0", port=a.port,
        metadata=getattr(pol, "metadata", None)).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
