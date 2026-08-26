"""Deployable pinned policy — the inference half of the method, with no sim, no renderer
and no experiment-script dependencies.

A deployment is a directory (see scripts/package_policy.py) containing:
    manifest.json     artifact list + sha256 + config name + action space + provenance
    params/           the openpi checkpoint directory
    norm_stats/       normalization assets for the training dataset
    pin_U.npy         (D, K) orthonormal basis, D = action_horizon * action_dim
    prior.npz         command source weights (numpy; no torch at inference)

Usage:
    from snmvp.deploy import PinnedPolicy
    pol = PinnedPolicy.from_bundle("/path/to/bundle")
    chunk = pol.act({"observation/image": img, "observation/wrist_image": wrist,
                     "observation/state": state, "prompt": "..."})

Steering: pol.nudge("z", 0.30) adds metres of net chunk displacement to every command until
cleared (see snmvp/steering semantics in experiments/rung3/steer_c.py).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

__all__ = ["PinnedPolicy", "NumpyMLP", "sha256_of"]


def sha256_of(path: str, _chunk: int = 1 << 20) -> str:
    """Hash a file, or a directory's files in sorted order (checkpoints are directories)."""
    h = hashlib.sha256()
    if os.path.isdir(path):
        for root, _dirs, files in sorted(os.walk(path)):
            for f in sorted(files):
                h.update(f.encode())
                with open(os.path.join(root, f), "rb") as fh:
                    while (b := fh.read(_chunk)):
                        h.update(b)
    else:
        with open(path, "rb") as fh:
            while (b := fh.read(_chunk)):
                h.update(b)
    return h.hexdigest()


class NumpyMLP:
    """SiLU MLP evaluated in numpy — keeps torch out of the deployment.

    Weights come from prior.npz written by scripts/package_policy.py, which verifies this
    forward against the original torch module before writing.
    """

    def __init__(self, weights: dict):
        self.W = [weights[f"W{i}"] for i in range(int(weights["n_layers"]))]
        self.b = [weights[f"b{i}"] for i in range(int(weights["n_layers"]))]
        self.mu = weights["mu"]
        self.sd = weights["sd"]
        self.tasks = [str(t) for t in weights["tasks"]] if "tasks" in weights else []
        self.kind = str(weights["kind"]) if "kind" in weights else "state_prior"

    @staticmethod
    def _silu(x):
        # branch-stable sigmoid: exp(-x) overflows for very negative x, and the naive
        # x/(1+exp(-x)) then loses precision well before it reaches the limit
        z = np.empty_like(x)
        pos = x >= 0
        z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        e = np.exp(x[~pos])
        z[~pos] = e / (1.0 + e)
        return x * z

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = (np.asarray(x, np.float32) - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = self._silu(h)
        return h


class OneHotTasks:
    """Exact-match task encoding — a SCAFFOLD. It cannot handle an instruction outside its
    list and raises rather than guessing; use the language encoder for deployment."""

    def __init__(self, tasks):
        self.tasks = list(tasks)

    def __call__(self, obs):
        p = str(obs.get("prompt", "")).strip()
        if p not in self.tasks:
            raise ValueError(f"prompt not in this bundle's task list: {p!r}. This bundle ships "
                             f"a one-hot scaffold encoder; it cannot generalise to new wording.")
        v = np.zeros(len(self.tasks), np.float32)
        v[self.tasks.index(p)] = 1.0
        return v


class LanguageEncoder:
    """Grounded encoding: PCA-projected post-fusion language-token embedding of the live
    instruction. No task list — any wording produces a vector.

    `latch_n > 0` averages the first n calls of an episode and then holds. That assumes the
    instruction's meaning is constant for the episode and is WRONG for multi-stage
    instructions (the active sub-goal is a live language x image reading); default is 0.
    """

    def __init__(self, policy, Em, P, latch_n: int = 0):
        self.policy = policy
        self.Em = np.asarray(Em, np.float32)
        self.P = np.asarray(P, np.float32)
        self.latch_n = int(latch_n)
        self._acc = None
        self._n = 0

    def reset(self):
        self._acc = None
        self._n = 0

    def _embed(self, obs):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask
        td = self.policy._input_transform(dict(obs))
        b = jax.tree.map(lambda x: jnp.asarray(x)[None], td)
        o = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
        tok, mask, ar = self.policy._model.embed_prefix(o)
        attn = make_attn_mask(mask, ar)
        pos = jnp.cumsum(mask, axis=1) - 1
        (pout, _), _ = self.policy._model.PaliGemma.llm([tok, None], mask=attn, positions=pos)
        n_txt = o.tokenized_prompt.shape[1]
        tm = o.tokenized_prompt_mask[..., None].astype(jnp.float32)
        pf = pout[:, -n_txt:, :].astype(jnp.float32)
        e = np.asarray((pf * tm).sum(1) / jnp.clip(tm.sum(1), 1e-6)).astype(np.float32)[0]
        return (e - self.Em) @ self.P

    def __call__(self, obs):
        if self.latch_n <= 0:
            return self._embed(obs)
        if self._n < self.latch_n:
            e = self._embed(obs)
            self._acc = e if self._acc is None else self._acc + e
            self._n += 1
        return self._acc / max(self._n, 1)


class PinnedPolicy:
    """openpi policy + pinned source noise + a command source.

    The command source is whatever produces c for the current observation. Bundles ship one
    of: 'state_prior' (state + task encoding -> c, numpy MLP) or 'constant' (a fixed c, for
    bring-up and for commanding a known movement without any predictor).
    """

    def __init__(self, policy, U: np.ndarray, prior: NumpyMLP | None, action_std: np.ndarray,
                 manifest: dict, task_encoder=None):
        self.policy = policy
        self.U = np.asarray(U, np.float32)
        self.prior = prior
        self.action_std = np.asarray(action_std, np.float32)
        self.manifest = manifest
        self.task_encoder = task_encoder
        self.H = int(manifest["action_horizon"])
        self.AD = int(manifest["action_dim"])
        if self.U.shape[0] != self.H * self.AD:
            raise ValueError(f"basis {self.U.shape} does not match {self.H}x{self.AD} chunks")
        self._dc = np.zeros(self.U.shape[1], np.float32)
        self._rng = np.random.default_rng()

    # ---- construction -------------------------------------------------------------
    @classmethod
    def from_bundle(cls, path: str, verify: bool = True):
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        if verify:
            for name, expected in manifest["sha256"].items():
                got = sha256_of(os.path.join(path, name))
                if got != expected:
                    raise ValueError(f"bundle artifact {name} does not match its manifest hash")
        import openpi.policies.policy_config as _pc
        import openpi.shared.normalize as _nz
        import openpi.training.config as _cfg
        cfg = _cfg.get_config(manifest["config"])
        ns = _pad_norm_stats(_nz.load(os.path.join(path, "norm_stats")), cfg.model.action_dim)
        # openpi takes the checkpoint DIRECTORY and appends params/ itself
        policy = _pc.create_trained_policy(cfg, path, norm_stats=ns)
        U = np.load(os.path.join(path, "pin_U.npy"))
        prior_path = os.path.join(path, "prior.npz")
        prior, enc = None, None
        if os.path.exists(prior_path):
            w = dict(np.load(prior_path, allow_pickle=True))
            prior = NumpyMLP(w)
            if "Em" in w and "P" in w:            # grounded language prior
                enc = LanguageEncoder(policy, w["Em"], w["P"],
                                      latch_n=int(os.environ.get("SNMVP_LATCH_N", "0")))
            elif "tasks" in w:                    # one-hot scaffold
                enc = OneHotTasks([str(t) for t in w["tasks"]])
        return cls(policy, U, prior, np.asarray(ns["actions"].std), manifest, task_encoder=enc)

    # ---- steering -----------------------------------------------------------------
    def nudge(self, axis: int | str, metres: float) -> np.ndarray:
        """Add `metres` of net chunk displacement along an action axis to every command."""
        j = axis if isinstance(axis, int) else int(self.manifest["axes"].index(axis))
        s = float(self.action_std[j])
        if s < 1e-6:
            raise ValueError(f"axis {axis!r} has zero action std in the training data — not steerable")
        m = np.zeros((self.H, self.AD), np.float64)
        m[:, j] = 1.0 / self.H
        self._dc = self._dc + ((metres / s) * (self.U.T @ m.reshape(-1))).astype(np.float32)
        return self._dc

    def clear_nudge(self):
        self._dc = np.zeros(self.U.shape[1], np.float32)

    def command_displacement(self, c: np.ndarray, axes=(0, 1, 2)) -> np.ndarray:
        """Physical (metres) net displacement encoded by c — for logging and visualisation."""
        y = (self.U.astype(np.float64) @ np.asarray(c, np.float64)).reshape(self.H, self.AD)
        return np.array([y[:, j].sum() * float(self.action_std[j]) for j in axes])

    # ---- inference ----------------------------------------------------------------
    def command_for(self, obs: dict) -> np.ndarray:
        if self.prior is None:
            return np.zeros(self.U.shape[1], np.float32) + self._dc
        state = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
        x = state if self.task_encoder is None else np.concatenate([state, self.task_encoder(obs)])
        return self.prior(x[None])[0].astype(np.float32) + self._dc

    def pinned_noise(self, c: np.ndarray) -> np.ndarray:
        g = self._rng.standard_normal((self.H, self.AD)).astype(np.float32).reshape(-1)
        return (g - (g @ self.U) @ self.U.T + (np.asarray(c, np.float32) @ self.U.T)
                ).reshape(self.H, self.AD).astype(np.float32)

    def act(self, obs: dict, c: np.ndarray | None = None) -> dict:
        """Return the openpi result dict; result['actions'] is the (H, action_dim) chunk."""
        c = self.command_for(obs) if c is None else np.asarray(c, np.float32) + self._dc
        out = dict(self.policy.infer(obs, noise=self.pinned_noise(c)))
        out["snmvp_command"] = c
        out["snmvp_command_displacement"] = self.command_displacement(c)
        return out


def _pad_norm_stats(ns, dim):
    """Pad dataset norm stats up to the model's action dim (openpi pads chunks with zeros)."""
    from openpi.transforms import NormStats
    out = {}
    for k, s in ns.items():
        n = np.asarray(s.mean).shape[-1]
        if n >= dim:
            out[k] = s
            continue
        p = dim - n
        ext = lambda a, fill: None if a is None else np.concatenate(
            [np.asarray(a, np.float32), np.full(p, fill, np.float32)])
        out[k] = NormStats(mean=ext(s.mean, 0.0), std=ext(s.std, 1.0),
                           q01=ext(s.q01, 0.0), q99=ext(s.q99, 1.0))
    return out
