"""Fit a state->c ridge prior on one task's few-shot demos, in the no-delta shared space, and save
it in serve_pca_pin's format (npz with W,b). Uses the data loader (SNMVP_EPISODES) so state/actions
are in the model's normalized space, matching serving. Env: SNMVP_PIN_U, SNMVP_EPISODES,
SNMVP_PRIOR_OUT, SNMVP_CONFIG (default pi0_libero_shared), SNMVP_NB (num batches)."""
import os
import sys

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
sys.path.insert(0, RD)
import pca_pin as PP  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402

U = PP.load_U(os.environ["SNMVP_PIN_U"])
OUT = os.environ["SNMVP_PRIOR_OUT"]
NB = int(os.environ.get("SNMVP_NB", "40"))


def main():
    cfg = _config.get_config(os.environ.get("SNMVP_CONFIG", "pi0_libero_shared"))
    loader = _data_loader.create_data_loader(cfg, sharding=None, shuffle=True, num_batches=NB, framework="jax")
    S, A = [], []
    for obs, act in loader:
        S.append(np.asarray(obs.state)); A.append(np.asarray(act))
    S = np.concatenate(S); A = np.concatenate(A)
    n = len(S); ntr = int(0.8 * n)
    Sf = S.reshape(n, -1); C = A.reshape(n, -1) @ U
    W, b = PP.fit_state_prior(Sf[:ntr], C[:ntr])
    Cpred = PP.apply_prior(W, b, Sf[ntr:])
    r2 = 1 - ((C[ntr:] - Cpred) ** 2).sum() / (((C[ntr:] - C[ntr:].mean(0)) ** 2).sum() + 1e-9)
    np.savez(OUT, W=W, b=b)
    print(f"PRIOR_HH_DONE {OUT} n={n} K={U.shape[1]} R2={r2:.3f}")


if __name__ == "__main__":
    main()
