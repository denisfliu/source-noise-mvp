"""OAT: Ordered Action Tokenization (Liu et al., arXiv:2602.04215) — toy-scale
autograd implementation, consistent with the toy_frame/toy_embodiment stack
(pure numpy/autograd, CPU, H=20 x 2 action chunks).

Faithful to the paper's two load-bearing pieces:
  - FSQ bottleneck (finite scalar quantization) with straight-through gradient.
  - Nested dropout over an ORDERED sequence of H_l tokens: each training step
    keeps a random prefix K ~ U{1..H_l} and masks the tail, forcing the encoder
    to pack the most reconstruction-critical information into the earliest
    tokens (coarse) and progressively finer detail into later tokens. This is
    the paper's PRIMARY ordering mechanism (Sec IV-B1); the causal attention
    over registers (IV-B2) is a secondary reinforcement we omit at toy scale.

The scientific question this exists to answer (the Rung-1 go/no-go gate): when
an OAT tokenizer is trained JOINTLY across a set of embodiments in a shared
task-space action frame, does the coarse->fine token ordering ALIGN with a
shared->body-specific split? i.e. do early tokens carry the goal/shared motion
(low mutual information with body identity) while body-idiosyncratic detail is
pushed into late tokens? If yes, the early-token prefix is a learned, ordered,
bottlenecked replacement for the hand-built coherence frame S_A, and freezing it
is a principled cross-embodiment invariant. If early tokens leak body identity,
OAT's reconstruction ordering is transcribing (the failure mode Denis already
hit twice) and the fix is a hybrid external ordering signal.
"""

import autograd.numpy as anp
import numpy as np
from autograd.extend import defvjp, primitive
from autograd.misc.optimizers import adam
from autograd import grad


# --------------------------- FSQ (straight-through) -------------------------

@primitive
def _ste_round(x):
    return np.round(x)


# identity gradient: the round is transparent to backprop (STE), exactly as FSQ.
defvjp(_ste_round, lambda ans, x: lambda g: g)


def fsq(h, levels):
    """Bound-and-round each latent scalar to one of `levels` values.
    Returns the quantized value normalized to [-1, 1] (decoder input)."""
    half = (levels - 1) / 2.0
    z = half * anp.tanh(h)
    zq = _ste_round(z)
    return zq / half


def fsq_ids(h, levels):
    """Discrete per-dim integer codes (plain numpy, eval-time; no grad)."""
    half = (levels - 1) / 2.0
    z = half * np.tanh(np.asarray(h, dtype=float))
    return np.round(z).astype(int) + int(half)          # in [0, levels-1]


# ------------------------------ architecture --------------------------------

class OATConfig:
    def __init__(self, H=20, D=2, H_l=8, d_fsq=2, levels=5, hid=128):
        self.H, self.D = H, D
        self.H_l, self.d_fsq, self.levels, self.hid = H_l, d_fsq, levels, hid
        self.in_dim = H * D
        self.n_lat = H_l * d_fsq
        self.codebook = levels ** d_fsq                  # distinct ids per token


def init_params(cfg, seed=0):
    rng = np.random.default_rng(seed)

    def mlp(dims):
        return [(rng.normal(size=(a, b)) / np.sqrt(a), np.zeros(b))
                for a, b in zip(dims[:-1], dims[1:])]

    return {"enc": mlp([cfg.in_dim, cfg.hid, cfg.hid, cfg.n_lat]),
            "dec": mlp([cfg.n_lat, cfg.hid, cfg.hid, cfg.in_dim]),
            "mask": rng.normal(size=cfg.d_fsq) * 0.1}


def _mlp_fwd(layers, x, final_act=None):
    h = x
    for w, b in layers[:-1]:
        h = anp.maximum(0.0, h @ w + b)
    w, b = layers[-1]
    out = h @ w + b
    return final_act(out) if final_act else out


def encode_q(params, cfg, chunks_flat):
    """chunks_flat (N, in_dim) -> quantized latents (N, H_l, d_fsq) in [-1,1]."""
    h = _mlp_fwd(params["enc"], chunks_flat)
    q = fsq(h, cfg.levels)                               # (N, n_lat)
    return q.reshape(q.shape[0], cfg.H_l, cfg.d_fsq)


def decode(params, cfg, tok_reps):
    """tok_reps (N, H_l, d_fsq) -> reconstruction (N, in_dim)."""
    x = tok_reps.reshape(tok_reps.shape[0], cfg.n_lat)
    return _mlp_fwd(params["dec"], x)


# ------------------------------- training -----------------------------------

def make_loss(cfg, chunks_flat, batch=256):
    N = chunks_flat.shape[0]
    ar = anp.arange(cfg.H_l)[None, :]

    def loss(params, it):
        rng = np.random.default_rng(it)
        idx = rng.integers(0, N, size=batch)
        x = chunks_flat[idx]
        q = encode_q(params, cfg, x)                     # (B,H_l,d_fsq)
        K = rng.integers(1, cfg.H_l + 1, size=batch)     # nested dropout prefix
        keep = (ar < K[:, None]).astype(float)[..., None]  # (B,H_l,1)
        reps = keep * q + (1.0 - keep) * params["mask"][None, None, :]
        recon = decode(params, cfg, reps)
        return anp.mean((recon - x) ** 2)

    return loss


def train(cfg, chunks_flat, seed=0, iters=8000, step=2e-3):
    loss = make_loss(cfg, chunks_flat)
    params = init_params(cfg, seed)
    return adam(grad(loss), params, num_iters=iters, step_size=step)


# ------------------------------- eval / probe --------------------------------

def tokenize(params, cfg, chunks_flat):
    """-> token ids (N, H_l): each token's code in [0, codebook)."""
    h = _mlp_fwd(params["enc"], chunks_flat)             # (N, n_lat)
    ids_dim = fsq_ids(h, cfg.levels).reshape(-1, cfg.H_l, cfg.d_fsq)
    powers = cfg.levels ** np.arange(cfg.d_fsq)
    return (ids_dim * powers).sum(-1)                    # (N, H_l)


def recon_at_K(params, cfg, chunks_flat, K):
    """Reconstruct using only the first K tokens (tail -> mask). MSE."""
    q = np.asarray(encode_q(params, cfg, chunks_flat))
    keep = (np.arange(cfg.H_l) < K).astype(float)[None, :, None]
    reps = keep * q + (1.0 - keep) * np.asarray(params["mask"])[None, None, :]
    recon = np.asarray(decode(params, cfg, reps))
    return float(np.mean((recon - chunks_flat) ** 2))
