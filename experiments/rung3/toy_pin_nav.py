"""Controlled test of language steerability of the source-noise pin, in the regime LIBERO lacks:
language sets the MOTION, not the target. Task: reach a (randomized) target from the origin, where
the instruction specifies the PATH -- a lateral bow, left..right (5 levels). Same start/target,
different instruction => very different trajectory, so the instruction drives a large share of the
action (and hence of the pin coordinate c=U^T a). We train a small flow-matching policy with the
pin applied exactly as pi0 does (clamp the action's U-coordinate into the source noise), fit a
(state, language)->c prior, and test: E1' variance decomposition of c; E2' prior R^2; E3' pass-
through + reconstruction; E4' steerability (fix target, swap/interpolate language -> path flips).
Mirrors pi0: x_t = t*noise + (1-t)*a, u_t = noise - a; pin: noise = noise - (noise@U)@U.T + (a@U)@U.T.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD, D, K = 16, 2, 32, 3          # path length, action dim (2D), flat dim, pin dim
NINSTR = 5
BOW = np.linspace(-0.5, 0.5, NINSTR)   # instruction = lateral bow (left..right)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen(n):
    ang = np.random.uniform(-0.4, 0.4, n)          # target direction spread (state)
    r = np.random.uniform(0.8, 1.2, n)
    T = np.stack([np.cos(ang) * r, np.sin(ang) * r], 1)   # target (start = origin)
    instr = np.random.randint(0, NINSTR, n)
    ts = np.linspace(0, 1, H + 1)[None, :, None]           # (1,H+1,1)
    line = ts * T[:, None, :]                              # straight interp (n,H+1,2)
    That = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    perp = np.stack([-That[:, 1], That[:, 0]], 1)          # left-perpendicular
    bow = (np.sin(np.pi * ts[..., 0]) * BOW[instr][:, None])[..., None] * perp[:, None, :]
    path = line + bow                                      # (n,H+1,2)
    act = np.diff(path, axis=1).reshape(n, D // AD * AD)   # (n,H*2)=(n,32? ) -> H*AD=32
    return act.astype(np.float32), T.astype(np.float32), instr.astype(np.int64)


class Flow(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(NINSTR, 8)
        s.net = nn.Sequential(nn.Linear(D + 1 + 2 + 8, 256), nn.SiLU(),
                              nn.Linear(256, 256), nn.SiLU(),
                              nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, D))

    def forward(s, x, t, T, instr):
        return s.net(torch.cat([x, t, T, s.emb(instr)], 1))


def to(*a):
    return [torch.as_tensor(x).to(DEV) for x in a]


def main():
    act, T, instr = gen(5000)
    va, vT, vi = gen(1500)
    # pin subspace U from PCA of training actions (same as the real pipeline)
    Xc = act - act.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    U = Vt[:K].T.astype(np.float32)                        # (D,K) orthonormal
    Ut = torch.as_tensor(U).to(DEV)
    cov = float((np.linalg.svd(Xc, compute_uv=False)[:K] ** 2).sum() / (np.linalg.svd(Xc, compute_uv=False) ** 2).sum())

    model = Flow().to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    A, Tt, I = to(act, T, instr)
    n = len(A)
    for step in range(8000):
        idx = torch.randint(0, n, (256,), device=DEV)
        a, tT, ii = A[idx], Tt[idx], I[idx]
        t = torch.rand(len(idx), 1, device=DEV)
        noise = torch.randn_like(a)
        # pin: replace U-coords of noise with the action's U-coords (pass-through)
        noise = noise - (noise @ Ut) @ Ut.T + (a @ Ut) @ Ut.T
        x_t = t * noise + (1 - t) * a
        u_t = noise - a
        loss = ((model(x_t, t, tT, ii) - u_t) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"train done loss={loss.item():.4f}  U K={K} coverage={cov:.3f}")

    # ---------- E1': variance decomposition of c by instruction (val set) ----------
    cval = va @ U                                          # (n,K)
    tot = cval.var(0).sum()
    within = np.mean([cval[vi == k].var(0).sum() * (vi == k).mean() for k in range(NINSTR)]) * NINSTR
    within = sum((vi == k).mean() * cval[vi == k].var(0).sum() for k in range(NINSTR))
    print(f"E1'  c variance: between/language={ (tot-within)/tot*100:5.1f}%  within/state={within/tot*100:5.1f}%  (LIBERO was 1-15%)")

    # ---------- E2': predict c from state / language / both ----------
    def r2(Xf, y):
        Xf = np.concatenate([Xf, np.ones((len(Xf), 1))], 1)
        ntr = int(0.8 * len(Xf))
        W, *_ = np.linalg.lstsq(Xf[:ntr], y[:ntr], rcond=None)
        p = Xf[ntr:] @ W
        return 1 - ((y[ntr:] - p) ** 2).sum() / (((y[ntr:] - y[ntr:].mean(0)) ** 2).sum() + 1e-9)
    oh = np.eye(NINSTR)[vi]
    ctr = act @ U; oh_tr = np.eye(NINSTR)[instr]
    print(f"E2'  state->c R^2={r2(vT, cval):.3f}   language->c R^2={r2(oh, cval):.3f}   "
          f"(state+language)->c R^2={r2(np.concatenate([vT, oh],1), cval):.3f}")
    # prior (state+language)->c fit on train for the steering demo
    Xtr = np.concatenate([T, oh_tr, np.ones((len(T), 1))], 1)
    Wp, *_ = np.linalg.lstsq(Xtr, ctr, rcond=None)

    def prior(Tq, iq):
        return (np.concatenate([Tq, np.eye(NINSTR)[iq], np.ones((len(Tq), 1))], 1) @ Wp).astype(np.float32)

    # ---------- sampler (Euler ODE from pinned noise, pi0 convention) ----------
    @torch.no_grad()
    def sample(Tq, iq, c=None, pin=True, steps=16, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        x = torch.randn(len(Tq), D, generator=g, device=DEV)
        Tq_, iq_ = to(Tq.astype(np.float32), iq.astype(np.int64))
        if pin and c is not None:
            cc = torch.as_tensor(c).to(DEV)
            x = x - (x @ Ut) @ Ut.T + cc @ Ut.T
        for j in range(steps):
            t = torch.full((len(Tq), 1), 1 - j / steps, device=DEV)
            x = x - model(x, t, Tq_, iq_) * (1.0 / steps)
        return x.cpu().numpy()

    # ---------- E3': pass-through + reconstruction on val ----------
    c_real = va @ U
    c_pred = prior(vT, vi)
    gen_oracle = sample(vT, vi, c_real, pin=True)
    gen_prior = sample(vT, vi, c_pred, pin=True)
    gen_nopin = sample(vT, vi, None, pin=False)

    def sub(a_):
        return a_ @ U

    def sr2(gp):
        return 1 - ((sub(gp) - c_real) ** 2).sum() / (((c_real - c_real.mean(0)) ** 2).sum() + 1e-9)
    def fr2(gp):
        return 1 - ((gp - va) ** 2).sum() / (((va - va.mean(0)) ** 2).sum() + 1e-9)
    pt = np.linalg.norm(sub(gen_oracle) - c_real, axis=1).mean() / (np.linalg.norm(c_real, axis=1).mean() + 1e-9)
    print(f"E3'  pass-through relerr(oracle)={pt:.3f}   subspace_R2: nopin={sr2(gen_nopin):.3f} prior={sr2(gen_prior):.3f} oracle={sr2(gen_oracle):.3f}")
    print(f"     full-action R2:              nopin={fr2(gen_nopin):.3f} prior={fr2(gen_prior):.3f} oracle={fr2(gen_oracle):.3f}")

    # ---------- E4': steerability -- fix target, sweep instruction ----------
    Tfix = np.array([[1.0, 0.0]], np.float32)
    viz = {"target": Tfix[0].tolist(), "H": H, "instr_bow": BOW.tolist(), "paths": {}}
    for k in range(NINSTR):
        c_k = prior(Tfix, np.array([k]))
        gen_k = sample(Tfix, np.array([k]), c_k, pin=True)[0].reshape(H, AD)
        dec_k = (c_k @ U.T).reshape(H, AD)                  # decoder: what the pin alone encodes
        path_gen = np.concatenate([[[0, 0]], np.cumsum(gen_k, 0)], 0)
        path_dec = np.concatenate([[[0, 0]], np.cumsum(dec_k, 0)], 0)
        viz["paths"][f"instr_{k}"] = {"bow": float(BOW[k]),
                                      "generated": path_gen.tolist(), "pin_decoded": path_dec.tolist()}
    # interpolation sweep of the pin coordinate between left and right (fixed target)
    cL, cR = prior(Tfix, np.array([0])), prior(Tfix, np.array([NINSTR - 1]))
    interp = []
    for lam in np.linspace(0, 1, 7):
        c_i = (1 - lam) * cL + lam * cR
        gi = sample(Tfix, np.array([NINSTR // 2]), c_i.astype(np.float32), pin=True)[0].reshape(H, AD)
        interp.append(np.concatenate([[[0, 0]], np.cumsum(gi, 0)], 0).tolist())
    viz["interpolation"] = interp
    json.dump(viz, open(os.path.join(RD, "toy_nav_viz.json"), "w"))
    print("TOY_NAV_DONE wrote toy_nav_viz.json")


if __name__ == "__main__":
    main()
