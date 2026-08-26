"""Command-head architecture toy: MSE vs concat-CFM vs FiLM-CFM vs GMM (MDN) under branch-state
ambiguity (2026-08-19, local CPU machine — continuation of the generative-head line after box
access loss; see RESEARCH_LOG 2026-08-13 entries and status_latest.md).

What it measures. The box's frontier finding was: the generative CFM head fixes command VALIDITY
(full-magnitude draws, center routing) but its posterior CALIBRATION over route modes is
uncontrolled — left-mode fraction under a left prompt ranged 0.0-0.6 across lam/detach and gate
ownership flipped on training seed, even though the conditioning information was fully present in
the head's input features (feature_separation_probe: probe-acc 1.00 at start). The localized
mechanism was CONDITIONING NEGLECT in the CFM velocity field (concat coupling is ignorable; ctx
gradients concentrate at low t). Two fixes were on the table when access ended: FiLM conditioning
(genfilm, launched, results unseen) and — the subject of this toy — an explicit mixture head whose
posterior is a trained, inspectable quantity rather than an emergent property of a velocity field.

This toy isolates the HEAD axis from the feature axis: features here carry task identity cleanly
(language embedding) and carry phase with a measured degradation near the tail (mirroring the
uniform tail-separability decay, 0.56-0.59, found across all box arms). Ground truth p(c | task, f)
is near-deterministic; ambiguity enters only through what the features can resolve. Heads compared
at matched trunk capacity and optimization budget, 5 training seeds each:

  mse         MLP regression (the pre-generative baseline; expect midpoint/shrinkage failures)
  cfm         conditional flow matching, conditioning by CONCAT (gen1-style)
  cfm_film    same CFM trunk, conditioning ONLY via per-layer FiLM (genfilm-style)
  gmm         mixture density network: explicit pi(o), mu(o), sigma(o), NLL loss; sample at serve
  gmm_argmax  same trained GMM, serve = mean of the argmax-weight component

Instruments (ported from the box):
  start calibration   left-mode fraction of draws at f<0.05 under the left prompt (truth 1.0 —
                      language fully determines the task; the C2/gen1 coin-flip situation)
  validity            distance of draws to the nearest true mode center, in jitter-sigma units
                      (mode-averaging produces invalid between-mode commands)
  tail calibration    P(stop-mode) vs the BAYES-OPTIMAL posterior computed from the generative
                      model by grid integration over phase (mean absolute calibration error on
                      boundary rows) — sampling cannot beat this; miscalibration shows against it
  shrinkage           at Bayes-ambiguous boundary rows, |forward command| of the head's MEAN
                      vs the late/stop mode magnitudes (the mh16 2.48-below-all-modes signature)
  seed spread         all of the above across 5 training seeds (the box's mode-allocation lottery)

Run: python toy_cmdhead.py            (CPU, ~2 min; writes results/toy_cmdhead.json)
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

RD = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- generative model
# Command space K=2: (lateral, forward). Piecewise-unimodal given (task, true phase f):
#   f < 0.70          route/cruise   (lat(task), 3.3)
#   0.70 <= f < 0.85  late           (0.3*lat(task), 6.8)
#   f >= 0.85         stop           (0.3, -3.0)   (normalized-unit stop is a NONZERO signature)
# Jitter sigma 0.3. Features: pos_lat (reveals task once moving; ~0 at start), pos_fwd
# (saturates at f=0.8 — goal hover aliases position), f_obs (phase cue, noise 0.01 -> 0.12 past
# f=0.7 — the tail observability decay), lang (8-d task embedding + paraphrase noise).
LAT = {0: 3.0, 1: -4.5}          # task 0 = left, 1 = right
JIT = 0.3
POS_FWD_NOISE = 0.3
F_OBS_NOISE_LO, F_OBS_NOISE_HI = 0.01, 0.12
_lang_rng = np.random.default_rng(1234)
LANG = {t: _lang_rng.normal(size=8).astype(np.float32) for t in (0, 1)}


def true_mode(task, f):
    if f < 0.70:
        return np.array([LAT[task], 3.3])
    if f < 0.85:
        return np.array([0.3 * LAT[task], 6.8])
    return np.array([0.3, -3.0])


def mode_centers(task):
    return np.stack([[LAT[task], 3.3], [0.3 * LAT[task], 6.8], [0.3, -3.0]])


def sample_rows(n, rng):
    task = rng.integers(0, 2, size=n)
    f = rng.uniform(0, 1, size=n)
    c = np.stack([true_mode(t, ff) for t, ff in zip(task, f)]) + rng.normal(0, JIT, (n, 2))
    pos_lat = np.array([LAT[t] for t in task]) * np.minimum(f, 0.7) + rng.normal(0, 0.2, n)
    pos_fwd = 10 * np.minimum(f, 0.8) + rng.normal(0, POS_FWD_NOISE, n)
    sig = np.where(f < 0.7, F_OBS_NOISE_LO, F_OBS_NOISE_HI)
    f_obs = f + rng.normal(0, 1, n) * sig
    lang = np.stack([LANG[t] for t in task]) + rng.normal(0, 0.1, (n, 8))
    X = np.column_stack([pos_lat, pos_fwd, f_obs, lang]).astype(np.float32)
    return X, c.astype(np.float32), task, f


def bayes_stop_posterior(pos_fwd, f_obs):
    """Bayes-optimal P(f >= 0.85 | pos_fwd, f_obs) by grid integration (uniform prior on f)."""
    fg = np.linspace(0, 1, 501)
    sig = np.where(fg < 0.7, F_OBS_NOISE_LO, F_OBS_NOISE_HI)
    ll = (-0.5 * ((pos_fwd[:, None] - 10 * np.minimum(fg, 0.8)) / POS_FWD_NOISE) ** 2
          - 0.5 * ((f_obs[:, None] - fg) / sig) ** 2 - np.log(sig))
    ll -= ll.max(1, keepdims=True)
    p = np.exp(ll)
    return p[:, fg >= 0.85].sum(1) / p.sum(1)


# ---------------------------------------------------------------- heads
def mlp(din, dout, hid=128):
    return nn.Sequential(nn.Linear(din, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU(),
                         nn.Linear(hid, dout))


class GMMHead(nn.Module):
    def __init__(self, din, k=2, m=4):
        super().__init__()
        self.k, self.m = k, m
        self.net = mlp(din, m * (1 + 2 * k))

    def forward(self, x):
        o = self.net(x)
        logit = o[:, :self.m]
        mu = o[:, self.m:self.m * (1 + self.k)].reshape(-1, self.m, self.k)
        logsig = o[:, self.m * (1 + self.k):].reshape(-1, self.m, self.k).clamp(-5, 2)
        return logit, mu, logsig

    def nll(self, x, c):
        logit, mu, logsig = self(x)
        logw = torch.log_softmax(logit, -1)
        comp = (-0.5 * ((c[:, None] - mu) / logsig.exp()) ** 2 - logsig
                - 0.5 * np.log(2 * np.pi)).sum(-1)
        return -torch.logsumexp(logw + comp, -1).mean()

    @torch.no_grad()
    def draw(self, x, ns, gen, argmax=False):
        logit, mu, logsig = self(x)
        w = torch.softmax(logit, -1)
        out = []
        for _ in range(ns):
            j = (w.argmax(-1) if argmax
                 else torch.multinomial(w, 1, generator=gen).squeeze(-1))
            m_sel = mu[torch.arange(len(x)), j]
            s_sel = logsig.exp()[torch.arange(len(x)), j]
            eps = 0 if argmax else torch.randn(m_sel.shape, generator=gen)
            out.append(m_sel + s_sel * eps)
        return torch.stack(out, 1)


class CFMHead(nn.Module):
    """v(c_t, t, ctx) with pi0's convention x_t = t*noise + (1-t)*c, target v = noise - c.
    film=False: ctx enters by concat (gen1). film=True: trunk sees only (c_t, temb); ctx enters
    exclusively as per-layer (1+gamma)*h + beta (genfilm)."""

    def __init__(self, din, k=2, hid=128, film=False):
        super().__init__()
        self.k, self.film, self.hid = k, film, hid
        tin = k + 1 + (0 if film else din)
        self.l1, self.l2, self.l3 = nn.Linear(tin, hid), nn.Linear(hid, hid), nn.Linear(hid, k)
        if film:
            self.g1, self.g2 = mlp(din, 2 * hid, 64), mlp(din, 2 * hid, 64)

    def v(self, ct, t, x):
        z = torch.cat([ct, t], -1) if self.film else torch.cat([ct, t, x], -1)
        h = torch.nn.functional.silu(self.l1(z))
        if self.film:
            g, b = self.g1(x).chunk(2, -1)
            h = h * (1 + g) + b
        h = torch.nn.functional.silu(self.l2(h))
        if self.film:
            g, b = self.g2(x).chunk(2, -1)
            h = h * (1 + g) + b
        return self.l3(h)

    def loss(self, x, c, gen):
        t = torch.rand(len(x), 1, generator=gen)
        noise = torch.randn(c.shape, generator=gen)
        ct = t * noise + (1 - t) * c
        return ((self.v(ct, t, x) - (noise - c)) ** 2).mean()

    @torch.no_grad()
    def draw(self, x, ns, gen, steps=10):
        out = []
        for _ in range(ns):
            ct = torch.randn(len(x), self.k, generator=gen)
            for i in range(steps):
                t = torch.full((len(x), 1), 1 - i / steps)
                ct = ct - self.v(ct, t, x) / steps
            out.append(ct)
        return torch.stack(out, 1)


def train_head(kind, Xtr, Ctr, seed, steps=4000, bs=256):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.tensor((Xtr - m) / s)
    ct = torch.tensor(Ctr)
    if kind == "mse":
        net = mlp(Xtr.shape[1], 2)
    elif kind == "gmm":
        net = GMMHead(Xtr.shape[1])
    else:
        net = CFMHead(Xtr.shape[1], film=(kind == "cfm_film"))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    for _ in range(steps):
        b = torch.randint(0, len(xt), (bs,), generator=gen)
        if kind == "mse":
            loss = ((net(xt[b]) - ct[b]) ** 2).mean()
        elif kind == "gmm":
            loss = net.nll(xt[b], ct[b])
        else:
            loss = net.loss(xt[b], ct[b], gen)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, (m, s)


def draws(kind, net, norm, X, ns, seed):
    gen = torch.Generator().manual_seed(seed + 999)
    xt = torch.tensor((X - norm[0]) / norm[1])
    if kind == "mse":
        with torch.no_grad():
            return net(xt)[:, None].repeat(1, ns, 1).numpy()
    if kind == "gmm":
        return net.draw(xt, ns, gen).numpy()
    if kind == "gmm_argmax":
        return net.draw(xt, ns, gen, argmax=True).numpy()
    return net.draw(xt, ns, gen).numpy()


# ---------------------------------------------------------------- evaluation
def evaluate(kind, net, norm, rng_eval, seed, ns=8):
    res = {}
    # start calibration: f<0.05, LEFT prompt only; truth = 100% left-route mode.
    n = 400
    f = rng_eval.uniform(0, 0.05, n)
    task = np.zeros(n, int)
    pos_lat = rng_eval.normal(0, 0.2, n)  # start states identical across tasks
    pos_fwd = 10 * np.minimum(f, 0.8) + rng_eval.normal(0, POS_FWD_NOISE, n)
    f_obs = f + rng_eval.normal(0, F_OBS_NOISE_LO, n)
    lang = np.stack([LANG[0]] * n) + rng_eval.normal(0, 0.1, (n, 8))
    Xs = np.column_stack([pos_lat, pos_fwd, f_obs, lang]).astype(np.float32)
    d = draws(kind, net, norm, Xs, ns, seed)                    # (n, ns, 2)
    dl = np.linalg.norm(d - np.array([LAT[0], 3.3]), axis=-1)
    dr = np.linalg.norm(d - np.array([LAT[1], 3.3]), axis=-1)
    res["start_left_mode_frac"] = float((dl < dr).mean())
    res["start_validity_sigma"] = float(np.minimum(dl, dr).mean() / JIT)

    # tail calibration + validity + shrinkage on boundary rows f in [0.75, 0.95].
    n = 1200
    f = rng_eval.uniform(0.75, 0.95, n)
    task = rng_eval.integers(0, 2, n)
    Xb, _, _, _ = (None, None, None, None)
    pos_lat = np.array([LAT[t] for t in task]) * np.minimum(f, 0.7) + rng_eval.normal(0, 0.2, n)
    pos_fwd = 10 * np.minimum(f, 0.8) + rng_eval.normal(0, POS_FWD_NOISE, n)
    sig = np.where(f < 0.7, F_OBS_NOISE_LO, F_OBS_NOISE_HI)
    f_obs = f + rng_eval.normal(0, 1, n) * sig
    lang = np.stack([LANG[t] for t in task]) + rng_eval.normal(0, 0.1, (n, 8))
    Xb = np.column_stack([pos_lat, pos_fwd, f_obs, lang]).astype(np.float32)
    p_bayes = bayes_stop_posterior(pos_fwd, f_obs)
    d = draws(kind, net, norm, Xb, ns, seed)                    # (n, ns, 2)
    cents = np.stack([mode_centers(t) for t in task])           # (n, 3, 2)
    dist = np.linalg.norm(d[:, :, None] - cents[:, None], axis=-1)   # (n, ns, 3)
    res["tail_validity_sigma"] = float(dist.min(-1).mean() / JIT)
    p_stop = (dist.argmin(-1) == 2).mean(1)
    res["tail_calib_mae"] = float(np.abs(p_stop - p_bayes).mean())
    amb = (p_bayes > 0.2) & (p_bayes < 0.8)
    mean_pred = d.mean(1)
    md = np.linalg.norm(mean_pred[:, None] - cents, axis=-1).min(-1)
    res["amb_rows"] = int(amb.sum())
    res["amb_mean_to_mode_sigma"] = float(md[amb].mean() / JIT) if amb.any() else None
    res["amb_absfwd_of_mean"] = float(np.abs(mean_pred[amb, 1]).mean()) if amb.any() else None
    return res


def main():
    rng = np.random.default_rng(0)
    Xtr, Ctr, _, _ = sample_rows(20000, rng)
    kinds = ["mse", "cfm", "cfm_film", "gmm", "gmm_argmax"]
    seeds = [0, 1, 2, 3, 4]
    all_res = {k: [] for k in kinds}
    trained = {}
    for seed in seeds:
        for kind in ["mse", "cfm", "cfm_film", "gmm"]:
            trained[(kind, seed)] = train_head(kind, Xtr, Ctr, seed)
        trained[("gmm_argmax", seed)] = trained[("gmm", seed)]
        for kind in kinds:
            net, norm = trained[(kind, seed)]
            r = evaluate(kind, net, norm, np.random.default_rng(7), seed)
            r["seed"] = seed
            all_res[kind].append(r)
            print(f"seed {seed} {kind:10s} "
                  f"startL={r['start_left_mode_frac']:.2f} "
                  f"startVal={r['start_validity_sigma']:.2f} "
                  f"tailVal={r['tail_validity_sigma']:.2f} "
                  f"tailCalMAE={r['tail_calib_mae']:.3f} "
                  f"ambMean2mode={r['amb_mean_to_mode_sigma'] if r['amb_mean_to_mode_sigma'] is None else round(r['amb_mean_to_mode_sigma'], 2)} "
                  f"|fwd|amb={r['amb_absfwd_of_mean'] if r['amb_absfwd_of_mean'] is None else round(r['amb_absfwd_of_mean'], 2)}",
                  flush=True)
    print("\n=== seed-aggregated (mean [min..max] over 5 training seeds) ===")
    for kind in kinds:
        rows = all_res[kind]
        def agg(key):
            v = [r[key] for r in rows if r[key] is not None]
            return f"{np.mean(v):.2f} [{min(v):.2f}..{max(v):.2f}]" if v else "n/a"
        print(f"{kind:10s} startL={agg('start_left_mode_frac')}  "
              f"tailVal={agg('tail_validity_sigma')}  tailCalMAE={agg('tail_calib_mae')}  "
              f"ambMean2mode={agg('amb_mean_to_mode_sigma')}")
    os.makedirs(f"{RD}/results", exist_ok=True)
    with open(f"{RD}/results/toy_cmdhead.json", "w") as fh:
        json.dump(all_res, fh, indent=1)
    print("TOY_CMDHEAD_DONE")


if __name__ == "__main__":
    main()
