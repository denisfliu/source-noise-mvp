"""Serve a jointly-trained flow: the command head comes from the flow's OWN checkpoint.

No external prior file, so there is no basis or feature-source pairing to get wrong — the head was
fitted against these exact weights and travels with them. Contrast the langprior path, where a
separately-cached prior could be (and was) served against a different checkpoint's VLM, costing 0/10
closed-loop at an offline c-R2 of 0.94.

  SNMVP_HEAD=1 SNMVP_PIN_U=<U> python serve_gate_pin_joint.py --ckpt <flow> --config pi0_gate \
      --norm <assets> --pin-u <U> --port 8900
"""
import argparse
import os
import sys

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)

# the head submodules are constructed env-gated inside pi0.py, so this must precede openpi imports
import joint_head                                                     # noqa: E402
_pin_u_early = os.environ.get("SNMVP_PIN_U", f"{RD}/pin_U_gate_rrr_k5.npy")
joint_head.enable_head(_pin_u_early)

import openpi.policies.policy_config as _pc                           # noqa: E402
import openpi.shared.normalize as _nz                                 # noqa: E402
import openpi.training.config as _cfg                                 # noqa: E402
from openpi.serving.websocket_policy_server import WebsocketPolicyServer  # noqa: E402
from openpi.transforms import NormStats                               # noqa: E402

H, AD = 50, 32


class SketchPrompt:
    """Corrective-sketch prompting (2026-08-25): SNMVP_PIN_PROMPT names a json
    {"points": [[x,y,z(,yaw)],...], "prompt_after": str, "enter_radius", "step_m",
    "sigma_serve", "end_margin_m"} — a coarse polyline covering ONLY the segment where the
    head goes wrong. The polyline is resampled at demo speed (step_m/control-step) and its
    per-step deltas are projected through U exactly like training actions, so the sketch
    speaks the head's own command language. Per-trial state machine: ARMED (normal serve)
    -> ACTIVE when the drone enters enter_radius of the first point (language prompt swaps
    to prompt_after, c comes from the sketch window at sigma_serve trust, progress tracked
    by forward-monotonic nearest-point — observational, no clock) -> DONE at the end margin
    (head resumes under prompt_after with the calibrated sigma map). The sketch carries no
    dynamics — the flow's denoising residual owns feasibility, which is the factorization
    claim under test."""

    def __init__(self, path, amean, astd, U):
        import json
        d = json.load(open(path))
        pts = np.asarray(d["points"], np.float32)
        if pts.shape[1] == 3:
            pts = np.concatenate([pts, np.zeros((len(pts), 1), np.float32)], 1)
        step = float(d.get("step_m", 0.025))
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1))])
        n = max(int(s[-1] / step), H + 2)
        u = np.linspace(0, s[-1], n)
        yaw = np.unwrap(pts[:, 3].astype(np.float64))
        self.P = np.stack([np.interp(u, s, pts[:, k]) for k in range(3)]
                          + [np.interp(u, s, yaw)], 1).astype(np.float32)
        self.A = np.zeros((n - 1, 7), np.float32)
        self.A[:, :3] = np.diff(self.P[:, :3], axis=0)
        self.A[:, 3] = np.diff(self.P[:, 3])
        self.enter = float(d.get("enter_radius", 0.45))
        self.end_i = n - 1 - max(int(float(d.get("end_margin_m", 0.1)) / step), 1)
        self.sigma = float(d.get("sigma_serve", 0.0))
        self.prompt_after = d["prompt_after"]
        self.amean, self.astd, self.U = amean, astd, U
        self._st = {}
        print(f"[sketch] {path}: {len(pts)} points -> {n} steps ({s[-1]:.2f} m), "
              f"enter_r={self.enter} sigma_serve={self.sigma}", flush=True)

    def step(self, trial, pos):
        """-> (c | None, sigma_serve | None, prompt_override | None, phase 0/1/2).

        Activation is nearest-point-on-the-whole-polyline (entry at that index), not
        radius-to-first-point: replans sample the flight only every ~1.25 m, so a
        first-point trigger both fires early across scene geometry the sketch is meant
        to avoid AND misses flights that join the corridor mid-way (both observed,
        2026-08-25 first screen). Handback requires ARRIVING at the sketch end, not
        just exhausting the index — an off-track flight keeps being pulled by the
        final window instead of being abandoned OOD."""
        st = self._st.setdefault(trial, {"phase": 0, "i": 0})
        if st["phase"] == 0:
            d = np.linalg.norm(self.P[:, :3] - pos, axis=1)
            if d.min() < self.enter:
                st["phase"] = 1
                st["i"] = min(int(d.argmin()), self.end_i - 1)
        if st["phase"] == 1:
            # forward-monotonic, capped just above the ~50-step replan stride so an
            # off-sketch excursion can't free-run the index to the end in one hop
            w = self.P[st["i"]:st["i"] + 90, :3]
            st["i"] += min(int(np.linalg.norm(w - pos, axis=1).argmin()), 65)
            if st["i"] >= self.end_i:
                if np.linalg.norm(pos - self.P[-1, :3]) < 0.6:
                    st["phase"] = 2
                else:
                    st["i"] = self.end_i - 1
        if st["phase"] == 1:
            seg = np.zeros((H, 7), np.float32)
            m = min(H, len(self.A) - st["i"])
            seg[:m] = self.A[st["i"]:st["i"] + m]
            ch = np.zeros((H, AD), np.float32)
            ch[:, :7] = (seg - self.amean) / (self.astd + 1e-6)
            return ch.reshape(-1) @ self.U, self.sigma, self.prompt_after, 1
        if st["phase"] == 2:
            return None, None, self.prompt_after, 2
        return None, None, None, 0


class JointPinPolicy:
    def __init__(self, policy, pin_u_path, act_norm=None):
        self.policy = policy
        self.U = np.load(pin_u_path).astype(np.float32)
        self.sketch = None
        sp = os.environ.get("SNMVP_PIN_PROMPT", "")
        if sp:
            if act_norm is None:
                raise ValueError("SNMVP_PIN_PROMPT needs action norm stats (act_norm)")
            amean = np.asarray(act_norm.mean[:7], np.float32)
            astd = np.asarray(act_norm.std[:7], np.float32)
            self.sketch = SketchPrompt(sp, amean, astd, self.U)
        self._rng = np.random.default_rng(0)
        self.CLOG = os.environ.get("CLOG", "")
        self._log = []
        # MDN serve state: per-trial latched component for pi-hysteresis. Keyed by the client's
        # "snmvp_trial" tag because two batch clients fly interleaved trials against one server —
        # a single global latch would carry one trial's commitment into another's replans.
        self._latch = {}
        self._hyst = float(os.environ.get("SNMVP_GMM_HYST", "0.2"))
        # sigma-gated pin trust (2026-08-21): SNMVP_GMM_SIGGATE="lo,hi,amin" softens the pin when
        # the head's OWN sigma* is high — alpha ramps 1 -> amin as ||sigma*|| goes lo -> hi, and
        # c_eff = alpha*c + (1-alpha)*(g@U) so alpha=0 is exactly the unpinned Gaussian. Grounded:
        # sigma* rank-tracks the head's command error at rho=0.82 on demo frames
        # (sigma_phase_probe); thresholds are demo-sigma quantiles, not phase/regime rules.
        sg = os.environ.get("SNMVP_GMM_SIGGATE", "")
        self._gate = tuple(float(v) for v in sg.split(",")) if sg else None
        if self._gate:
            print(f"[joint] sigma gate: lo,hi,amin = {self._gate}", flush=True)
        # sigma-CONDITIONED serve (2026-08-21, the trained trust dial): SNMVP_SIGMA_MAP names a
        # json {"sig_star": [...], "sig_serve": [...], "cap": f} mapping the head's ||sigma*|| to
        # the pin-noise level the flow was TRAINED to expect (fractions of c-std). The command is
        # always delivered at full amplitude; sigma_serve only tells the flow how much to trust
        # it. Requires a SNMVP_PIN_NOISE_COND-trained checkpoint — on any other flow the value is
        # silently ignored at embed time, so the map is only set for conditioned arms.
        self._sigmap = None
        mp = os.environ.get("SNMVP_SIGMA_MAP", "")
        if mp:
            import json as _json
            d = _json.load(open(mp))
            self._sigmap = (np.asarray(d["sig_star"], np.float32),
                            np.asarray(d["sig_serve"], np.float32), float(d["cap"]))
            print(f"[joint] sigma-conditioned serve: map from {mp} cap={d['cap']}", flush=True)
        print(f"[joint] head from the checkpoint, U {self.U.shape} from {pin_u_path}", flush=True)

    def infer(self, obs):
        obs = dict(obs)
        trial = obs.pop("snmvp_trial", "default")
        sk_c, sk_sig, sk_phase = None, None, 0
        if self.sketch is not None:
            pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
            sk_c, sk_sig, sk_prompt, sk_phase = self.sketch.step(trial, pos)
            if sk_prompt is not None:
                obs["prompt"] = sk_prompt      # before head_c: the head sees the swapped task
            if sk_phase == 1:
                self._latch.pop(trial, None)   # re-latch fresh on handback under the new prompt
        if hasattr(self.policy._model, "snmvp_gmm_out"):
            # argmax-mode serve with hysteresis: switch components only when the incumbent's
            # weight has fallen more than SNMVP_GMM_HYST below the argmax — commits like the toy's
            # gmm_argmax but without chattering at pi ~ 0.5. pi AND the served component's
            # ||sigma|| are logged per replan (CLOG row = [pos(3), c(K), pi(M), signorm(1)]):
            # sigma tracks the head's own command error at rho=0.82 on demo frames
            # (sigma_phase_probe 2026-08-20), so the closed-loop sigma trace is the direct test
            # of whether the thrash rows are confident-wrong or known-uncertain.
            c, w, mu, sig = joint_head.head_c(self.policy, [obs], return_gmm=True)
            w, mu, sig = w[0], mu[0], sig[0]
            j = int(w.argmax())
            jprev = self._latch.get(trial)
            if jprev is not None and w[jprev] >= w[j] - self._hyst:
                j = jprev
            self._latch[trial] = j
            c = mu[j]
            sstar = float(np.linalg.norm(sig[j]))
            alpha = 1.0
            if self._gate:
                lo, hi, amin = self._gate
                alpha = float(np.clip((hi - sstar) / max(hi - lo, 1e-6), amin, 1.0))
            sig_serve = None
            if self._sigmap is not None:
                xs, ys, cap = self._sigmap
                sig_serve = float(np.clip(np.interp(sstar, xs, ys), 0.0, cap))
            extra = np.concatenate([w, [sstar, alpha,
                                        sig_serve if sig_serve is not None else -1.0,
                                        sk_phase]]).astype(np.float32)
        else:
            c, alpha, sig_serve = joint_head.head_c(self.policy, [obs])[0], 1.0, None
            extra = np.asarray([sk_phase], np.float32)
        if sk_c is not None:
            c, sig_serve, alpha = sk_c.astype(np.float32), sk_sig, 1.0
        if self.CLOG:
            pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
            self._log.append(np.concatenate([pos, c, extra]).astype(np.float32))
            np.save(self.CLOG, np.stack(self._log))
        g = self._rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        c_eff = alpha * c + (1.0 - alpha) * (g @ self.U)
        noise = (g - (g @ self.U) @ self.U.T + (c_eff @ self.U.T)).reshape(H, AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise, snmvp_sigma=sig_serve)


def _pad(ns, dim):
    o = {}
    for k, s in ns.items():
        n = len(s.mean)
        if n >= dim:
            o[k] = s
            continue
        p = dim - n
        ext = lambda a, f: None if a is None else np.concatenate(
            [np.asarray(a, np.float32), np.full(p, f, np.float32)])
        o[k] = NormStats(mean=ext(s.mean, 0), std=ext(s.std, 1), q01=ext(s.q01, 0), q99=ext(s.q99, 1))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi0_gate")
    ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", default=f"{RD}/pin_U_gate_rrr_k5.npy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8900)
    a = ap.parse_args()
    cfg = _cfg.get_config(a.config)
    raw_ns = _nz.load(a.norm)
    ns = _pad(raw_ns, cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    pin = JointPinPolicy(policy, a.pin_u, act_norm=raw_ns["actions"])
    print(f"[serve_gate_pin_joint] ready on ws://{a.host}:{a.port}", flush=True)
    WebsocketPolicyServer(policy=pin, host=a.host, port=a.port).serve_forever()


if __name__ == "__main__":
    main()
