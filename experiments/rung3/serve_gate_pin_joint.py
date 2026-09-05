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


from sketch_prompt import SketchPrompt  # noqa: E402  (extracted 2026-08-30)
from advice_prompt import AdvicePrompt  # noqa: E402  (2026-09-03)
from reason_prompt import ReasonPrompt  # noqa: E402  (2026-09-03)


class JointPinPolicy:
    def __init__(self, policy, pin_u_path, act_norm=None):
        self.policy = policy
        self.U = np.load(pin_u_path).astype(np.float32)
        self._rng = np.random.default_rng(int(os.environ.get("SNMVP_NOISE_SEED", "0")))
        # SNMVP_NOISE_SEED varies the residual-noise stream (the pin's orthogonal
        # complement) for rollout-seed replications; default 0 = historical behavior
        self.sketch = None
        sp = os.environ.get("SNMVP_PIN_PROMPT", "")
        if sp:
            if act_norm is None:
                raise ValueError("SNMVP_PIN_PROMPT needs action norm stats (act_norm)")
            amean = np.asarray(act_norm.mean[:7], np.float32)
            astd = np.asarray(act_norm.std[:7], np.float32)
            self.sketch = SketchPrompt(sp, amean, astd, self.U)
        # minimal command-space advice (2026-09-03): SNMVP_PIN_ADVICE names an AdvicePrompt json;
        # after the gate-1 transit only the named command coordinates are overridden by a
        # pursuit toward the target(s); the head keeps the rest. Mutually exclusive with a sketch.
        self.advice = None
        adv = os.environ.get("SNMVP_PIN_ADVICE", "")
        if adv:
            if self.sketch is not None:
                raise ValueError("SNMVP_PIN_ADVICE and SNMVP_PIN_PROMPT are mutually exclusive")
            if act_norm is None:
                raise ValueError("SNMVP_PIN_ADVICE needs action norm stats (act_norm)")
            amean = np.asarray(act_norm.mean[:7], np.float32)
            astd = np.asarray(act_norm.std[:7], np.float32)
            self.advice = AdvicePrompt(adv, amean, astd, self.U)
        # VLM movement reasoning (2026-09-03): SNMVP_PIN_REASON=http://host:port of vlm_reason_server;
        # the reasoner's words fill the coarse coordinates every replan, the head keeps the rest.
        self.reason = None
        ru = os.environ.get("SNMVP_PIN_REASON", "")
        if ru:
            if self.sketch is not None or self.advice is not None:
                raise ValueError("SNMVP_PIN_REASON is exclusive with SNMVP_PIN_PROMPT / SNMVP_PIN_ADVICE")
            if act_norm is None:
                raise ValueError("SNMVP_PIN_REASON needs action norm stats (act_norm)")
            amean = np.asarray(act_norm.mean[:7], np.float32)
            astd = np.asarray(act_norm.std[:7], np.float32)
            self.reason = ReasonPrompt(ru, amean, astd, self.U, mode=os.environ.get("SNMVP_REASON_MODE", "coarse_xyz"),
                                       log_path=os.environ.get("SNMVP_REASON_LOG", ""))
        # ablation (2026-09-03): SNMVP_PIN_OFF=1 serves the pin-trained flow with PLAIN Gaussian noise —
        # no command in the source — at trust sigma SNMVP_PIN_OFF_SIGMA (default: the sigma cap 1.5,
        # i.e. "do not trust the command"). The head still runs (logged) but is not used.
        # diagnostic (2026-09-04): SNMVP_PIN_DECODE_ONLY=1 executes the decoded command U c itself —
        # the minimum-norm chunk, no denoising — so the trajectory shows what the 16 words encode alone.
        # =1: execute the decoded minimum-norm chunk U c (deterministic). =2: execute the PINNED SOURCE SAMPLE
        # itself, z = g - U U^T g + U c (Gaussian in the orthogonal complement, exactly c along U), no denoising.
        self.decode_only = int(os.environ.get("SNMVP_PIN_DECODE_ONLY", "0") or 0)
        self._amean7 = None if act_norm is None else np.asarray(act_norm.mean[:7], np.float32)
        self._astd7 = None if act_norm is None else np.asarray(act_norm.std[:7], np.float32)
        if self.decode_only:
            print(f"[joint] DECODE ONLY mode {self.decode_only}: " + ("executing U c (deterministic), no flow" if self.decode_only == 1
                  else "executing the pinned source sample z itself, no flow"), flush=True)
        self.pin_off = os.environ.get("SNMVP_PIN_OFF", "") == "1"
        self.pin_off_sigma = float(os.environ.get("SNMVP_PIN_OFF_SIGMA", "1.5"))
        if self.pin_off:
            print(f"[joint] PIN OFF: plain noise, sigma_serve={self.pin_off_sigma}", flush=True)
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
        self.bridge = None
        bp = os.environ.get("SNMVP_INTENT_WS", "")
        if bp:
            from intent_bridge import IntentBridge
            self.bridge = IntentBridge(int(bp))
            self._amean_b = np.asarray(act_norm.mean[:32] if act_norm is not None else np.zeros(32), np.float32)
            self._astd_b = np.asarray(np.pad(np.asarray(act_norm.std, np.float32),
                                             (0, max(0, 32 - len(act_norm.std)))), np.float32)                 if act_norm is not None else np.ones(32, np.float32)
            self._cstd_b = np.std(self._rng.standard_normal((512, H * AD)).astype(np.float32) @ self.U, axis=0)
            try:
                z = np.load("/home/dfliu/ctxrun/pingap_rows.npz")
                self._cstd_b = z["cstd"].astype(np.float32)
            except Exception:
                pass
            print(f"[joint] intent bridge on ws://127.0.0.1:{bp}", flush=True)
        print(f"[joint] head from the checkpoint, U {self.U.shape} from {pin_u_path}", flush=True)

    def infer(self, obs):
        obs = dict(obs)
        trial = obs.pop("snmvp_trial", "default")
        # dual-obs mode (real-in-the-loop emulator, 2026-08-28): when the client ships
        # snmvp_cmd_image/_wrist, the COMMAND path (head) reads those (e.g. the sim twin's
        # render at the current pose) while the flow executes on the main observation
        cmd_obs = None
        if "snmvp_cmd_image" in obs:
            cmd_obs = dict(obs)
            cmd_obs["observation/image"] = obs.pop("snmvp_cmd_image")
            cmd_obs["observation/wrist_image"] = obs.pop("snmvp_cmd_wrist")
            for k in ("snmvp_cmd_image", "snmvp_cmd_wrist"):
                cmd_obs.pop(k, None)
        sk_c, sk_sig, sk_phase = None, None, 0
        if self.sketch is not None:
            pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
            sk_c, sk_sig, sk_prompt, sk_phase = self.sketch.step(trial, pos)
            if sk_prompt is not None:
                obs["prompt"] = sk_prompt      # before head_c: the head sees the swapped task
            if sk_phase == 1:
                self._latch.pop(trial, None)   # re-latch fresh on handback under the new prompt
        adv_ch = None
        if self.advice is not None:
            pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
            adv_ch, _adv_sig, adv_prompt, sk_phase = self.advice.window(trial, pos)
            if adv_prompt is not None:
                obs["prompt"] = adv_prompt     # the head sees the remaining task
            if sk_phase == 1:
                self._latch.pop(trial, None)
        if hasattr(self.policy._model, "snmvp_gmm_out"):
            # argmax-mode serve with hysteresis: switch components only when the incumbent's
            # weight has fallen more than SNMVP_GMM_HYST below the argmax — commits like the toy's
            # gmm_argmax but without chattering at pi ~ 0.5. pi AND the served component's
            # ||sigma|| are logged per replan (CLOG row = [pos(3), c(K), pi(M), signorm(1)]):
            # sigma tracks the head's own command error at rho=0.82 on demo frames
            # (sigma_phase_probe 2026-08-20), so the closed-loop sigma trace is the direct test
            # of whether the thrash rows are confident-wrong or known-uncertain.
            c, w, mu, sig = joint_head.head_c(self.policy, [cmd_obs or obs], return_gmm=True)
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
            c, alpha, sig_serve = joint_head.head_c(self.policy, [cmd_obs or obs])[0], 1.0, None
            extra = np.asarray([sk_phase], np.float32)
        if sk_c is not None:
            c, sig_serve, alpha = sk_c.astype(np.float32), sk_sig, 1.0
        if adv_ch is not None:
            c, sig_serve, alpha = self.advice.compose(c, adv_ch), 0.0, 1.0
        if self.reason is not None:
            rs_ch, _trace = self.reason.window(trial, obs, obs.get("prompt", ""))
            c, sig_serve, alpha = self.reason.compose(c, rs_ch), 0.0, 1.0
        if self.CLOG:
            pos = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
            self._log.append(np.concatenate([pos, c, extra]).astype(np.float32))
            np.save(self.CLOG, np.stack(self._log))
        if self.bridge is not None:
            from intent_bridge import sentence as _sent
            import math as _m
            pos_b = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
            while True:
                intent = pos_b + np.cumsum((self.U @ c).reshape(H, AD)[:, :3]
                                           * self._astd_b[:3], axis=0)
                if self.bridge.sigma_override is not None:
                    sig_serve = float(self.bridge.sigma_override)
                dec = self.bridge.propose({
                    "pos": pos_b.tolist(), "c": np.asarray(c, np.float32).tolist(),
                    "sigma_star": float(sstar) if "sstar" in dir() else -1.0,
                    "sigma_serve": sig_serve if sig_serve is not None else -1.0,
                    "phase": int(sk_phase),
                    "intent": np.concatenate([pos_b[None], intent]).tolist(),
                    "text": _sent(np.asarray(c, np.float32), self.U, self._astd_b, self._cstd_b)})
                if dec.get("action") == "rotate":
                    th = _m.radians(float(dec.get("deg", 15)))
                    a_c = (self.U @ c).reshape(H, AD).copy()
                    R = np.array([[_m.cos(th), -_m.sin(th)], [_m.sin(th), _m.cos(th)]], np.float32)
                    a_c[:, :2] = a_c[:, :2] @ R.T
                    c = a_c.reshape(-1) @ self.U
                    continue
                break
        g = self._rng.standard_normal((H, AD)).astype(np.float32).reshape(-1)
        if self.decode_only:
            cc = np.asarray(c, np.float32)
            src = (self.U @ cc) if self.decode_only == 1 else (g - (g @ self.U) @ self.U.T + cc @ self.U.T)
            ch = src.reshape(H, AD)[:, :7]
            act = ch * (self._astd7 + 1e-6) + self._amean7
            return {"actions": act.astype(np.float32), "state": np.asarray(obs["observation/state"], np.float32)}
        if self.pin_off:
            out = self.policy.infer(obs, noise=g.reshape(H, AD).astype(np.float32), snmvp_sigma=self.pin_off_sigma)
            return out
        c_eff = alpha * c + (1.0 - alpha) * (g @ self.U)
        noise = (g - (g @ self.U) @ self.U.T + (c_eff @ self.U.T)).reshape(H, AD).astype(np.float32)
        out = self.policy.infer(obs, noise=noise, snmvp_sigma=sig_serve)
        if self.bridge is not None:
            acts = np.asarray(out["actions"], np.float32)[:H, :3]
            pos_b = np.asarray(obs["observation/state"], np.float32).reshape(-1)[:3]
            self.bridge.executed({"chunk": np.concatenate(
                [pos_b[None], pos_b + np.cumsum(acts, axis=0)]).tolist()})
        return out


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
