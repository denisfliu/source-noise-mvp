"""Pose-preserving trajectory algebra for the gate drone data (A1 of the augmentation plan).

Because the drone obs is a pure function of pose (state = [x,y,z,yaw,0,0,0], no
velocities — verified 2026-08-04), any traversal that visits recorded poses can
reuse the recorded frames verbatim. Generators (all return episode dicts in the
data_gate_synth schema {image, wrist, state, action, lang}, images as VIEWS of
the source arrays — materialize only when writing):

  reverse(ep)         — from the endpoint (penguin) back through the gate to the
                        start: states/frames reversed, actions = -action[::-1]
                        (per-step deltas), heading unchanged (turned-around poses
                        were never rendered).
  crop_to_gate(ep)    — start -> gate-plane crossing, + hover extension (repeat
                        final pose with zero actions): "fly to the gate and stop".
  crop_from_gate(ep)  — gate crossing -> end: "fly to the stuffed animal ...".
  hover(ep, t, n)     — hold a single pose: n repeats of frame t, zero actions.

Run as a script for the G0 support checks (CPU): normalization support of
augmented actions, c ranges vs the training clamp, U capture ratio per variant,
and reversal self-consistency. Pre-registered G0 bars are printed with PASS/FAIL.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gate_ctx_common as gc

# right-gate geometry (mocap frame), from the scoring chain (gate_video_overlay.py)
R_GANCH = np.array([0.544, -1.147, 0.074]); R_GNRM = np.array([0.385, -0.923, 0.0])
R_GNRM = R_GNRM / np.linalg.norm(R_GNRM)
L_GANCH = gc.GATE; L_GNRM = gc.NRM0

PROMPT_BACK = "fly back through the gate and return to the start point"
PROMPT_TO_GATE = "fly to the gate and stop in front of it"
PROMPT_TO_TOY = "fly to the stuffed animal and hover over it"
PROMPT_HOLD = "hold position"


def gate_cross_idx(ep):
    """First crossing index of the episode's own gate plane. None for center-route
    episodes (their gate geometry isn't wired here yet — crops skip them; reverse and
    hover remain valid for all tasks)."""
    if ep["lang"] == gc.PROMPT_L:
        anch, nrm = L_GANCH, L_GNRM
    elif ep["lang"] == gc.PROMPT_R:
        anch, nrm = R_GANCH, R_GNRM
    else:
        return None
    s = (ep["state"][:, :3] - anch) @ nrm
    cr = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
    return int(cr[0]) + 1 if len(cr) else None


def _ep(image, wrist, state, action, lang, fidx):
    """fidx: SOURCE frame index per output frame — lets consumers map any augmented
    frame back to the original stored/renderable pose (images may be None)."""
    return {"image": image, "wrist": wrist, "state": state,
            "action": action.astype(np.float32), "lang": lang,
            "fidx": np.asarray(fidx, np.int32)}


def _sl(a, sl):
    return None if a is None else a[sl]


def _trim(ep):
    """Consistent lengths: n actions, n+1 frames (some eps carry a trailing pad action)."""
    n = min(len(ep["action"]), len(ep["state"]) - 1)
    return n


def reverse(ep):
    n = _trim(ep)  # action[t] = state[t+1]-state[t] on dims 0..3
    return _ep(_sl(ep.get("image"), slice(n, None, -1)), _sl(ep.get("wrist"), slice(n, None, -1)),
               ep["state"][n::-1], -ep["action"][:n][::-1], PROMPT_BACK,
               np.arange(n + 1)[::-1])


def _hover_ext(ep, t, n):
    rep = lambda a: None if a is None else np.repeat(a[t:t + 1], n, 0)
    im = rep(ep.get("image")); wr = rep(ep.get("wrist"))
    st = np.repeat(ep["state"][t:t + 1], n, 0)
    ac = np.zeros((n - 1, ep["action"].shape[1]), np.float32)
    return im, wr, st, ac


def crop_to_gate(ep, hover_n=40):
    g = gate_cross_idx(ep)
    if g is None or g < 8:
        return None
    im, wr, st, ac = _hover_ext(ep, g, hover_n)
    cat = lambda a, b: None if a is None else np.concatenate([a[:g + 1], b])
    return _ep(cat(ep.get("image"), im), cat(ep.get("wrist"), wr),
               np.concatenate([ep["state"][:g + 1], st]),
               np.concatenate([ep["action"][:g], np.zeros((1, ep["action"].shape[1]), np.float32), ac]),
               PROMPT_TO_GATE,
               np.concatenate([np.arange(g + 1), np.full(hover_n, g)]))


def crop_from_gate(ep):
    g = gate_cross_idx(ep)
    if g is None or g > len(ep["action"]) - 8:
        return None
    n = _trim(ep)
    return _ep(_sl(ep.get("image"), slice(g, None)), _sl(ep.get("wrist"), slice(g, None)),
               ep["state"][g:], ep["action"][g:], PROMPT_TO_TOY, np.arange(g, len(ep["state"])))


def hover(ep, t, n=60):
    im, wr, st, ac = _hover_ext(ep, t, n)
    return _ep(im, wr, st, ac, PROMPT_HOLD, np.full(n, t))


def augment(eps, hover_per_ep=1, rng=None):
    """Full augmented set: originals + reverse + crops + hovers (forward-dominant)."""
    rng = rng or np.random.default_rng(0)
    out = list(eps)
    for ep in eps:
        out.append(reverse(ep))
        for f in (crop_to_gate, crop_from_gate):
            a = f(ep)
            if a is not None:
                out.append(a)
        for _ in range(hover_per_ep):
            out.append(hover(ep, int(rng.integers(0, len(ep["action"])))))
    return out


# ---------------- G0: offline support checks ----------------
if __name__ == "__main__":
    ns, amean, astd = gc.load_norm()
    s = ns["actions"]
    q01 = np.asarray(s.q01) if s.q01 is not None else None
    q99 = np.asarray(s.q99) if s.q99 is not None else None
    eps = gc.load_eps(with_images=True)
    U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))

    # reversal self-consistency RELATIVE to the data's own action/state consistency:
    # original actions are only approximate finite-diffs (mean dev ~4e-4, spikes ~9cm,
    # teleop/MPC jitter — measured 2026-08-04), so the bar is "reversal adds nothing",
    # not an absolute threshold.
    def recon_err(st, ac):
        rec = st[0, :4] + np.concatenate([np.zeros((1, 4)), np.cumsum(ac[:, :4], 0)])
        return np.abs(rec - st[:len(rec), :4]).max()
    fwd, rev = [], []
    for ep in eps[:20]:
        n = _trim(ep)
        fwd.append(recon_err(ep["state"][:n + 1], ep["action"][:n]))
        r = reverse(ep)
        rev.append(recon_err(r["state"], r["action"]))
    print("G0.a reversal adds no inconsistency: fwd max %.2e vs rev max %.2e  %s"
          % (max(fwd), max(rev), "PASS" if max(rev) <= max(fwd) * 1.05 + 1e-6 else "FAIL"), flush=True)

    variants = {"orig": eps,
                "reverse": [reverse(e) for e in eps],
                "crop_to_gate": [a for e in eps if (a := crop_to_gate(e))],
                "crop_from_gate": [a for e in eps if (a := crop_from_gate(e))],
                "hover": [hover(e, len(e["action"]) // 2) for e in eps]}
    print("episode counts:", {k: len(v) for k, v in variants.items()}, flush=True)

    ok_clip, ok_c = True, True
    for name, veps in variants.items():
        A = np.concatenate([e["action"] for e in veps], 0)[:, :4]
        clip = np.mean((A < q01[:4]) | (A > q99[:4])) if q01 is not None else float("nan")
        C = []
        cap = []
        for e in veps:
            for t in range(0, len(e["action"]), gc.STRIDE):
                y = gc.segY(e["action"][t:], amean, astd)
                c = y @ U
                C.append(c)
                ny = np.linalg.norm(y)
                if ny > 1e-6:
                    cap.append(np.linalg.norm(U @ c) / ny)
        C = np.stack(C)
        out_frac = np.mean((C < gc.CLO) | (C > gc.CHI))
        print("%-14s q-clip %.3f%%  c out-of-clamp %.2f%%  c-range [%.1f, %.1f]  U-capture %.2f"
              % (name, 100 * clip, 100 * out_frac, C.min(), C.max(),
                 np.mean(cap) if cap else float("nan")), flush=True)
        if name != "orig":
            ok_clip &= (np.isnan(clip) or clip < 0.02)
            ok_c &= (out_frac < 0.02)
    print("G0.b augmented action q01/q99 clip <2%%: %s" % ("PASS" if ok_clip else "FAIL"), flush=True)
    print("G0.c augmented c inside training clamp (<2%% out): %s" % ("PASS" if ok_c else "FAIL"), flush=True)
    print("G0_DONE", flush=True)
