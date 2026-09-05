"""SketchPrompt extracted from serve_gate_pin_joint (2026-08-30) so head-free servers
(scratch-sketch mechanism control) can reuse the identical state machine."""
import numpy as np

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
        # pursuit carrot (2026-08-28): when >0, the served window is built from a rejoin
        # curve — the drone's cross-track offset decays to zero over `carrot` steps — so
        # the pin itself carries the comeback instead of outsourcing lateral correction
        # to the vision residual. carrot=0 reproduces the original open-track behavior.
        self.carrot = int(d.get("carrot", 0))
        self.prompt_after = d["prompt_after"]
        self.amean, self.astd, self.U = amean, astd, U
        self._st = {}
        print(f"[sketch] {path}: {len(pts)} points -> {n} steps ({s[-1]:.2f} m), "
              f"enter_r={self.enter} sigma_serve={self.sigma}", flush=True)

    def step(self, trial, pos):
        """-> (c | None, sigma_serve | None, prompt_override | None, phase 0/1/2): the
        window's projection through U (the pin's command)."""
        ch, sig, prompt, phase = self.window(trial, pos)
        return (None if ch is None else ch.reshape(-1) @ self.U), sig, prompt, phase

    def window(self, trial, pos):
        """-> (normalized (H, AD) sketch chunk | None, sigma_serve | None, prompt_override | None,
        phase 0/1/2). The chunk is the resampled polyline's per-step deltas in the flow's
        normalized action units (dims 0-3; other dims at the dataset mean, i.e. 0) — what the
        pin projects, and what an SDEdit-style baseline uses whole as its guide.

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
            i = st["i"]
            m = min(H, len(self.A) - i)
            if self.carrot > 0:
                off = pos - self.P[i, :3]
                w = np.maximum(0.0, 1.0 - np.arange(1, m + 1) / self.carrot)[:, None]
                pts = self.P[i + 1:i + 1 + m, :3] + off[None, :] * w
                seg = np.zeros((H, 7), np.float32)
                seg[:m, :3] = np.diff(np.concatenate([pos[None, :], pts]), axis=0)
                seg[:m, 3] = self.A[i:i + m, 3]
            else:
                seg = np.zeros((H, 7), np.float32)
                seg[:m] = self.A[i:i + m]
            ch = np.zeros((H, AD), np.float32)
            ch[:, :4] = (seg[:, :4] - self.amean[:4]) / (self.astd[:4] + 1e-6)
            return ch, self.sigma, self.prompt_after, 1
        if st["phase"] == 2:
            return None, None, self.prompt_after, 2
        return None, None, None, 0


