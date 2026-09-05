"""AdvicePrompt (2026-09-03, Denis: "you are going through the first gate but not the second —
a more automatic correction than a hand-drawn route"). Minimal command-space advice for the
compound task: no drawn path, just TARGET POINTS. Once the drone has transited gate 1
(observational trigger: plane crossing inside the published aperture; a stand-in for the human
saying "now"), each replan's command is a straight-line pursuit toward the current target at
demo pace, and only the command coordinates named by `mode` are overridden — the head keeps
the rest. When the last target is reached the head takes over under `prompt_after`.

json: {"targets": [[x,y,z], ...], "mode": "all"|"coarse_xy"|"h50_xy"|"none",
       "trigger": "none" (default; active from the first replan) | "gate1" (legacy: plane
       crossing of "gate1_corners" — uses scene geometry, kept only for the 2026-09-03 cells),
       "swap_prompt": false (default: the task prompt is never touched) | true (prompt_after
       once the last target is reached), "prompt_after": str,
       "idle_gate": 0.0 (default: always override) | f>0 (override only while the head's own
       coarsest x,y displacement is below f x the pursuit's — "fill in when the model is parking"),
       "reach_radius": 0.35, "step_m": 0.025}
Targets are consumed in order by distance from the drone; nothing else is observed.

modes (coordinate index = 4*band + axis; bands = steps 0-6, 6-12, 12-25, 25-50; axes x,y,z,yaw):
  all        every coordinate from the pursuit chunk (== a 2-click sketch with carrot)
  coarse_xy  bands 2-3 of x and y only: {8, 9, 12, 13}   (fine horizons, z, yaw stay the head's)
  h50_xy     band 3 of x and y only:    {12, 13}          (the single coarsest displacement word)
  none       prompt swap only — the language control
The same object serves an SDEdit baseline through window(): the pursuit chunk is its guide.
"""
import json

import numpy as np

H, AD = 50, 32
MODES = {"all": list(range(16)), "coarse_xy": [8, 9, 12, 13], "h50_xy": [12, 13], "none": []}


class AdvicePrompt:
    def __init__(self, path, amean, astd, U):
        d = json.load(open(path))
        self.trigger = d.get("trigger", "none")
        if self.trigger == "gate1":
            C = np.asarray(d["gate1_corners"], np.float64)
            u = C[1] - C[0]; self.W = float(np.linalg.norm(u)); self.u = u / self.W
            v = C[3] - C[0]; self.Hh = float(np.linalg.norm(v)); self.v = v / self.Hh
            self.n = np.cross(self.u, self.v); self.n /= np.linalg.norm(self.n); self.c0 = C[0]
        self.swap_prompt = bool(d.get("swap_prompt", False))
        self.idle_gate = float(d.get("idle_gate", 0.0))
        self.targets = np.asarray(d["targets"], np.float32)
        self.mode = d.get("mode", "coarse_xy")
        self.dims = MODES[self.mode]
        self.prompt_after = d.get("prompt_after")
        self.reach = float(d.get("reach_radius", 0.35))
        self.step = float(d.get("step_m", 0.025))
        self.amean, self.astd, self.U = amean, astd, U
        self._st = {}
        print(f"[advice] mode={self.mode} dims={self.dims} trigger={self.trigger} swap_prompt={self.swap_prompt} "
              f"idle_gate={self.idle_gate} targets={self.targets.round(2).tolist()}", flush=True)

    def _crossed_gate1(self, prev, pos):
        s0, s1 = float((prev - self.c0) @ self.n), float((pos - self.c0) @ self.n)
        if s0 * s1 >= 0:
            return False
        x = prev + (s0 / (s0 - s1)) * (pos - prev)
        a, b = float((x - self.c0) @ self.u), float((x - self.c0) @ self.v)
        return 0 <= a <= self.W and 0 <= b <= self.Hh

    def window(self, trial, pos):
        """-> (normalized (H, AD) pursuit chunk | None, sigma | None, prompt | None, phase 0/1/2)."""
        pos = np.asarray(pos, np.float32)
        st = self._st.setdefault(trial, {"phase": 0 if self.trigger == "gate1" else 1, "i": 0, "prev": None})
        if st["phase"] == 0:
            if st["prev"] is not None and self._crossed_gate1(st["prev"].astype(np.float64), pos.astype(np.float64)):
                st["phase"] = 1
            st["prev"] = pos.copy()
        if st["phase"] == 1:
            while st["i"] < len(self.targets) and np.linalg.norm(pos - self.targets[st["i"]]) < self.reach:
                st["i"] += 1
            if st["i"] >= len(self.targets):
                st["phase"] = 2
        if st["phase"] == 1:
            # straight-line pursuit at demo pace through the remaining targets
            pts, p = [], pos.copy()
            for t in self.targets[st["i"]:]:
                d = t - p; L = float(np.linalg.norm(d)); k = max(1, int(np.ceil(L / self.step)))
                pts.extend(p + d * (j / k) for j in range(1, k + 1)); p = t
            pts = np.asarray(pts[:H], np.float32)
            seg = np.zeros((H, 7), np.float32)
            seg[:len(pts), :3] = np.diff(np.concatenate([pos[None], pts]), axis=0)
            ch = np.zeros((H, AD), np.float32)
            ch[:, :4] = (seg[:, :4] - self.amean[:4]) / (self.astd[:4] + 1e-6)
            return ch, 0.0, (self.prompt_after if self.trigger == "gate1" else None), 1
        if st["phase"] == 2:
            return None, None, (self.prompt_after if (self.swap_prompt or self.trigger == "gate1") else None), 2
        return None, None, None, 0

    def compose(self, c_head, ch):
        """Override only this mode's coordinates of the head's command with the pursuit's."""
        c = np.asarray(c_head, np.float32).copy()
        if not self.dims:
            return c
        cp = ch.reshape(-1) @ self.U
        if self.idle_gate > 0:
            # fill in only when the head has no coarse plan of its own (it is parking)
            if np.linalg.norm(c[[12, 13]]) >= self.idle_gate * np.linalg.norm(cp[[12, 13]]):
                return c
        c[self.dims] = cp[self.dims]
        return c
