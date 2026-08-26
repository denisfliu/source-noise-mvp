"""Oracle-command server for the compound-task investigation (Denis, 2026-08-06).

Normal no-clock prior for every prompt EXCEPT --oracle-task; for that task the
command comes from an oracle instead of the learned prior:
  --oracle demo_nn   c = the true chunk-c of the NEAREST demo state of that task
                     (pure demonstration oracle — tests whether the flow completes
                     the compound leg given demo-faithful commands)
  --oracle waypoint  c = U^T segY(straight-line chunk toward the gate-2 aperture
                     center; once past the gate plane, toward the goal) — the
                     geometry-computed command (c_hold at a waypoint); no learning.
The VLM-generalization story: both oracles are information a scene-reading VLM
could supply (which demo family / which waypoint); success here bounds what a
generalizing command source buys.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

GATE2_CENTER = np.array([2.756, -0.3275, 1.51])  # aim z from DEMO transit altitude (demos cross
                                                 # at z~1.49-1.54; the safety-AABB mid z=1.0 is the
                                                 # POST span, ~0.5 m below the hoop — every 2026-08-05
                                                 # oracle run clipped the frame there)
GATE2_PLANE_Y = -0.3275
GOAL = np.array([1.525, -0.615, 1.0])
# right scene (safety right_gate.yaml post centres): diagonal aperture; normal points
# along the expected transit direction (dy sign -1)
RIGHT_CENTER = np.array([0.5595, -1.150, 1.49])  # demo transit altitude, same lesson
RIGHT_N = np.array([0.477, -0.878, 0.0]) / np.linalg.norm([0.477, -0.878, 0.0])


class OraclePinPolicy:
    def __init__(self, policy, pin_u, prior_path, oracle, oracle_task, geom="compound"):
        self.policy = policy
        self.U = np.load(pin_u).astype(np.float32)
        d = torch.load(prior_path, map_location="cpu", weights_only=False)
        layers, din = [], d["in_dim"]
        for h in d["hidden"]:
            layers += [nn.Linear(din, h), nn.SiLU()]; din = h
        layers += [nn.Linear(din, 5)]
        self.prior = nn.Sequential(*layers); self.prior.load_state_dict(d["state_dict"]); self.prior.eval()
        self.mu, self.sd, self.tasks = d["mu"], d["sd"], list(d["tasks"])
        self.oracle, self.oracle_task, self.geom = oracle, oracle_task, geom
        self.H, self.AD = gc.H, gc.AD
        self._rng = np.random.default_rng()
        ns, self.amean, self.astd = gc.load_norm()
        if oracle == "demo_nn":
            eps = gc.load_eps(with_images=False)
            sel = [e for e in eps if e["lang"] == oracle_task]
            self.DS = np.concatenate([e["state"][:, :3] for e in sel])
            cs = []
            for e in sel:
                for t in range(len(e["state"])):
                    tt = min(t, len(e["action"]) - 1)
                    cs.append(gc.segY(e["action"][tt:], self.amean, self.astd) @ self.U)
            self.DC = np.stack(cs).astype(np.float32)
        print(f"[oracle] mode={oracle} task={oracle_task[:40]!r} tasks={len(self.tasks)}", flush=True)

    def _prior_c(self, obs, prompt):
        ms = np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
        v = np.zeros(len(self.tasks), np.float32)
        if prompt not in self.tasks:
            raise ValueError(f"unknown prompt {prompt!r}")
        v[self.tasks.index(prompt)] = 1.0
        x = np.concatenate([ms, v]).astype(np.float32)
        with torch.no_grad():
            return self.prior(torch.tensor(((x - self.mu) / self.sd)[None]))[0].numpy().astype(np.float32)

    def _oracle_c(self, pos):
        if self.oracle == "demo_nn":
            i = np.linalg.norm(self.DS - pos, axis=1).argmin()
            return self.DC[i]
        if self.geom == "right":
            # right scene, same three ingredients: carry-through beyond the (diagonal)
            # plane, an east clearance waypoint on the way back, then the goal
            d = (pos - RIGHT_CENTER) @ RIGHT_N
            if d > 0.25:
                target = np.array([1.5, -1.45, 1.2]) if pos[0] < 1.3 else GOAL
            else:
                target = RIGHT_CENTER + 0.40 * RIGHT_N
            return self._chunk_toward(target, pos)
        # waypoint: straight-line chunk toward a point BEYOND gate-2's plane (carry-through:
        # aiming at the center itself decays the command to zero at the aperture mouth and
        # the target flip-flops across the plane — threshold stall, 0/5 on 2026-08-05), then
        # the goal once decisively through. Stateless: past the through-zone the goal is
        # further -y, so no reversal.
        if pos[1] < GATE2_PLANE_Y - 0.20:
            target = GOAL
        elif pos[0] < 1.9:
            # clear east of gate-1's aperture AABB (x<=1.18) before turning south —
            # a straight line to gate 2 from the gate-1 exit recrosses gate-1's plane
            # wrong-direction while still inside the aperture slab (or6, 2026-08-05:
            # wrong=1, gate-1 latch lost); demos loop around the post the same way
            target = np.array([2.05, 0.85, 1.45])
        else:
            target = GATE2_CENTER + np.array([0.0, -0.40, 0.0])
        return self._chunk_toward(target, pos)

    def _chunk_toward(self, target, pos):
        delta = target - pos
        dist = np.linalg.norm(delta)
        net = delta / (dist + 1e-9) * min(dist, 1.0)     # cap chunk reach at 1 m
        chunk = np.zeros((self.H, 7), np.float32)
        chunk[:, :3] = net / self.H
        return (gc.segY(chunk, self.amean, self.astd) @ self.U).astype(np.float32)

    def infer(self, obs):
        prompt = str(obs.get("prompt", ""))
        pos = np.asarray(obs["observation/state"], np.float32)[:3]
        c = self._oracle_c(pos) if prompt == self.oracle_task else self._prior_c(obs, prompt)
        g = self._rng.standard_normal((self.H, self.AD)).astype(np.float32).reshape(-1)
        noise = (g - (g @ self.U) @ self.U.T + (c @ self.U.T)).reshape(self.H, self.AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--norm", required=True)
    ap.add_argument("--pin-u", required=True); ap.add_argument("--prior", required=True)
    ap.add_argument("--oracle", required=True, choices=["demo_nn", "waypoint"])
    ap.add_argument("--oracle-task", default=gc.PROMPT_CFL)
    ap.add_argument("--geom", default="compound", choices=["compound", "right"])
    ap.add_argument("--config", default="pi0_gate"); ap.add_argument("--port", type=int, default=8805)
    a = ap.parse_args()
    cfg = _cfg.get_config(a.config)
    ns = gc.pad_norm_stats(_nz.load(a.norm), cfg.model.action_dim)
    policy = _pc.create_trained_policy(cfg, a.ckpt, norm_stats=ns)
    pin = OraclePinPolicy(policy, a.pin_u, a.prior, a.oracle, a.oracle_task, a.geom)
    print(f"[serve_gate_pin_oracle] ready on ws://127.0.0.1:{a.port}", flush=True)
    WebsocketPolicyServer(pin, host="127.0.0.1", port=a.port).serve_forever()


if __name__ == "__main__":
    main()
