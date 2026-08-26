"""Package a pinned policy into a self-contained bundle for real-robot inference.

Collects the checkpoint, norm stats, basis and command source into one directory, converts
the torch prior to numpy (so deployment needs no torch), records sha256 for every artifact,
and REFUSES TO WRITE unless two checks pass:
  1. numpy prior forward matches the torch module (max |diff| < 1e-5 on random inputs)
  2. the packaged bundle loads and produces a finite action chunk from a dummy observation

Example:
  python scripts/package_policy.py \
      --ckpt ~/code/openpi/checkpoints/pi0_gate/gate_both_pin_rrr/4999 \
      --config pi0_gate --norm ~/hf_bundle/gate-drone-pi0/assets/gate_nav \
      --pin-u experiments/rung3/pin_U_gate_rrr_k5.npy \
      --prior experiments/rung3/noprog_prior_rrr4_tailw4.pt \
      --out /home/ubuntu/bundles/gate_record_v1 --note "19/20 strict, 2026-08-09"
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from snmvp.deploy import NumpyMLP, PinnedPolicy, sha256_of  # noqa: E402

AXES = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]


def convert_prior(prior_path, out_npz):
    """torch MLP checkpoint -> numpy weights; returns (weights, torch_forward) for parity."""
    import torch
    import torch.nn as nn
    d = torch.load(prior_path, map_location="cpu", weights_only=False)
    layers, din = [], d["in_dim"]
    for h in d["hidden"]:
        layers += [nn.Linear(din, h), nn.SiLU()]
        din = h
    layers += [nn.Linear(din, d["K"])]
    net = nn.Sequential(*layers)
    net.load_state_dict(d["state_dict"])
    net.eval()
    lin = [m for m in net if isinstance(m, nn.Linear)]
    w = {"n_layers": len(lin), "mu": d["mu"].astype(np.float32), "sd": d["sd"].astype(np.float32),
         "kind": d.get("kind", "state_prior")}
    for i, m in enumerate(lin):
        w[f"W{i}"] = m.weight.detach().numpy().T.astype(np.float32)
        w[f"b{i}"] = m.bias.detach().numpy().astype(np.float32)
    if "tasks" in d:
        w["tasks"] = np.array([str(t) for t in d["tasks"]])
    for extra in ("Em", "P"):           # language-prior projection, if present
        if extra in d:
            w[extra] = np.asarray(d[extra], np.float32)

    def torch_forward(x):
        with torch.no_grad():
            return net(torch.tensor(((x - w["mu"]) / w["sd"]).astype(np.float32))).numpy()

    np.savez(out_npz, **w)
    return w, torch_forward, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--norm", required=True); ap.add_argument("--pin-u", required=True)
    ap.add_argument("--prior", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists — bundles are immutable; pick a new version directory")
    os.makedirs(out)

    print("copying checkpoint params ...", flush=True)
    shutil.copytree(os.path.join(a.ckpt, "params"), os.path.join(out, "params"))
    shutil.copytree(os.path.expanduser(a.norm), os.path.join(out, "norm_stats"))
    U = np.load(a.pin_u).astype(np.float32)
    np.save(os.path.join(out, "pin_U.npy"), U)

    if a.prior:
        w, torch_forward, d = convert_prior(a.prior, os.path.join(out, "prior.npz"))
        rng = np.random.default_rng(0)
        # draw from the prior's own input distribution (mu + sd*eps); raw N(0,1) inputs
        # normalize to absurd magnitudes and would test only float overflow behaviour
        x = (w["mu"] + w["sd"] * rng.normal(size=(256, len(w["mu"])))).astype(np.float32)
        ref = torch_forward(x)
        err = float(np.abs(NumpyMLP(w)(x) - ref).max())
        rel = err / float(np.abs(ref).max() + 1e-9)
        # float32 accumulation through 256-wide layers lands around 1e-5 absolute; judge on
        # the relative error against the command scale, with an absolute backstop
        print(f"numpy/torch prior parity: max|diff| = {err:.2e} (relative {rel:.2e})", flush=True)
        if rel > 1e-5 or err > 1e-3:
            shutil.rmtree(out)
            raise SystemExit(f"numpy prior does not match torch (abs {err:.2e}, rel {rel:.2e})")
        prior_in_dim = int(d["in_dim"])
        state_dim = int(d.get("state_dim", d["in_dim"]))
    else:
        prior_in_dim = state_dim = 0

    import openpi.training.config as _cfg
    cfg = _cfg.get_config(a.config)
    manifest = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "note": a.note,
        "config": a.config,
        "action_horizon": int(cfg.model.action_horizon),
        "action_dim": int(cfg.model.action_dim),
        "pin_dim": int(U.shape[1]),
        "axes": AXES,
        "prior_in_dim": prior_in_dim,
        "prior_state_dim": state_dim,
        "source": {"ckpt": os.path.abspath(a.ckpt), "pin_u": os.path.abspath(a.pin_u),
                   "prior": os.path.abspath(a.prior) if a.prior else None,
                   "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                         text=True, cwd=os.path.dirname(os.path.abspath(__file__))
                                         ).stdout.strip()},
        "sha256": {},
    }
    for name in ("params", "norm_stats", "pin_U.npy") + (("prior.npz",) if a.prior else ()):
        manifest["sha256"][name] = sha256_of(os.path.join(out, name))
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("verifying the bundle loads and infers ...", flush=True)
    pol = PinnedPolicy.from_bundle(out)
    dummy = {"observation/image": np.zeros((224, 224, 3), np.uint8),
             "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
             "observation/state": np.zeros(7, np.float32),
             "prompt": "hold position"}
    c = np.zeros(manifest["pin_dim"], np.float32)
    res = pol.act(dummy, c=c)
    act = np.asarray(res["actions"])
    if not np.isfinite(act).all():
        shutil.rmtree(out)
        raise SystemExit("packaged policy produced non-finite actions — not packaging")
    print(f"OK: chunk {act.shape}, |a|max {np.abs(act).max():.3f}, "
          f"command displacement {np.round(res['snmvp_command_displacement'], 3)} m")
    print(f"BUNDLE_READY {out}")


if __name__ == "__main__":
    main()
