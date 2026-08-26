"""Rebuild the language-prior feature cache in the SERVING convention.

`langprior_pipeline` embeds the stored demo frames; the server embeds live gsplat renders of the same
scene, and the two disagree enough that 25-42% of the prior's embedding inputs land beyond 3 sigma of
their own training distribution at the very first inference of an episode — before the drone has
moved. Training a command source on features it will never be shown is the defect, so this re-extracts
the cache from live renders at the demo states. No new demos are needed: the states are known.

Two stages, because the renderer and the policy live in different environments.

  # stage 1 (gsplat env), per scene — renders at every cached row's state
  CUDA_VISIBLE_DEVICES=1 /tmp/tv/bin/python langprior_relive.py --render --out /home/ubuntu/relive
  # stage 2 (openpi env) — embeds those frames, writes langprior_feats_live.npz
  CUDA_VISIBLE_DEVICES=1 openpi/.venv/bin/python langprior_relive.py --embed --frames /home/ubuntu/relive

Then rebuild the prior against the same basis the flow was trained with:
  UPATH=pin_U_gate_rrr_k5.npy CACHE=langprior_feats_live.npz OUT=langprior_live.pt \
    python langprior_rebasis.py
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

RD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RD)
STRIDE = 6                      # must match langprior_pipeline / langprior_rebasis
TASKS = ["go through the center gate from the left and hover over the stuffed animal",
         "go through the center gate from the right and hover over the stuffed animal",
         "go through the gate on the left and hover over the stuffed animal",
         "go through the gate on the right and hover over the stuffed animal"]
# ep0000-0049 centre-from-left, 0050-0099 centre-from-right, 0100-0149 left, 0150-0199 right.
# The centre tasks need the move_gate scene edit, so they render in a separate pass.
SCENE_OF = lambda ep: "center" if ep < 100 else ("left" if ep < 150 else "right")


def rows_of(ep):
    d = np.load(f"{RD}/data_gate_synth/ep_{ep:04d}.npz", allow_pickle=True)
    st = d["state"].astype(np.float32)
    return st, list(range(0, len(st) - 5, STRIDE))


def stage_render(a):
    """Live renders at every cached row's state, one npz per scene."""
    import render_skew_probe as RS      # left-scene splat + serving render path
    os.makedirs(a.out, exist_ok=True)
    todo = [e for e in range(200) if SCENE_OF(e) == a.scene]
    if a.scene != "left":
        print(f"[relive] scene={a.scene}: render_skew_probe carries the LEFT splat only; "
              f"{a.scene} needs its own checkpoint/edit before this pass is valid", flush=True)
        if a.scene != "left":
            raise SystemExit("refusing to write frames rendered from the wrong scene")
    F, EP, TT = [], [], []
    for n, e in enumerate(todo):
        st, ts = rows_of(e)
        for t in ts:
            img = RS.to(RS.to(RS.rend(st[t, :3], -float(st[t, 3])), 256, Image.BILINEAR), 224, Image.BICUBIC)
            F.append(img); EP.append(e); TT.append(t)
        if n % 5 == 0:
            print(f"  ep {e} rows {len(F)}", flush=True)
    np.savez_compressed(f"{a.out}/frames_{a.scene}.npz", img=np.stack(F),
                        ep=np.array(EP), t=np.array(TT))
    print(f"[relive] wrote {len(F)} frames for scene {a.scene}", flush=True)


def stage_embed(a):
    """Embed the rendered frames with the same pooling the server uses."""
    import openpi.policies.policy_config as PC
    import openpi.training.config as C
    import gate_ctx_common as gc
    policy = PC.create_trained_policy(C.get_config("pi0_gate"), a.ckpt, default_prompt="")
    E, S, ep_ix, frac = [], [], [], []
    for scene in a.scenes:
        f = f"{a.frames}/frames_{scene}.npz"
        if not os.path.exists(f):
            print(f"[relive] missing {f}; skipping {scene}", flush=True)
            continue
        Z = np.load(f)
        IMG, EP, TT = Z["img"], Z["ep"], Z["t"]
        for e in sorted(set(EP.tolist())):
            st, _ = rows_of(e)
            sel = np.where(EP == e)[0]
            raws = [{"observation/image": IMG[i], "observation/wrist_image": IMG[i],
                     "observation/state": st[TT[i]], "prompt": TASKS[e // 50]} for i in sel]
            for j in range(0, len(raws), 8):
                E.append(gc.lang_pool(policy, raws[j:j + 8]))
            for i, r in zip(sel, raws):
                S.append(np.asarray(policy._input_transform(dict(r))["state"]).reshape(-1))
                ep_ix.append(int(e)); frac.append(TT[i] / (len(st) - 1))
            if e % 20 == 0:
                print(f"  ep {e} rows {len(S)}", flush=True)
    out = f"{RD}/langprior_feats_live.npz"
    np.savez_compressed(out, E=np.concatenate(E, 0).astype(np.float32),
                        S=np.array(S, np.float32), ep=np.array(ep_ix, np.int64),
                        frac=np.array(frac, np.float32))
    print(f"[relive] wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--scene", default="left")
    ap.add_argument("--scenes", nargs="+", default=["left"])
    ap.add_argument("--out", default="/home/ubuntu/relive")
    ap.add_argument("--frames", default="/home/ubuntu/relive")
    ap.add_argument("--ckpt", default="/home/ubuntu/code/openpi/checkpoints/pi0_gate/gate_pin_zeropad/4999")
    a = ap.parse_args()
    if a.render:
        stage_render(a)
    elif a.embed:
        stage_embed(a)
    else:
        ap.error("pass --render or --embed")


if __name__ == "__main__":
    main()
