"""Is the gate policy timid because of the MODEL or only because of the SIM RENDER (domain gap)?
Feed gate_both_scratch REAL training observations (start frames + mid frames from data_gate_real) and
compare its predicted action chunk to the ground-truth action chunk. If the model reproduces the real
forward motion on real images, the sim's hovering is a render-fidelity gap; if it's timid on real images
too, the model itself is the bottleneck. Reports the x-forward column: GT vs predicted [first,last,span]
and net, plus overall action correlation."""
import json
import os
import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
B = os.path.expanduser("~/hf_bundle/gate-drone-pi0")
import sys
sys.path.insert(0, B)
from gate_inference import GatePolicy


def main():
    gp = GatePolicy(ckpt=os.path.join(B, "checkpoints/gate_both_scratch"),
                    norm_path=os.path.join(B, "assets/gate_nav"),
                    mode="scratch", bgr2rgb=False, wrist="separate")   # data already RGB in npz
    meta = json.load(open(os.path.join(RD, "data_gate_real", "meta.json")))
    keys = sorted(meta)[:8]
    print(f"{'ep':<10}{'t':>3} | {'GT x [first,last,span,net]':>34} | {'PRED x [first,last,span,net]':>34} | corr", flush=True)
    for k in keys:
        d = np.load(os.path.join(RD, "data_gate_real", k + ".npz"))
        T = len(d["action"])
        for t in [0, T // 3]:
            gt = d["action"][t:t + 50, :7].astype(np.float32)
            pred = gp.infer(d["image"][t], d["state"][t].astype(np.float32), meta[k]["lang"], wrist=d["wrist"][t])
            n = min(len(gt), len(pred)); gt, pred = gt[:n], pred[:n]
            gx, px = gt[:, 0], pred[:, 0]
            def stat(a): return f"[{a[0]:+.3f},{a[-1]:+.3f},{a.max()-a.min():.3f},{a[-1]-a[0]:+.3f}]"
            corr = float(np.corrcoef(gt.reshape(-1), pred.reshape(-1))[0, 1])
            print(f"{k:<10}{t:>3} | {stat(gx):>34} | {stat(px):>34} | {corr:+.2f}", flush=True)
    print("FWD_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
