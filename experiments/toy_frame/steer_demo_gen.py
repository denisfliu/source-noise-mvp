"""Regenerate the steering demo from VERIFIED models (corrects the earlier
Jacobian-artifact demo). Trains condition/pin/csfm (steer_probe), then for one
scene dumps a clean 5x5 trajectory grid over two interpretable knobs (bend =
lateral bin-1 Im, S-curve = lateral bin-2 Im) in the CANONICAL frame (progress
-> +x, bend vertical) plus honest direct-sweep metrics: steering slope, phase
follow, magnitude tracking, cross-leak.
"""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import steer_probe as sp
from dataset import H, ACT_SCALE, make_dataset, to_canonical

def canon_pos(chunk, ang):
    c = to_canonical(chunk, ang)
    return np.concatenate([np.zeros((chunk.shape[0], 1, 2)),
                           np.cumsum(c / ACT_SCALE, axis=1)], axis=1)

def main():
    sc, obs, ch, ang = make_dataset(200, 8, np.random.default_rng(7))
    fo, fc, fa = np.repeat(obs, 8, 0), ch.reshape(-1, H, 2), np.repeat(ang, 8)
    he_sc, he_o, he_ch, he_a = make_dataset(60, 8, np.random.default_rng(7777))
    C_nat = sp.coeffs(to_canonical(he_ch, he_a[:, None])).mean(1)
    scale = np.abs(C_nat).mean(0) + 1e-6
    S = 6                                   # scene index for the visual
    scene = he_sc[S]
    canon_scene = {"radius": float(scene["radius"]), "s_o": float(scene["s_o"]),
                   "lateral": float(scene["lateral"]), "obst_r": float(scene["obst_r"])}
    knob = np.linspace(-2.5, 2.5, 5)
    out = {"scene": canon_scene, "knob": knob.round(2).tolist(), "arms": {}}

    for arm in ["condition", "pin", "csfm"]:
        p = sp.train(arm, fo, fc, fa)
        er = np.random.default_rng(0)
        # clean pure-Im1 sweep: slope + cross-leak
        cmd_v, prod_v, leak_v = [], [], []
        for a in np.linspace(-2.5, 2.5, 9):
            C = C_nat.copy(); C[:, 0] = 0.0; C[:, 1] = a
            pc = sp.prod_coeffs(sp.integrate(arm, p, he_o, he_a, C, er), he_a)
            cmd_v.append(a); prod_v.append(float(pc[:, 1].mean())); leak_v.append(float(abs(pc[:, 0]).mean()))
        slope = float(np.polyfit(cmd_v, prod_v, 1)[0])
        crossleak = float(np.mean(leak_v))
        # phase follow (deg) at natural radius
        r = scale[1]; pe = []
        for th in np.linspace(0, 2*np.pi, 8, endpoint=False):
            C = C_nat.copy(); C[:, 0] = r*np.cos(th); C[:, 1] = r*np.sin(th)
            pc = sp.prod_coeffs(sp.integrate(arm, p, he_o, he_a, C, er), he_a)
            pe.append(float(sp.circdist(np.arctan2(pc[:, 1], pc[:, 0]), th).mean()))
        phase_deg = float(np.degrees(np.mean(pe)))
        # magnitude tracking at 2x
        C = C_nat.copy(); C[:, 0] = 2*r; C[:, 1] = 0.0
        pc = sp.prod_coeffs(sp.integrate(arm, p, he_o, he_a, C, er), he_a)
        mag2x = float(np.sqrt(pc[:, 0]**2 + pc[:, 1]**2).mean() / (2*r))

        # 5x5 trajectory grid for scene S, canonical frame
        oS = he_o[S:S+1]; aS = he_a[S:S+1]; base = C_nat[S:S+1].copy()
        grid = []
        for v1 in knob:
            row = []
            for v2 in knob:
                C = base.copy(); C[0, 1] = v1; C[0, 3] = v2
                chk = sp.integrate(arm, p, oS, aS, C, np.random.default_rng(0))
                pos = canon_pos(chk, aS)[0]
                row.append([[round(float(x), 3), round(float(y), 3)] for x, y in pos])
            grid.append(row)
        out["arms"][arm] = {"grid": grid, "slope": round(slope, 3),
                            "crossleak": round(crossleak, 3), "phase_deg": round(phase_deg, 1),
                            "mag2x": round(mag2x, 3)}
        print(arm, "slope", round(slope, 3), "phase", round(phase_deg, 1),
              "mag2x", round(mag2x, 3), "crossleak", round(crossleak, 3), flush=True)

    json.dump(out, open(os.path.join(HERE, "results", "steer_demo2.json"), "w"))
    print("DONE")

if __name__ == "__main__":
    main()
