"""Paper-grade anatomy of the RRR pin basis U: temporal-mode decomposition per
action dimension, principal angles vs the pure net-displacement subspace, task-axis
loading, and predictable-variance capture vs PCA. CPU."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gate_ctx_common as gc
import gate_traj_algebra as ta

ns, amean, astd = gc.load_norm()
U = np.load(os.path.join(gc.RD, "pin_U_gate_rrr_k5.npy"))
H, AD = gc.H, gc.AD
DIMS = ["x", "y", "z", "yaw"]

def dct_basis(H, n):
    t = np.arange(H)
    B = [np.ones(H) / np.sqrt(H)]
    for k in range(1, n):
        v = np.cos(np.pi * k * (t + 0.5) / H)
        B.append(v / np.linalg.norm(v))
    return np.stack(B)

B = dct_basis(H, 6)
print("=== energy decomposition per pin coordinate (percent of column energy) ===")
for j in range(5):
    col = U[:, j].reshape(H, AD)
    tot = (col ** 2).sum()
    print(f"--- c_{j}")
    for d in range(4):
        coeffs = B @ col[:, d]
        e = [100 * c ** 2 / tot for c in coeffs]
        dt = 100 * (col[:, d] ** 2).sum() / tot
        if dt > 1.0:
            print("   %-3s: net=%5.1f%%  ramp=%5.1f%%  curv=%4.1f%%  hi-freq=%4.1f%%   dim-total=%5.1f%%"
                  % (DIMS[d], e[0], e[1], e[2], dt - sum(e[:3]), dt))
    pad = 100 * (col[:, 4:] ** 2).sum() / tot
    if pad > 0.5:
        print("   padded dims: %.1f%%" % pad)

D = np.zeros((H * AD, 4))
for d in range(4):
    v = np.zeros((H, AD)); v[:, d] = 1 / np.sqrt(H)
    D[:, d] = v.reshape(-1)
s = np.linalg.svd(D.T @ U, compute_uv=False)
print("=== principal angles span(U) vs net-displacement subspace [deg]:",
      np.round(np.degrees(np.arccos(np.clip(s, 0, 1))), 1))

eps = gc.load_eps(with_images=False)
C0 = np.stack([gc.segY(e["action"], amean, astd) @ U for e in eps])
CR = np.stack([gc.segY(ta.reverse(e)["action"], amean, astd) @ U for e in eps])
langs = np.array([e["lang"] for e in eps])
print("fwd/back axis per coord:", np.round((C0 - CR).mean(0), 2))
print("L/R axis per coord:     ",
      np.round(C0[langs == gc.PROMPT_L].mean(0) - C0[langs == gc.PROMPT_R].mean(0), 2))

Y = np.stack([gc.segY(e["action"][t:], amean, astd)
              for e in eps for t in range(0, len(e["action"]), 24)])
Yc = Y - Y.mean(0)
tot = (Yc ** 2).sum()
capU = ((Yc @ U) ** 2).sum() / tot
_, S, Vt = np.linalg.svd(Yc, full_matrices=False)
capPCA = (S[:5] ** 2).sum() / (S ** 2).sum()
print("chunk-variance captured: span(U) %.3f vs top-5 PCA %.3f (n=%d chunks)" % (capU, capPCA, len(Y)))
print("U_ANATOMY_DONE")
