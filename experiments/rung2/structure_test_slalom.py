"""Does DISCOVERED structure help the bottlenecked policy on the HARDER slalom
task (two obstacles, S-curve detour)? Same methodology as structure_test.py:
  A     scratch flow policy (no structure)
  F     coherence-discovered structure pinned into source noise + learned prior
  Frand random-frame pin + its prior (control)
Hypothesis: F > A and F > Frand, replicating the single-obstacle result on a task
whose discoverable structure is richer (endpoint + two opposite lateral bends).

Success (offline, on the generated chunk) = reach target within TOL AND every
point clears BOTH obstacle disks.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
sys.path.insert(0, os.path.join(HERE, "..", "toy_frame"))
import flow_embod as fe                # noqa: E402
from pin import extract_mags           # noqa: E402

ARM = os.environ.get("SNMVP_ARM", "Panda")
DATA = os.path.join(HERE, "data_slalom", f"{ARM}.npz")
H = 32
TOL = 0.03
fe.H = H
fe.OBS_DIM = 7                          # [radius, o1_cx,o1_cy,o1_r, o2_cx,o2_cy,o2_r]
N_HELD = 20
ITERS = 8000
N_ROLL = 8
SEEDS = [0, 1, 2]


def canonicalize(obs, obst_r):
    """obs (S,6)=[disp_xy,o1_xy,o2_xy]; obst_r (S,2). Returns scene descriptor
    (S,7)=[radius, o1_cx,o1_cy,o1_r, o2_cx,o2_cy,o2_r] and reach angles (S,)."""
    disp, o1, o2 = obs[:, :2], obs[:, 2:4], obs[:, 4:6]
    ang = np.arctan2(disp[:, 1], disp[:, 0])
    rad = np.linalg.norm(disp, axis=1)
    c, s = np.cos(-ang), np.sin(-ang)
    def rot(o):
        return c * o[:, 0] - s * o[:, 1], s * o[:, 0] + c * o[:, 1]
    o1x, o1y = rot(o1); o2x, o2y = rot(o2)
    scene = np.stack([rad, o1x, o1y, obst_r[:, 0], o2x, o2y, obst_r[:, 1]], axis=1)
    return scene, ang


def slalom_success(gen_chunks, disp, o1, o2, r1, r2, tol):
    """gen (R,M,H,2) start-rel scaled -> per-(roll,scene) reach + clear both."""
    R, M = gen_chunks.shape[:2]
    pos = np.cumsum(gen_chunks, axis=2)
    ok = np.zeros((R, M))
    for r in range(R):
        for m in range(M):
            end_ok = np.linalg.norm(pos[r, m, -1] - disp[m]) < tol
            c1 = (np.linalg.norm(pos[r, m] - o1[m], axis=1) > r1[m]).all()
            c2 = (np.linalg.norm(pos[r, m] - o2[m], axis=1) > r2[m]).all()
            ok[r, m] = float(end_ok and c1 and c2)
    return float(ok.mean())


def main():
    d = np.load(DATA)
    chunks, obs, obst_r, succ = (d["chunks"].astype(float), d["obs"], d["obst_r"], d["success"])
    S, N = chunks.shape[:2]
    print(f"{ARM}: {S} scenes x {N} demos; demo slalom-success ceiling {succ.mean():.3f}")
    scene, ang = canonicalize(obs, obst_r)
    disp = obs[:, :2]; o1xy = obs[:, 2:4]; o2xy = obs[:, 4:6]

    SCALE = 1.0 / np.abs(chunks).mean()
    chunks_s = chunks * SCALE
    disp_s, o1_s, o2_s = disp * SCALE, o1xy * SCALE, o2xy * SCALE
    r1_s, r2_s = obst_r[:, 0] * SCALE, obst_r[:, 1] * SCALE
    tol_s = TOL * SCALE
    scene_s = scene.copy()
    scene_s[:, [0, 1, 2, 3, 4, 5, 6]] *= SCALE           # all columns are lengths
    print(f"ACT_SCALE={SCALE:.1f} scaled tol={tol_s:.3f} mean|disp|={np.linalg.norm(disp_s,axis=1).mean():.2f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(S)
    tr, he = perm[N_HELD:], perm[:N_HELD]

    bodies = {f"d{i}": chunks_s[tr][:, i:i+1] for i in range(N)}
    set_a = list(bodies.keys())
    ang_tr = ang[tr]
    S_F, diag = fe.freeze_frame(bodies, ang_tr, set_a=set_a)
    print(f"discovered frame S_F: {len(S_F)} pins",
          [(tuple(np.round(p['axis'],1)), p['omega'], p['mode'], p.get('mag')) for p in S_F])
    print("lat gamma/gamma2:", diag["lat"]["gamma"], "/", diag["lat"]["gamma2"])
    print("lat energy_frac:", diag["lat"]["energy_frac"])

    A_pool = np.concatenate([fe.tfd.to_canonical(bodies[b], ang_tr[:, None]) for b in set_a],
                            axis=1).reshape(-1, H, 2)
    mag_norm = extract_mags(A_pool, S_F).mean(axis=0) if S_F else np.array([])

    def flat(idx):
        ch = chunks_s[idx].reshape(-1, H, 2)
        ob = np.repeat(scene_s[idx], N, axis=0)
        an = np.repeat(ang[idx], N)
        return ob, ch, an
    ob_tr, ch_tr, an_tr = flat(tr)

    he_scene, he_ang = scene_s[he], ang[he]
    he_disp, he_o1, he_o2 = disp_s[he], o1_s[he], o2_s[he]
    he_r1, he_r2 = r1_s[he], r2_s[he]

    results = {}
    for seed in SEEDS:
        prior = fe.build_shared_prior(bodies, scene_s[tr], ang_tr, set_a, S_F, 100 + seed)
        rand = fe.rand_frame(seed)
        prior_r = fe.build_shared_prior(bodies, scene_s[tr], ang_tr, set_a, rand, 200 + seed)
        pA = fe.train_executor(ob_tr, ch_tr, an_tr, [], None, mag_norm, seed, ITERS)
        pF = fe.train_executor(ob_tr, ch_tr, an_tr, S_F, None, mag_norm, seed, ITERS)
        pR = fe.train_executor(ob_tr, ch_tr, an_tr, rand, None, mag_norm, seed, ITERS)
        pt, pm, _ = fe.prior_predict(prior, he_scene, S_F)
        ptr, pmr, _ = fe.prior_predict(prior_r, he_scene, rand)

        def ev(params, pins, tgt, mg):
            r = np.random.default_rng(500 + seed)
            M = he_scene.shape[0]
            obs_r = np.tile(he_scene, (N_ROLL, 1)); ang_r = np.tile(he_ang, N_ROLL)
            tg = np.tile(tgt, (N_ROLL, 1)) if tgt is not None else None
            mgt = np.tile(mg, (N_ROLL, 1)) if mg is not None else None
            ch = fe.rollout(params, obs_r, ang_r, pins, tg, None, r, mag_targets=mgt)
            return slalom_success(ch.reshape(N_ROLL, M, H, 2), he_disp, he_o1, he_o2,
                                  he_r1, he_r2, tol_s)

        row = {"A": ev(pA, [], None, None),
               "F": ev(pF, S_F, pt, pm),
               "Frand": ev(pR, rand, ptr, pmr)}
        results[f"s{seed}"] = {k: round(v, 3) for k, v in row.items()}
        print(f"seed{seed}: {results[f's{seed}']}", flush=True)

    pooled = {k: round(float(np.mean([results[f's{s}'][k] for s in SEEDS])), 3)
              for k in ["A", "F", "Frand"]}
    verdict = {"pooled": pooled, "F_minus_A": round(pooled["F"] - pooled["A"], 3),
               "F_minus_Frand": round(pooled["F"] - pooled["Frand"], 3),
               "demo_ceiling": round(float(succ.mean()), 3), "n_pins": len(S_F),
               "task": "two-obstacle slalom (S-curve detour)"}
    out = {"per_seed": results, "verdict": verdict}
    json.dump(out, open(os.path.join(HERE, "slalom_result.json"), "w"), indent=2)
    print("VERDICT:", json.dumps(verdict, indent=2))
    print("STRUCTURE_TEST_SLALOM_DONE=ok")


if __name__ == "__main__":
    main()
