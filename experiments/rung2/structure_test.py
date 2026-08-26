"""Does DISCOVERED structure help a policy-bottlenecked policy on a HARD task at
real (robosuite) scale? The scaled analog of the toy_frame +17-pt learned-frame
result — the one open test of original goal (b).

Task: robosuite obstacle-reach (real arm kinematics; a straight reach fails, the
policy must learn the detour). Single embodiment (Panda). We compare, held-out:
  A     scratch flow policy (no structure)
  F     coherence-discovered structure pinned into source noise + learned prior
  Frand random-frame pin + its prior (control: is it the LEARNED structure?)
Hypothesis (goal b, where the POLICY is the bottleneck): F > A and F > Frand.

Reuses the toy_frame/toy_embodiment flow+coherence+pin machinery. Trick: treat
each DEMO as a "body" so flow_embod.freeze_frame's cross-body coherence becomes
coherence-over-demos (the single-embodiment structure = the shared detour shape).
Success (offline, on the generated chunk) = reach target within TOL AND every
point clears the obstacle disk — the real obstacle-avoidance metric.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "toy_embodiment"))
sys.path.insert(0, os.path.join(HERE, "..", "toy_frame"))
import flow_embod as fe                # noqa: E402
from pin import extract_mags           # noqa: E402
import transfer_smoke as sm            # noqa: E402  (_feats_from_targets)

ARM = os.environ.get("SNMVP_ARM", "Panda")
DATA = os.path.join(HERE, "data_obst", f"{ARM}.npz")
H = 32
TOL = 0.03
fe.H = H
fe.OBS_DIM = 4                          # [radius, obst_canon_x, obst_canon_y, obst_r]
N_HELD = 20
ITERS = 8000
N_ROLL = 8
SEEDS = [0, 1, 2]


def canonicalize(chunks, obs, obst_r):
    """chunks (S,N,H,2) global start-rel; obs (S,4)=[disp_xy,obst_xy]. Returns
    canonical scene descriptor (S,4)=[radius,obst_cx,obst_cy,obst_r], reach
    angles (S,), and the global chunks unchanged (executor rotates internally)."""
    disp, obst = obs[:, :2], obs[:, 2:4]
    ang = np.arctan2(disp[:, 1], disp[:, 0])
    rad = np.linalg.norm(disp, axis=1)
    c, s = np.cos(-ang), np.sin(-ang)                # rotate obst into canonical
    ox = c * obst[:, 0] - s * obst[:, 1]
    oy = s * obst[:, 0] + c * obst[:, 1]
    scene = np.stack([rad, ox, oy, obst_r], axis=1)
    return scene, ang


def obst_success(gen_chunks, disp, obst, obst_r, tol):
    """gen (R,M,H,2) start-rel scaled -> per-(roll,scene) reach+clearance."""
    R, M = gen_chunks.shape[:2]
    pos = np.cumsum(gen_chunks, axis=2)              # (R,M,H,2)
    ok = np.zeros((R, M))
    for r in range(R):
        for m in range(M):
            end_ok = np.linalg.norm(pos[r, m, -1] - disp[m]) < tol
            clear = (np.linalg.norm(pos[r, m] - obst[m], axis=1) > obst_r[m]).all()
            ok[r, m] = float(end_ok and clear)
    return float(ok.mean())


def main():
    d = np.load(DATA)
    chunks, obs, obst_r, succ = (d["chunks"].astype(float), d["obs"], d["obst_r"], d["success"])
    S, N = chunks.shape[:2]
    print(f"{ARM}: {S} scenes x {N} demos; demo detour-success ceiling {succ.mean():.3f}")
    scene, ang = canonicalize(chunks, obs, obst_r)
    disp, obstxy = obs[:, :2], obs[:, 2:4]

    # scale actions to O(1) (raw ~mm/step underfits) — scale chunks, disp, obst, tol
    SCALE = 1.0 / np.abs(chunks).mean()
    chunks_s = chunks * SCALE
    disp_s, obst_s, obr_s, tol_s = disp * SCALE, obstxy * SCALE, obst_r * SCALE, TOL * SCALE
    scene_s = scene.copy(); scene_s[:, 0] *= SCALE; scene_s[:, 1:3] *= SCALE; scene_s[:, 3] *= SCALE
    print(f"ACT_SCALE={SCALE:.1f} scaled tol={tol_s:.3f} mean|disp|={np.linalg.norm(disp_s,axis=1).mean():.2f}")

    # split scenes
    rng = np.random.default_rng(0)
    perm = rng.permutation(S)
    tr, he = perm[N_HELD:], perm[:N_HELD]

    # demos-as-bodies so freeze_frame/build_shared_prior operate OVER DEMOS
    bodies = {f"d{i}": chunks_s[tr][:, i:i+1] for i in range(N)}   # each (|tr|,1,H,2)
    set_a = list(bodies.keys())
    ang_tr = ang[tr]
    S_F, diag = fe.freeze_frame(bodies, ang_tr, set_a=set_a)
    print(f"discovered frame S_F: {len(S_F)} pins",
          [(tuple(np.round(p['axis'],1)), p['omega'], p['mode'], p.get('mag')) for p in S_F])

    A_pool = np.concatenate([fe.tfd.to_canonical(bodies[b], ang_tr[:, None]) for b in set_a],
                            axis=1).reshape(-1, H, 2)
    mag_norm = extract_mags(A_pool, S_F).mean(axis=0) if S_F else np.array([])

    # flatten train demos: obs (scene descriptor) repeated per demo
    def flat(idx):
        ch = chunks_s[idx].reshape(-1, H, 2)
        ob = np.repeat(scene_s[idx], N, axis=0)
        an = np.repeat(ang[idx], N)
        return ob, ch, an
    ob_tr, ch_tr, an_tr = flat(tr)

    # held-out eval scene tensors (scaled, global)
    he_scene, he_ang = scene_s[he], ang[he]
    he_disp, he_obst, he_obr = disp_s[he], obst_s[he], obr_s[he]

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
            return obst_success(ch.reshape(N_ROLL, M, H, 2), he_disp, he_obst, he_obr, tol_s)

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
               "note": "F>A => discovered structure helps the bottlenecked policy "
                       "on the hard task (scaled analog of toy +17pt); F>Frand => "
                       "it is the LEARNED structure, not any pin."}
    out = {"per_seed": results, "verdict": verdict}
    json.dump(out, open(os.path.join(HERE, "structure_result.json"), "w"), indent=2)
    print("VERDICT:", json.dumps(verdict, indent=2))
    print("STRUCTURE_TEST_DONE=ok")


if __name__ == "__main__":
    main()
