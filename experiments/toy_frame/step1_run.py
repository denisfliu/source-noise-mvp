"""Step 1 runner: synthetic-recovery sanity, then heatmaps + G1 on the real
toy_frame dataset. CPU-only, no flow training. Outputs JSON + ASCII heatmaps
to results/step1/."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from coherence import ascii_heatmap, coherence_maps, select_structure  # noqa: E402
from dataset import H, make_dataset, to_canonical  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results" / "step1"
THETAS = np.radians(np.arange(0, 180, 2.0))
RNG = np.random.default_rng(7)


def synthetic_recovery():
    """Plant known (u, S) in random fields; the estimator must find them.
    Case A: shared phase (mod 2pi) at bins {1,3} along 37 deg.
    Case B: shared phase UP TO SIGN at bin 2 along 120 deg (gamma2 only)."""
    M, N = 100, 8
    B = H // 2 + 1
    verdicts = {}
    for case, (theta_true, bins, flip) in {
        "A_mod2pi": (np.radians(37.0), [1, 3], False),
        "B_modpi": (np.radians(120.0), [2], True),
    }.items():
        u = np.array([np.cos(theta_true), np.sin(theta_true)])
        v = np.array([-u[1], u[0]])
        chunks = np.zeros((M, N, H, 2))
        for m in range(M):
            shared_phase = RNG.uniform(0, 2 * np.pi, size=len(bins))
            for n in range(N):
                spec = (RNG.normal(size=B) + 1j * RNG.normal(size=B))
                for k, b in enumerate(bins):
                    mag = 3.0 + RNG.uniform(0, 1)
                    ph = shared_phase[k] + (np.pi * RNG.integers(0, 2) if flip else 0)
                    spec[b] = mag * np.exp(1j * ph)
                z = np.fft.irfft(spec, n=H)
                w = np.fft.irfft(RNG.normal(size=B) + 1j * RNG.normal(size=B), n=H)
                chunks[m, n] = np.outer(z, u) + np.outer(w, v)
        g, g2 = coherence_maps(chunks, THETAS)
        grid = g if not flip else np.nan_to_num(g2)
        # recovered direction at the planted bins
        errs, vals = [], []
        for b in bins:
            i = int(np.nanargmax(grid[:, b]))
            err = np.degrees(abs(((THETAS[i] - theta_true) + np.pi / 2) % np.pi - np.pi / 2))
            errs.append(round(float(err), 1))
            vals.append(round(float(grid[i, b]), 3))
        # cross-check: the OTHER estimator should NOT fire for case B's planted bin
        cross = float(np.nanmax(g[:, bins[0]])) if flip else None
        verdicts[case] = {"planted_theta_deg": round(np.degrees(theta_true), 1),
                          "planted_bins": bins, "recovered_err_deg": errs,
                          "gamma_at_recovery": vals,
                          "plain_gamma_at_planted_bin_if_flip": cross,
                          "pass": all(e <= 4.0 for e in errs) and all(v > 0.9 for v in vals)}
    return verdicts


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== synthetic recovery sanity ===")
    syn = synthetic_recovery()
    print(json.dumps(syn, indent=2))
    if not all(v["pass"] for v in syn.values()):
        print("STEP1_FINAL=synthetic_failed")
        (OUT / "synthetic.json").write_text(json.dumps(syn, indent=2))
        return

    print("=== real dataset heatmaps ===")
    scenes, obs, chunks, angles = make_dataset(200, 8, RNG)
    canon = to_canonical(chunks, angles[:, None])
    g, g2 = coherence_maps(canon, THETAS)
    sel = select_structure(g, g2, THETAS)

    print(ascii_heatmap(g, THETAS, "gamma (plain, mod 2pi):"))
    print()
    print(ascii_heatmap(g2, THETAS, "gamma2 (angle-doubled, mod pi; DC/Nyquist masked):"))
    print()
    print("selection:", json.dumps(sel, indent=2))

    (OUT / "synthetic.json").write_text(json.dumps(syn, indent=2))
    np.save(OUT / "gamma.npy", g)
    np.save(OUT / "gamma2.npy", g2)
    (OUT / "selection.json").write_text(json.dumps(
        {"thetas_deg_step": 2.0, "n_scenes": 200, "n_demos": 8, **sel}, indent=2))
    print(f"STEP1_FINAL={'g1_pass' if sel['g1_pass'] else 'g1_FAILED'}")


if __name__ == "__main__":
    main()
