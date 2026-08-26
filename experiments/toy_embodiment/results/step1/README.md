# toy_embodiment Step 1 — cross-embodiment coherence (G-frame)

Rung 1 of docs/cross_embodiment_plan.md. Task-space (tip-delta) actions
for all bodies (invariant linear, pin exact); embodiment = reach +
radial-authority feasibility (embodiments.py). Coherence = phase
agreement ACROSS BODIES doing the same scene.

## Synthetic recovery (sanity)
- planted theta 35.0deg / omega 2 -> recovered 34.4deg (err 0.6deg)
- coherence at planted bin 1.0; with a divergent body added 0.694 (should drop)

## Demo success ceiling per body (achieved tip paths solve the task?)
| arm2 | arm3 | arm4 | point |
|---|---|---|---|
| 0.622 | 0.829 | 0.998 | 1.0 |

## Shared frame over set A {arm2,arm3,arm4}
- g1_pass = True; selected pins:
    - {'axis_deg': np.float64(0.0), 'omega': 0, 'mode': 'mod2pi', 'gamma': 1.0}
    - {'axis_deg': np.float64(90.0), 'omega': 0, 'mode': 'mod2pi', 'gamma': 0.963}
    - {'axis_deg': np.float64(0.0), 'omega': 1, 'mode': 'mod2pi', 'gamma': 0.995}
    - {'axis_deg': np.float64(90.0), 'omega': 1, 'mode': 'mod2pi', 'gamma': 0.871}
    - {'axis_deg': np.float64(0.0), 'omega': 2, 'mode': 'mod2pi', 'gamma': 0.995}
    - {'axis_deg': np.float64(90.0), 'omega': 2, 'mode': 'mod2pi', 'gamma': 0.86}
    - {'axis_deg': np.float64(0.0), 'omega': 3, 'mode': 'mod2pi', 'gamma': 0.967}
    - {'axis_deg': np.float64(90.0), 'omega': 3, 'mode': 'mod2pi', 'gamma': 0.815}
    - {'axis_deg': np.float64(0.0), 'omega': 4, 'mode': 'mod2pi', 'gamma': 0.981}
    - {'axis_deg': np.float64(90.0), 'omega': 4, 'mode': 'mod2pi', 'gamma': 0.801}
    - {'axis_deg': np.float64(0.0), 'omega': 5, 'mode': 'mod2pi', 'gamma': 0.984}
    - {'axis_deg': np.float64(90.0), 'omega': 5, 'mode': 'mod2pi', 'gamma': 0.803}
    - {'axis_deg': np.float64(0.0), 'omega': 6, 'mode': 'mod2pi', 'gamma': 0.984}
    - {'axis_deg': np.float64(90.0), 'omega': 6, 'mode': 'mod2pi', 'gamma': 0.828}
    - {'axis_deg': np.float64(0.0), 'omega': 7, 'mode': 'mod2pi', 'gamma': 0.984}
    - {'axis_deg': np.float64(90.0), 'omega': 7, 'mode': 'mod2pi', 'gamma': 0.867}
    - {'axis_deg': np.float64(0.0), 'omega': 8, 'mode': 'mod2pi', 'gamma': 0.984}
    - {'axis_deg': np.float64(90.0), 'omega': 8, 'mode': 'mod2pi', 'gamma': 0.866}
    - {'axis_deg': np.float64(0.0), 'omega': 9, 'mode': 'mod2pi', 'gamma': 0.983}
    - {'axis_deg': np.float64(90.0), 'omega': 9, 'mode': 'mod2pi', 'gamma': 0.867}
    - {'axis_deg': np.float64(0.0), 'omega': 10, 'mode': 'mod2pi', 'gamma': 0.983}
    - {'axis_deg': np.float64(90.0), 'omega': 10, 'mode': 'mod2pi', 'gamma': 0.867}

## Pairwise cross-body coherence c(i,j) on selected bins
| pair | c |
|---|---|
| arm2~arm3 | 0.933 |
| arm2~arm4 | 0.937 |
| arm2~point | 0.792 |
| arm3~arm4 | 0.924 |
| arm3~point | 0.806 |
| arm4~point | 0.812 |

- mean c(arm,arm) = 0.931  |  mean c(arm,point) = 0.803
- Reading: the point robot (drone analog, unconstrained) should be the
  most divergent from the arm family, so c(arm,arm) > c(arm,point).

## G-FRAME VERDICT
- synthetic_ok: True
- frame_found: True (22 pins)
- divergence_ordering_ok: True
- **G_FRAME_PASS = True**

See heatmaps.txt for the gamma/gamma2 grids; *.npy/*.json are the raw
artifacts. Next: Steps 2-4 (front-half prior, per-body executors,
freeze-and-adapt transfer) per docs/toy_embodiment_plan.md.