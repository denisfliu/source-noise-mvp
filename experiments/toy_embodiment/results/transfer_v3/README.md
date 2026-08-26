# toy_embodiment Steps 2-4 — cross-embodiment transfer

Rung 1 of docs/cross_embodiment_plan.md. Frozen shared frame S_A (3 pins) + frozen scene->invariant prior trained on set A {arm2,arm3,arm4}; only the executor is adapted on held-out body B.
Task-space actions (invariant linear, pin exact).

Cross-body coherence c(B, setA): {"point_phase0": 0.797, "point_phase15": 0.725, "point_phase30": 0.718, "point_phase45": 0.639}

## Success on held-out scenes (100 scenes x 8 rollouts)

| B | seed | n | T | Toracle | S | Cond | Trand |
|---|---|---|---|---|---|---|---|
| point_phase0 | 0 | 10 | **0.4688** | 0.49 | 0.3262 | 0.5112 | 0.3287 |
| point_phase0 | 0 | 25 | **0.6288** | 0.705 | 0.5175 | 0.6625 | 0.4963 |
| point_phase0 | 1 | 10 | **0.615** | 0.68 | 0.4562 | 0.4325 | 0.5125 |
| point_phase0 | 1 | 25 | **0.6338** | 0.7037 | 0.4475 | 0.6488 | 0.55 |
| point_phase0 | 2 | 10 | **0.5** | 0.5413 | 0.51 | 0.615 | 0.3638 |
| point_phase0 | 2 | 25 | **0.6038** | 0.6475 | 0.4913 | 0.7312 | 0.515 |
| point_phase15 | 0 | 10 | **0.4525** | 0.4587 | 0.3538 | 0.49 | 0.3588 |
| point_phase15 | 0 | 25 | **0.615** | 0.6787 | 0.5262 | 0.6438 | 0.4575 |
| point_phase15 | 1 | 10 | **0.5188** | 0.6088 | 0.4763 | 0.515 | 0.4425 |
| point_phase15 | 1 | 25 | **0.595** | 0.71 | 0.4225 | 0.695 | 0.4637 |
| point_phase15 | 2 | 10 | **0.45** | 0.5225 | 0.4512 | 0.6388 | 0.3837 |
| point_phase15 | 2 | 25 | **0.605** | 0.6525 | 0.495 | 0.735 | 0.5012 |
| point_phase30 | 0 | 10 | **0.4238** | 0.405 | 0.375 | 0.4075 | 0.3613 |
| point_phase30 | 0 | 25 | **0.5975** | 0.6375 | 0.4612 | 0.6025 | 0.4763 |
| point_phase30 | 1 | 10 | **0.515** | 0.5687 | 0.4587 | 0.5288 | 0.3837 |
| point_phase30 | 1 | 25 | **0.5713** | 0.6362 | 0.4025 | 0.6512 | 0.41 |
| point_phase30 | 2 | 10 | **0.3875** | 0.3962 | 0.3075 | 0.4113 | 0.36 |
| point_phase30 | 2 | 25 | **0.5413** | 0.5637 | 0.4562 | 0.7013 | 0.505 |
| point_phase45 | 0 | 10 | **0.41** | 0.4487 | 0.2888 | 0.38 | 0.28 |
| point_phase45 | 0 | 25 | **0.5675** | 0.6038 | 0.4525 | 0.5437 | 0.4113 |
| point_phase45 | 1 | 10 | **0.45** | 0.4875 | 0.3862 | 0.5012 | 0.3362 |
| point_phase45 | 1 | 25 | **0.5312** | 0.5387 | 0.3362 | 0.53 | 0.3262 |
| point_phase45 | 2 | 10 | **0.3463** | 0.33 | 0.3375 | 0.3137 | 0.29 |
| point_phase45 | 2 | 25 | **0.5012** | 0.5375 | 0.4163 | 0.6275 | 0.4113 |

## G-transfer (pooled over n=10,25)

- **point_phase0**: T=0.575 vs S=0.458 / Cond=0.6 / Trand=0.461  ->  T>S True, T>Cond False, T>Trand True (gain 0.117)
- **point_phase15**: T=0.539 vs S=0.454 / Cond=0.62 / Trand=0.435  ->  T>S True, T>Cond False, T>Trand True (gain 0.085)
- **point_phase30**: T=0.506 vs S=0.41 / Cond=0.55 / Trand=0.416  ->  T>S True, T>Cond False, T>Trand True (gain 0.096)
- **point_phase45**: T=0.468 vs S=0.37 / Cond=0.483 / Trand=0.343  ->  T>S True, T>Cond False, T>Trand True (gain 0.098)

- **G_transfer_pass = False**
- G-predict (directional, n=2 bodies): gain order ['point_phase15', 'point_phase30', 'point_phase45', 'point_phase0'] vs coherence order ['point_phase45', 'point_phase30', 'point_phase15', 'point_phase0']

Reading: T = frozen arm-learned frame+prior + executor adapted on B's
few demos; S = B from scratch on the same demos; Cond = same invariant
conditioned not pinned; Trand = random-frame pin. T>S = transfer helps;
T>Cond = the pin channel; T>Trand = the LEARNED frame specifically.