# toy_embodiment Steps 2-4 — cross-embodiment transfer

Rung 1 of docs/cross_embodiment_plan.md. Frozen shared frame S_A (3 pins) + frozen scene->invariant prior trained on set A {arm2,arm3,arm4}; only the executor is adapted on held-out body B.
Task-space actions (invariant linear, pin exact).

Cross-body coherence c(B, setA): {"arm5": 0.92, "arm_short": 0.908, "point": 0.914, "point_drag": 0.898}

## Success on held-out scenes (100 scenes x 8 rollouts)

| B | seed | n | T | Toracle | S | Cond | Trand |
|---|---|---|---|---|---|---|---|
| arm5 | 0 | 10 | **0.465** | 0.4813 | 0.4925 | 0.4475 | 0.35 |
| arm5 | 0 | 25 | **0.6675** | 0.68 | 0.4213 | 0.7075 | 0.3663 |
| arm5 | 1 | 10 | **0.59** | 0.635 | 0.5288 | 0.4637 | 0.3987 |
| arm5 | 1 | 25 | **0.6125** | 0.6613 | 0.4412 | 0.65 | 0.3975 |
| arm5 | 2 | 10 | **0.575** | 0.5837 | 0.5725 | 0.5575 | 0.4562 |
| arm5 | 2 | 25 | **0.5988** | 0.6625 | 0.4537 | 0.6775 | 0.4363 |
| arm_short | 0 | 10 | **0.2313** | 0.2313 | 0.2313 | 0.2925 | 0.215 |
| arm_short | 0 | 25 | **0.3187** | 0.275 | 0.2087 | 0.3225 | 0.1787 |
| arm_short | 1 | 10 | **0.2762** | 0.2725 | 0.165 | 0.22 | 0.1837 |
| arm_short | 1 | 25 | **0.2725** | 0.2863 | 0.2537 | 0.3538 | 0.2275 |
| arm_short | 2 | 10 | **0.4525** | 0.4662 | 0.3875 | 0.445 | 0.415 |
| arm_short | 2 | 25 | **0.3987** | 0.4325 | 0.3063 | 0.405 | 0.3013 |
| point | 0 | 10 | **0.5225** | 0.5238 | 0.4562 | 0.5725 | 0.4 |
| point | 0 | 25 | **0.6625** | 0.66 | 0.47 | 0.7063 | 0.3762 |
| point | 1 | 10 | **0.5975** | 0.65 | 0.6075 | 0.5062 | 0.435 |
| point | 1 | 25 | **0.615** | 0.6637 | 0.46 | 0.6825 | 0.3887 |
| point | 2 | 10 | **0.5938** | 0.605 | 0.5312 | 0.6212 | 0.5288 |
| point | 2 | 25 | **0.615** | 0.6925 | 0.4775 | 0.6438 | 0.45 |
| point_drag | 0 | 10 | **0.5613** | 0.55 | 0.505 | 0.575 | 0.395 |
| point_drag | 0 | 25 | **0.6425** | 0.6637 | 0.4738 | 0.68 | 0.3488 |
| point_drag | 1 | 10 | **0.5737** | 0.5625 | 0.57 | 0.4925 | 0.5125 |
| point_drag | 1 | 25 | **0.6112** | 0.6312 | 0.45 | 0.7388 | 0.3962 |
| point_drag | 2 | 10 | **0.5537** | 0.535 | 0.58 | 0.5275 | 0.49 |
| point_drag | 2 | 25 | **0.6262** | 0.6238 | 0.555 | 0.6687 | 0.485 |

## G-transfer (pooled over n=10,25)

- **arm5**: T=0.585 vs S=0.485 / Cond=0.584 / Trand=0.401  ->  T>S True, T>Cond True, T>Trand True (gain 0.1)
- **arm_short**: T=0.325 vs S=0.259 / Cond=0.34 / Trand=0.254  ->  T>S True, T>Cond False, T>Trand True (gain 0.066)
- **point**: T=0.601 vs S=0.5 / Cond=0.622 / Trand=0.43  ->  T>S True, T>Cond False, T>Trand True (gain 0.101)
- **point_drag**: T=0.595 vs S=0.522 / Cond=0.614 / Trand=0.438  ->  T>S True, T>Cond False, T>Trand True (gain 0.072)

- **G_transfer_pass = False**
- G-predict (directional, n=2 bodies): gain order ['arm_short', 'point_drag', 'arm5', 'point'] vs coherence order ['point_drag', 'arm_short', 'point', 'arm5']

Reading: T = frozen arm-learned frame+prior + executor adapted on B's
few demos; S = B from scratch on the same demos; Cond = same invariant
conditioned not pinned; Trand = random-frame pin. T>S = transfer helps;
T>Cond = the pin channel; T>Trand = the LEARNED frame specifically.