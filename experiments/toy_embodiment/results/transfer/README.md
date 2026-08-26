# toy_embodiment Steps 2-4 — cross-embodiment transfer

Rung 1 of docs/cross_embodiment_plan.md. Frozen shared frame S_A (8 pins) + frozen scene->invariant prior trained on set A {arm2,arm3,arm4}; only the executor is adapted on held-out body B.
Task-space actions (invariant linear, pin exact).

Cross-body coherence c(B, setA): {"point": 0.893, "arm4": 0.933}

## Success on held-out scenes (100 scenes x 8 rollouts)

| B | seed | n | T | Toracle | S | Cond | Trand |
|---|---|---|---|---|---|---|---|
| point | 0 | 5 | **0.4025** | 0.4713 | 0.2188 | 0.2375 | 0.2425 |
| point | 0 | 10 | **0.6587** | 0.7925 | 0.4487 | 0.5062 | 0.4375 |
| point | 0 | 25 | **0.6625** | 0.7738 | 0.4825 | 0.6913 | 0.4288 |
| point | 0 | 50 | **0.6737** | 0.8013 | 0.52 | 0.6613 | 0.4437 |
| point | 1 | 5 | **0.2225** | 0.2313 | 0.1525 | 0.19 | 0.2238 |
| point | 1 | 10 | **0.5687** | 0.6262 | 0.46 | 0.41 | 0.445 |
| point | 1 | 25 | **0.6562** | 0.7462 | 0.495 | 0.655 | 0.4612 |
| point | 1 | 50 | **0.6625** | 0.8013 | 0.4963 | 0.6525 | 0.48 |
| point | 2 | 5 | **0.13** | 0.1625 | 0.3362 | 0.4575 | 0.3125 |
| point | 2 | 10 | **0.7137** | 0.82 | 0.5175 | 0.5913 | 0.505 |
| point | 2 | 25 | **0.7063** | 0.8462 | 0.5225 | 0.7375 | 0.4863 |
| point | 2 | 50 | **0.6963** | 0.84 | 0.5025 | 0.6775 | 0.4537 |
| arm4 | 0 | 5 | **0.2737** | 0.3137 | 0.1638 | 0.1338 | 0.1825 |
| arm4 | 0 | 10 | **0.4825** | 0.53 | 0.3875 | 0.3975 | 0.3025 |
| arm4 | 0 | 25 | **0.6613** | 0.7388 | 0.4375 | 0.6238 | 0.395 |
| arm4 | 0 | 50 | **0.665** | 0.7688 | 0.4612 | 0.6412 | 0.435 |
| arm4 | 1 | 5 | **0.2025** | 0.2175 | 0.1988 | 0.1663 | 0.16 |
| arm4 | 1 | 10 | **0.5825** | 0.6288 | 0.4387 | 0.37 | 0.435 |
| arm4 | 1 | 25 | **0.6525** | 0.7275 | 0.4475 | 0.585 | 0.4387 |
| arm4 | 1 | 50 | **0.6575** | 0.7712 | 0.4925 | 0.6388 | 0.4462 |
| arm4 | 2 | 5 | **0.095** | 0.1375 | 0.29 | 0.3 | 0.2812 |
| arm4 | 2 | 10 | **0.6837** | 0.7863 | 0.5012 | 0.6162 | 0.4587 |
| arm4 | 2 | 25 | **0.7238** | 0.855 | 0.4875 | 0.6763 | 0.5012 |
| arm4 | 2 | 50 | **0.665** | 0.7987 | 0.4763 | 0.6488 | 0.4363 |

## G-transfer (pooled, low n<=10)

- **point**: T=0.449 vs S=0.356 / Cond=0.399 / Trand=0.361  ->  T>S True, T>Cond True, T>Trand True (gain 0.094)
- **arm4**: T=0.387 vs S=0.33 / Cond=0.331 / Trand=0.303  ->  T>S True, T>Cond True, T>Trand True (gain 0.057)

- **G_transfer_pass = True**
- G-predict (directional, n=2 bodies): gain order ['arm4', 'point'] vs coherence order ['point', 'arm4']

Reading: T = frozen arm-learned frame+prior + executor adapted on B's
few demos; S = B from scratch on the same demos; Cond = same invariant
conditioned not pinned; Trand = random-frame pin. T>S = transfer helps;
T>Cond = the pin channel; T>Trand = the LEARNED frame specifically.