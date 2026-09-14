# openpi-snmvp working-tree snapshot

`~/code/openpi-snmvp` is a checkout of Physical-Intelligence/openpi at 15a9616 with this project's
modifications kept in the working tree (pin/head/sigma training and serving in pi0.py, `Policy.infer(noise=,
snmvp_sigma=, snmvp_t_start=)`, the `pi0_gate` config, data loader, checkpoint and weight-loader changes).
`openpi_snmvp_working_tree_<date>.patch` is `git diff` of that tree; to reproduce the serving stack:

    git clone https://github.com/Physical-Intelligence/openpi.git openpi-snmvp && cd openpi-snmvp
    git checkout 15a9616 && git apply <this repo>/patches/openpi_snmvp_working_tree_2026-09-04.patch

The 2026-09-04 snapshot adds `snmvp_t_start` (SDEdit-style partial denoising start time) to the sampler.

The 2026-09-11 snapshot adds the fused serve path: `Pi0.snmvp_prefix` (one prefix pass returning the
LLM hidden states and the KV cache), `Pi0.sample_actions_cached` (denoise on a supplied cache), and
`Policy.infer(..., cache=)`. The joint pin server runs the command head and the flow from one prefix pass
(SNMVP_FUSED=1, default): 133 -> 88 ms per replan on the dry client, outputs equal to the two-pass path to
bf16 noise (|dc| <= 0.12% of cstd, chunks within 1 mm over 8 steps).

The 2026-09-13 snapshot adds the coarse-only cross-domain swap (`_XDomSwap` mode=coarse: only the pinned
coordinates of a real chunk are replaced by the matched sim chunk's, via the minimum-acceleration chunk with
those band sums; env SNMVP_XDOM_SWAP_MODE=coarse, SNMVP_XDOM_NORM=<norm_stats.json>).
