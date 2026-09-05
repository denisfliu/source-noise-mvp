# openpi-snmvp working-tree snapshot

`~/code/openpi-snmvp` is a checkout of Physical-Intelligence/openpi at 15a9616 with this project's
modifications kept in the working tree (pin/head/sigma training and serving in pi0.py, `Policy.infer(noise=,
snmvp_sigma=, snmvp_t_start=)`, the `pi0_gate` config, data loader, checkpoint and weight-loader changes).
`openpi_snmvp_working_tree_<date>.patch` is `git diff` of that tree; to reproduce the serving stack:

    git clone https://github.com/Physical-Intelligence/openpi.git openpi-snmvp && cd openpi-snmvp
    git checkout 15a9616 && git apply <this repo>/patches/openpi_snmvp_working_tree_2026-09-04.patch

The 2026-09-04 snapshot adds `snmvp_t_start` (SDEdit-style partial denoising start time) to the sampler.
