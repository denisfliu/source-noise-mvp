"""Resolve a global LIBERO dataset task index to (suite-local index, language-onehot csv) for the
closed-loop client. Matches the global task's language string (from data_libero_multi meta) to the
suite's tasks, and builds the onehot in the SAME task ordering fit_lang_prior used (sorted unique
tasks). Prints: "<local_index> <onehot_csv>". Env: SNMVP_G (global idx), SNMVP_SU (suite name)."""
import contextlib
import json
import os
import sys

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
g = int(os.environ["SNMVP_G"])
su = os.environ["SNMVP_SU"]
meta = json.load(open(os.path.join(RD, "data_libero_multi", "meta.json")))
lang = next(v["lang"] for v in meta.values() if v["task"] == g)
# libero import + suite construction print [info] lines to stdout; keep our stdout clean.
with contextlib.redirect_stdout(sys.stderr):
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[su]()
    local = -1
    for i in range(suite.n_tasks):
        if suite.get_task(i).language.strip().lower() == lang.strip().lower():
            local = i
            break
tasks = sorted({v["task"] for v in meta.values()})
oh = np.zeros(len(tasks), int)
oh[tasks.index(g)] = 1
print("RESOLVED", local, ",".join(str(int(x)) for x in oh))
