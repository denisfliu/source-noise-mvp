"""Low-data-ladder datasets (north-star claim (b), 2026-08-06): equal per-task
subsets of local/gate_nav — n_per_task in {3, 10, 40} -> local/gate_nav_n{12,40,160}.

LeRobot v2.1 mechanics: episodes are renumbered densely; episode_index and the
global `index` column are rewritten inside each parquet; episodes.jsonl,
episodes_stats.jsonl and info.json totals are rebuilt; tasks.jsonl unchanged.
Episode choice is deterministic and nested (n12 ⊂ n40 ⊂ n160) so ladder points
differ only by data quantity. Selection is STRATIFIED BY DOMAIN (2026-08-07 fix):
episodes 0-99 are real-domain, 100-299 synth-domain; taking the first n per task
filled every rung's left/right demos with real episodes only (they sort first),
while the closed-loop eval runs in the gsplat synth renderer — a domain confound.
Real and synth episodes are interleaved per task (real, synth, real, ...) before
taking the first n, giving each rung an even domain mix on the gate tasks; the
center tasks are synth-only either way. Norm stats: reuse the full-data assets
(same normalization across the ladder — differences are data, not scaling).
"""
from itertools import zip_longest
import json
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

SRC = os.path.expanduser("~/.cache/huggingface/lerobot/local/gate_nav")
import os as _os
# env override so a new rung can be built without rebuilding existing ones
LADDER = ({int(_os.environ["ONLY_N"]): int(_os.environ["ONLY_N"]) // 4}
          if _os.environ.get("ONLY_N") else {12: 3, 40: 10, 160: 40})

eps = [json.loads(l) for l in open(f"{SRC}/meta/episodes.jsonl")]
stats = [json.loads(l) for l in open(f"{SRC}/meta/episodes_stats.jsonl")]
assert len(eps) == len(stats)
by_task = {}
for e in eps:
    by_task.setdefault(e["tasks"][0], []).append(e["episode_index"])

for total, nper in LADDER.items():
    keep = []
    for t in sorted(by_task):
        idxs = sorted(by_task[t])
        real = [i for i in idxs if i < 100]
        synth = [i for i in idxs if i >= 100]
        inter = [i for pair in zip_longest(real, synth) for i in pair if i is not None]
        keep += inter[:nper]
    keep = sorted(keep)
    dst = os.path.expanduser(f"~/.cache/huggingface/lerobot/local/gate_nav_n{total}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(f"{dst}/data/chunk-000"); os.makedirs(f"{dst}/meta")
    shutil.copy(f"{SRC}/meta/tasks.jsonl", f"{dst}/meta/tasks.jsonl")
    new_eps, new_stats, gidx, tot_frames = [], [], 0, 0
    for new_i, old_i in enumerate(keep):
        # pyarrow round-trip: preserves the HF `features` schema metadata that a
        # pandas rewrite drops (image struct columns decode as raw dicts otherwise)
        tb = pq.read_table(f"{SRC}/data/chunk-000/episode_{old_i:06d}.parquet")
        n = tb.num_rows
        for col, vals in (("episode_index", [new_i] * n),
                          ("index", list(range(gidx, gidx + n)))):
            ci = tb.schema.get_field_index(col)
            tb = tb.set_column(ci, tb.schema.field(ci),
                               pa.array(vals, type=tb.schema.field(ci).type))
        gidx += n; tot_frames += n
        pq.write_table(tb, f"{dst}/data/chunk-000/episode_{new_i:06d}.parquet")
        e = dict(eps[old_i]); assert e["episode_index"] == old_i
        e["episode_index"] = new_i; new_eps.append(e)
        s = dict(stats[old_i]); s["episode_index"] = new_i; new_stats.append(s)
    with open(f"{dst}/meta/episodes.jsonl", "w") as f:
        for e in new_eps:
            f.write(json.dumps(e) + "\n")
    with open(f"{dst}/meta/episodes_stats.jsonl", "w") as f:
        for s in new_stats:
            f.write(json.dumps(s) + "\n")
    info = json.load(open(f"{SRC}/meta/info.json"))
    info["total_episodes"] = len(keep); info["total_frames"] = tot_frames
    info["total_videos"] = 0
    info["splits"] = {"train": f"0:{len(keep)}"}
    json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)
    print(f"gate_nav_n{total}: {len(keep)} eps, {tot_frames} frames -> {dst}", flush=True)
print("SUBSAMPLE_DONE")
