"""LIBERO low-data ladder subsets (2026-08-09): equal per-task subsets of the 40-task
physical-intelligence/libero set -> local/libero_n{N} with N demos per task.

LeRobot v2.0 layout (differs from the drone set's v2.1): meta/episodes.jsonl + stats.json,
no episodes_stats.jsonl, and data is sharded across chunk-000/chunk-001 by episode index.
Episode choice is the first N per task (deterministic, nested), episode_index and the global
`index` column are renumbered, and info.json totals are rebuilt. Norm stats: reuse the
full-set assets so ladder points differ only by data quantity.

    python scripts/build_libero_subsets.py --n 2 5 10
"""
import argparse
import json
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

SRC = os.path.expanduser("~/.cache/huggingface/lerobot/physical-intelligence/libero")
DST_ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/local")
CHUNK = 1000


def src_parquet(info, ep_index):
    return os.path.join(SRC, info["data_path"].format(episode_chunk=ep_index // CHUNK,
                                                      episode_index=ep_index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", required=True, help="demos per task")
    a = ap.parse_args()

    info_src = json.load(open(f"{SRC}/meta/info.json"))
    eps = [json.loads(l) for l in open(f"{SRC}/meta/episodes.jsonl")]
    by_task = {}
    for e in eps:
        by_task.setdefault(e["tasks"][0], []).append(e["episode_index"])
    print(f"source: {len(eps)} episodes, {len(by_task)} tasks, "
          f"{min(len(v) for v in by_task.values())}-{max(len(v) for v in by_task.values())} per task",
          flush=True)

    for nper in a.n:
        keep = []
        for t in sorted(by_task):
            keep += sorted(by_task[t])[:nper]
        keep = sorted(keep)
        dst = f"{DST_ROOT}/libero_n{nper}"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(f"{dst}/data/chunk-000")
        os.makedirs(f"{dst}/meta")
        shutil.copy(f"{SRC}/meta/tasks.jsonl", f"{dst}/meta/tasks.jsonl")
        shutil.copy(f"{SRC}/meta/stats.json", f"{dst}/meta/stats.json")
        new_eps, gidx, frames = [], 0, 0
        for new_i, old_i in enumerate(keep):
            tb = pq.read_table(src_parquet(info_src, old_i))
            n = tb.num_rows
            for col, vals in (("episode_index", [new_i] * n),
                              ("index", list(range(gidx, gidx + n)))):
                ci = tb.schema.get_field_index(col)
                tb = tb.set_column(ci, tb.schema.field(ci),
                                   pa.array(vals, type=tb.schema.field(ci).type))
            gidx += n
            frames += n
            pq.write_table(tb, f"{dst}/data/chunk-000/episode_{new_i:06d}.parquet")
            e = dict(eps[old_i])
            assert e["episode_index"] == old_i
            e["episode_index"] = new_i
            new_eps.append(e)
        with open(f"{dst}/meta/episodes.jsonl", "w") as f:
            for e in new_eps:
                f.write(json.dumps(e) + "\n")
        info = dict(info_src)
        info["total_episodes"] = len(keep)
        info["total_frames"] = frames
        info["total_chunks"] = 1
        info["total_videos"] = 0
        info["splits"] = {"train": f"0:{len(keep)}"}
        json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)
        print(f"libero_n{nper}: {len(keep)} eps ({nper}/task), {frames} frames -> {dst}", flush=True)
    print("LIBERO_SUBSETS_DONE", flush=True)


if __name__ == "__main__":
    main()
