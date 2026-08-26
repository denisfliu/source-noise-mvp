"""Stream a capped number of episodes from one Open X-Embodiment dataset and save
6-D end-effector delta action chunks. The action per step is world_vector (3) plus
rotation_delta (3), the representation shared across the datasets chosen. Chunks are
fixed-length windows of the action sequence. Each dataset's chunks are centered per
channel and scaled to unit root-mean-square, so that reconstruction is compared on
the shape of the action sequence rather than its magnitude, which differs by robot.
"""
import os
import numpy as np
import tensorflow_datasets as tfds

DS = os.environ["SNMVP_OXE_DS"]
N_EP = int(os.environ.get("SNMVP_NEP", "150"))
H = 16
STRIDE = 8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_oxe")
os.makedirs(OUT, exist_ok=True)


def main():
    b = tfds.builder_from_directory(f"gs://gresearch/robotics/{DS}/0.1.0")
    ds = b.as_dataset(split="train")                         # cap episodes in the loop
    chunks = []
    n_ep = 0
    for ep in ds:
        if n_ep >= N_EP:
            break
        acts = []
        for st in ep["steps"]:
            a = st["action"]
            acts.append(np.concatenate([a["world_vector"].numpy().reshape(-1)[:3],
                                        a["rotation_delta"].numpy().reshape(-1)[:3]]))
        acts = np.asarray(acts, dtype=np.float64)
        for i in range(0, len(acts) - H + 1, STRIDE):
            chunks.append(acts[i:i + H])
        n_ep += 1
    chunks = np.asarray(chunks)                              # (M, H, 6)
    flat = chunks.reshape(-1, 6)
    chunks = chunks - flat.mean(axis=0)
    chunks = chunks / (np.sqrt((chunks ** 2).mean()) + 1e-9)
    np.savez(os.path.join(OUT, f"{DS}.npz"), chunks=chunks)
    print(f"{DS}: episodes={n_ep} chunks={chunks.shape}", flush=True)
    print("OXE_EXTRACT_DONE=ok")


if __name__ == "__main__":
    main()
