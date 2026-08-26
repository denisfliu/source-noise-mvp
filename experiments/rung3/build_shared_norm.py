"""Shared space via PER-DOMAIN standardization. LIBERO and Bridge actions have ~30x different
scales, so combined stats would be dominated by LIBERO and crush Bridge to near-zero. Instead
each domain is normalized by its OWN raw-delta stats (both become ~unit-variance), giving a
common standardized space; the single pin U (built on LIBERO-standardized actions) is the
cross-domain basis. Here we compute LIBERO's OWN raw-delta action + state stats (extra delta
transform OFF) and place them at the pi0_libero_shared asset dir used for training/U. Bridge eval
uses its own stats (experiments/rung3/bridge_norm, already built)."""
import glob
import os

import numpy as np
import openpi.shared.normalize as N
import lerobot.common.datasets.lerobot_dataset as L

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
RAW = os.path.join(RD, "data_bridge_raw")
ASSET = "/home/ubuntu/code/openpi/assets/pi0_libero_shared/physical-intelligence/libero"


def main():
    ds = L.LeRobotDataset("physical-intelligence/libero")
    hf = ds.hf_dataset.with_format("numpy")
    lib_a = np.asarray(hf["actions"], dtype=np.float32)
    lib_s = np.asarray(hf["state"], dtype=np.float32)
    rl_a = N.RunningStats(); rl_a.update(lib_a); lib_action = rl_a.get_statistics()
    rl_s = N.RunningStats(); rl_s.update(lib_s); lib_state = rl_s.get_statistics()

    os.makedirs(ASSET, exist_ok=True)
    N.save(ASSET, {"state": lib_state, "actions": lib_action})
    N.save(os.path.join(RD, "norm_shared_libero"), {"state": lib_state, "actions": lib_action})
    np.set_printoptions(precision=3, suppress=True)
    print("LIBERO frames", len(lib_a))
    print("LIBERO raw-delta action mean", np.asarray(lib_action.mean), "std", np.asarray(lib_action.std))
    print("SHARED_NORM_DONE asset=" + ASSET)


if __name__ == "__main__":
    main()
