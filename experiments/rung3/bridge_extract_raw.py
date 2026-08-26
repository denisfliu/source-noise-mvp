"""Stage A of Bridge -> LeRobot: stream N Bridge episodes from the Open X-Embodiment
GCS mirror and save each as an intermediate npz (images uint8, state, 7-D action =
world_vector(3)+rotation_delta(3)+gripper(1)) plus a language string in meta.json.
Runs in a tfds-only uv env (no lerobot); Stage B converts these to LeRobot in the
openpi venv. Schema is auto-detected and logged for the first episode so the script
adapts to Bridge's exact key names. Env: SNMVP_NEP (episodes, default 300)."""
import os, json
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds

N_EP = int(os.environ.get("SNMVP_NEP", "300"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_bridge_raw")
os.makedirs(OUT, exist_ok=True)


def pick_image_key(obs):
    for k in obs:
        v = np.asarray(obs[k])
        if v.ndim == 3 and v.shape[-1] == 3 and v.dtype == np.uint8:
            return k
    for k in obs:
        if "image" in k.lower():
            return k
    return None


def pick_gripper_key(act):
    for cand in ["open_gripper", "gripper_closedness_action", "gripper", "grasp"]:
        if cand in act:
            return cand
    return None


def decode_lang(obs):
    # Bridge stores the instruction in observation["natural_language_instruction"] (bytes).
    for k in obs.keys():
        if "instruction" in k.lower() and "embed" not in k.lower():
            try:
                v = np.asarray(obs[k]).item()
                return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
            except Exception:
                pass
    return ""


def main():
    b = tfds.builder_from_directory("gs://gresearch/robotics/bridge/0.1.0")
    ds = b.as_dataset(split="train")
    meta = {}
    n = 0
    for ep in ds:
        if n >= N_EP:
            break
        imgs, states, acts, lang = [], [], [], None
        ikey = gkey = None
        for st in ep["steps"]:
            obs = st["observation"]
            act = st["action"]
            if ikey is None:
                ikey = pick_image_key(obs)
            img = np.asarray(obs[ikey])
            if img.shape[:2] != (256, 256):
                img = np.asarray(Image.fromarray(img).resize((256, 256), Image.BICUBIC))
            imgs.append(img.astype(np.uint8))
            state = np.asarray(obs.get("state", obs.get("proprio", np.zeros(7, np.float32))))
            states.append(state.reshape(-1).astype(np.float32))
            if hasattr(act, "keys"):
                if gkey is None:
                    gkey = pick_gripper_key(act)
                wv = np.asarray(act["world_vector"]).reshape(-1)[:3]
                rd = np.asarray(act["rotation_delta"]).reshape(-1)[:3]
                g = np.asarray(act[gkey]).reshape(-1)[:1] if gkey else np.zeros(1, np.float32)
                a = np.concatenate([wv, rd, g]).astype(np.float32)
            else:
                a = np.asarray(act).reshape(-1)[:7].astype(np.float32)
            acts.append(a)
            if lang is None:
                lang = decode_lang(obs)
        imgs = np.stack(imgs)
        states = np.stack(states)
        acts = np.stack(acts)
        np.savez_compressed(os.path.join(OUT, f"ep_{n:04d}.npz"), image=imgs, state=states, action=acts)
        meta[f"ep_{n:04d}"] = {"lang": lang or "", "T": int(len(acts)), "ikey": ikey,
                               "gkey": gkey, "state_dim": int(states.shape[1])}
        if n == 0:
            print(f"SCHEMA ikey={ikey} gkey={gkey} img={imgs.shape} state={states.shape} "
                  f"act={acts.shape} lang={lang!r}")
        n += 1
        if n % 25 == 0:
            print(f"...{n} episodes")
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
    print(f"BRIDGE_RAW_DONE n={n} out={OUT}")


if __name__ == "__main__":
    main()
