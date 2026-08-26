"""Re-extract one Open X-Embodiment dataset with FULL observation (image + language + shared 6-D EE-delta
action), for the VLM-grounded meta-translation. The 6-D action (world_vector + rotation_delta) is the
representation shared across arm embodiments, so c can live in one space across robots. We keep only
anchor frames (stride) to bound size: per anchor we save the image, the H-step action chunk, and the
episode's language instruction. Schema keys are probed (OXE datasets differ). Output: data_oxe_full/<DS>.npz
with images (M,224,224,3 uint8), chunks (M,H,6 f32), langs (M str), plus per-frame episode id."""
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DS = os.environ["SNMVP_OXE_DS"]
N_EP = int(os.environ.get("SNMVP_NEP", "120"))
H, STRIDE = 16, 8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_oxe_full")
os.makedirs(OUT, exist_ok=True)


def img_key(obs):
    for k in ("image", "rgb", "agentview_rgb", "image_0", "hand_image", "wrist_image"):
        if k in obs:
            return k
    for k, v in obs.items():
        try:
            if len(v.shape) == 3 and v.shape[-1] == 3:
                return k
        except Exception:
            pass
    return None


def lang_of(step, obs):
    for src, k in ((step, "language_instruction"), (obs, "natural_language_instruction"), (obs, "language_instruction")):
        if k in src:
            v = src[k].numpy()
            return (v.decode("utf-8", "ignore") if isinstance(v, bytes) else str(v)).strip()
    return ""


def action6(a):
    if isinstance(a, dict):
        return np.concatenate([a["world_vector"].numpy().reshape(-1)[:3], a["rotation_delta"].numpy().reshape(-1)[:3]])
    a = a.numpy().reshape(-1)
    return a[:6]


def main():
    b = tfds.builder_from_directory(f"gs://gresearch/robotics/{DS}/0.1.0")
    ds = b.as_dataset(split="train")
    IMGS, CH, LANG, EID = [], [], [], []
    n_ep = 0
    for ep in ds:
        if n_ep >= N_EP:
            break
        steps = list(ep["steps"])
        obs0 = steps[0]["observation"]; ik = img_key(obs0)
        if ik is None:
            continue
        lang = lang_of(steps[0], obs0)
        acts = np.asarray([action6(s["action"]) for s in steps], np.float32)
        imgs = [tf.image.resize(s["observation"][ik], (224, 224), antialias=True).numpy().astype(np.uint8) for s in steps]
        for i in range(0, len(acts) - H + 1, STRIDE):
            IMGS.append(imgs[i]); CH.append(acts[i:i + H]); LANG.append(lang); EID.append(n_ep)
        n_ep += 1
        if n_ep % 25 == 0:
            print(f"  {DS} ep {n_ep} frames {len(IMGS)} lang0='{LANG[0][:40]}'", flush=True)
    IMGS = np.asarray(IMGS, np.uint8); CH = np.asarray(CH, np.float32)
    np.savez_compressed(os.path.join(OUT, f"{DS}.npz"), images=IMGS, chunks=CH,
                        langs=np.asarray(LANG), eid=np.asarray(EID))
    print(f"{DS}: eps={n_ep} frames={len(IMGS)} img={IMGS.shape} ch={CH.shape} langs={len(set(LANG))} uniq", flush=True)
    print("OXE_FULL_DONE=ok")


if __name__ == "__main__":
    main()
