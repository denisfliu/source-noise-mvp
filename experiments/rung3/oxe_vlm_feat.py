"""Extract contextualized VLM features (image+language, pi0 PaliGemma) for the full-obs OXE embodiments,
so we can build the shared VLM-grounded c across robots and later the unpaired OT-flow alignment. Feeds
each OXE frame's image as observation/image and its language as prompt (state/wrist dummy -- the prefix
is image+language only). One cache per embodiment: vlm_feat_oxe_<ds>.npz {X (M,2048), chunks, eid, langs}."""
import os
import sys

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
DSS = os.environ.get("OXE_DSS", "bridge,berkeley_autolab_ur5,toto,viola").split(",")
CKPT = os.path.expanduser("~/code/openpi/checkpoints/pi0_libero_shared/snmvp_src_pin_rrr/4999")
NORM = os.path.join(RD, "norm_shared_libero")
BS = 16


def main():
    import jax, jax.numpy as jnp
    import openpi.training.config as _config
    import openpi.policies.policy_config as _policy_config
    import openpi.shared.normalize as _normalize
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    ns = _normalize.load(NORM)
    policy = _policy_config.create_trained_policy(_config.get_config("pi0_libero_shared"), CKPT, norm_stats=ns)

    def ctx(raws):
        tds = [policy._input_transform(dict(r)) for r in raws]
        b = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], 0), *tds)
        obs = _model.preprocess_observation(None, _model.Observation.from_dict(b), train=False)
        tok, mask, ar = policy._model.embed_prefix(obs)
        out, _ = policy._model.PaliGemma.llm([tok, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
        po = out[0].astype(jnp.float32); m = mask[..., None].astype(jnp.float32)
        return np.asarray((po * m).sum(1) / jnp.clip(m.sum(1), 1e-6))

    dummy_state = np.zeros(8, np.float32)
    for ds in DSS:
        f = os.path.join(RD, "data_oxe_full", ds + ".npz")
        if not os.path.exists(f):
            print(f"SKIP {ds} (no data)", flush=True); continue
        d = np.load(f, allow_pickle=True); imgs = d["images"]; langs = d["langs"]
        X = []
        for i in range(0, len(imgs), BS):
            raws = [{"observation/image": imgs[j], "observation/wrist_image": imgs[j],
                     "observation/state": dummy_state, "prompt": str(langs[j])} for j in range(i, min(i + BS, len(imgs)))]
            X.append(ctx(raws))
            if i % (BS * 20) == 0:
                print(f"  {ds} {i}/{len(imgs)}", flush=True)
        X = np.concatenate(X, 0).astype(np.float32)
        np.savez_compressed(os.path.join(RD, f"vlm_feat_oxe_{ds}.npz"), X=X, chunks=d["chunks"], eid=d["eid"], langs=langs)
        print(f"{ds}: VLM feats {X.shape} -> cached", flush=True)
    print("OXE_VLM_DONE", flush=True)


if __name__ == "__main__":
    main()
