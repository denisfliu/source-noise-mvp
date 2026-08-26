"""Bind a pin prior to the basis it was fit for.

c = U^T a is basis-dependent: a prior fit for U1 and served with U2 emits a command that means
nothing, and the failure is silent — the flights just go somewhere wrong. This has bitten us once
already (LIBERO arms served against a prior built for a different basis). The only guard so far has
been matching env vars across chain scripts by hand, so this records the basis identity inside the
prior checkpoint and checks it at serve time.

Builders:  d.update(stamp(upath))  before torch.save
Servers:   verify(prior_dict, pin_u_path)  after loading both
"""
import hashlib
import os

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def stamp(upath, feat_ckpt=None):
    """Fields to merge into a prior checkpoint so it names its own basis and feature source.

    feat_ckpt is the flow checkpoint whose VLM produced the language features the prior was fit on.
    It matters as much as the basis: gc.lang_pool reads post-fusion language tokens out of the SERVED
    model, so a prior trained on features from checkpoint A and served on checkpoint B consumes a
    representation from different weights than it learned against — silently, with excellent offline
    metrics, because offline evaluation reuses the same cache. That is exactly what happened to the
    gate language priors (cache built on gate_both_pin_rrr/4999, served on gate_pin_zeropad/4999).
    """
    U = np.load(upath)
    d = {"pin_u": os.path.abspath(upath), "pin_u_sha": sha(upath), "pin_u_shape": list(U.shape)}
    if feat_ckpt:
        d["feat_ckpt"] = os.path.abspath(feat_ckpt)
    return d


def verify_features(d, serve_ckpt, strict=False):
    """Check a prior's feature-source checkpoint against the one it is being served with.

    Default is non-strict: a mismatch is loud but not fatal, because deliberately serving a different
    flow is a legitimate experiment (e.g. the ladder rungs). Silence is what we cannot afford.
    """
    if not d.get("feat_ckpt"):
        return None
    label = "/".join(d["feat_ckpt"].rstrip("/").split("/")[-2:])   # exp-name/step, not just the step
    if os.path.abspath(serve_ckpt).rstrip("/") == d["feat_ckpt"].rstrip("/"):
        print(f"[pin_basis] feature source verified: {label}", flush=True)
        return True
    msg = (f"[pin_basis] FEATURE-SOURCE MISMATCH: this prior's language features were extracted with "
           f"{d['feat_ckpt']} but it is being served on {serve_ckpt}. The VLM weights differ, so the "
           f"embedding it consumes is not the one it was trained on.")
    if strict:
        raise SystemExit(msg)
    print(msg, flush=True)
    return False


def verify(d, pin_u_path, strict=True):
    """Check a loaded prior against the basis it is about to be served with.

    Unstamped priors (built before this existed) can only be shape-checked; say so rather than
    implying the pairing was verified.
    """
    U = np.load(pin_u_path)
    name = os.path.basename(pin_u_path)
    if d.get("K") is not None and int(d["K"]) != U.shape[1]:
        raise SystemExit(f"[pin_basis] prior K={d['K']} but basis {name} has {U.shape[1]} columns")
    if "pin_u_sha" not in d:
        print(f"[pin_basis] WARNING: prior has no basis stamp; only K={U.shape[1]} was checked "
              f"against {name}. Pairing is UNVERIFIED.", flush=True)
        return False
    if d["pin_u_sha"] == sha(pin_u_path):
        print(f"[pin_basis] basis verified: {name}", flush=True)
        return True
    msg = (f"[pin_basis] BASIS MISMATCH: prior was fit for {d['pin_u']} "
           f"(sha {d['pin_u_sha'][:12]}) but is being served with {pin_u_path} "
           f"(sha {sha(pin_u_path)[:12]}). The commanded c would be meaningless.")
    if strict:
        raise SystemExit(msg)
    print(msg, flush=True)
    return False
