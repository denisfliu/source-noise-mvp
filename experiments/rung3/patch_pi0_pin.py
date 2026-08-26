"""Idempotent patch: add an env-gated source-noise pin to pi0.compute_loss.

When SNMVP_PIN_U points to a .npy of an orthonormal (D=ah*ad, K) matrix U, the
source noise's U-subspace coordinate is replaced by the action's own coordinate on
U, so the flow-matching target velocity is zero on U and the coordinate passes
through to the generated action. Unset SNMVP_PIN_U leaves behavior identical to
baseline. No inference-side change is needed: sample_actions already accepts a
`noise` argument, so pinned noise is built externally at eval time.
"""
import re, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "src/openpi/models/pi0.py"
src = open(PATH).read()

if "SNMVP pin" in src:
    print("already patched"); sys.exit(0)

# 1) module-level loader after the logger line
loader = '''logger = logging.getLogger("openpi")

# --- SNMVP source-noise pin (env-gated; no-op unless SNMVP_PIN_U is set) ---
import os as _os
import numpy as _np
_PIN_U = None
_pin_path = _os.environ.get("SNMVP_PIN_U")
if _pin_path:
    _PIN_U = _np.load(_pin_path).astype("float32")  # numpy constant (baked into jit), (D=ah*ad, K)
    logger.info(f"SNMVP pin enabled: U from {_pin_path} shape {_PIN_U.shape}")
'''
src, n1 = re.subn(r'logger = logging\.getLogger\("openpi"\)\n', loader, src, count=1)
assert n1 == 1, "logger line not found"

# 2) pin the noise in compute_loss, right after it is sampled
pin_block = '''        noise = jax.random.normal(noise_rng, actions.shape)
        if _PIN_U is not None:
            _U = jnp.asarray(_PIN_U)
            _b = actions.shape[0]
            _af = actions.reshape(_b, -1)
            _nf = noise.reshape(_b, -1)
            _nf = _nf - (_nf @ _U) @ _U.T + (_af @ _U) @ _U.T
            noise = _nf.reshape(actions.shape)
'''
src, n2 = re.subn(r'        noise = jax\.random\.normal\(noise_rng, actions\.shape\)\n',
                  pin_block, src, count=1)
assert n2 == 1, "noise sampling line not found"

open(PATH, "w").write(src)
print("patched pi0.py: added SNMVP env-gated source-noise pin")
