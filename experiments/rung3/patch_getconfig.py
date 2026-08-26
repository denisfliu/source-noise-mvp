"""Idempotent patch: env-gated checkpoint-init override in get_config. When
SNMVP_INIT_CKPT points to an openpi params dir, the returned TrainConfig's weight_loader
is replaced by CheckpointWeightLoader(that path), so a run initializes from that
checkpoint (used for few-shot adaptation from a source model). No-op otherwise."""
import re, sys
PATH = sys.argv[1]
src = open(PATH).read()
if "SNMVP_INIT_CKPT" in src:
    print("already patched"); sys.exit(0)
old = "    return _CONFIGS_DICT[config_name]\n"
new = (
    "    _cfg = _CONFIGS_DICT[config_name]\n"
    "    import os as _os\n"
    "    _init = _os.environ.get(\"SNMVP_INIT_CKPT\")\n"
    "    if _init:\n"
    "        import dataclasses as _dc, logging as _lg\n"
    "        _cfg = _dc.replace(_cfg, weight_loader=weight_loaders.CheckpointWeightLoader(_init))\n"
    "        _lg.getLogger(\"openpi\").info(f\"SNMVP init-from-checkpoint: {_init}\")\n"
    "    return _cfg\n"
)
src2, n = re.subn(re.escape(old), new, src, count=1)
assert n == 1, "get_config return anchor not found"
open(PATH, "w").write(src2)
print("patched config.py: SNMVP_INIT_CKPT checkpoint-init override")
