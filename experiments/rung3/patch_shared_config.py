"""Idempotent patch: register a 'pi0_libero_shared' config on demand inside get_config.
It is a copy of pi0_libero_low_mem_finetune with extra_delta_transform=False, so the model
action space is simply normalized raw deltas (no cumulative delta transform) -- which the
offline evaluator can reproduce faithfully. Norm stats are read from assets/pi0_libero_shared/."""
import sys

PATH = sys.argv[1]
src = open(PATH).read()
if "pi0_libero_shared" in src:
    print("already patched"); sys.exit(0)
anchor = '    """Get a config by name."""\n'
assert src.count(anchor) == 1, "expected one get_config docstring"
block = anchor + (
    '    if config_name == "pi0_libero_shared" and config_name not in _CONFIGS_DICT:\n'
    "        import dataclasses as _dcs\n"
    '        _base = _CONFIGS_DICT["pi0_libero_low_mem_finetune"]\n'
    "        _CONFIGS_DICT[config_name] = _dcs.replace(\n"
    '            _base, name="pi0_libero_shared",\n'
    "            data=_dcs.replace(_base.data, extra_delta_transform=False))\n"
)
open(PATH, "w").write(src.replace(anchor, block))
print("patched config.py: registered pi0_libero_shared")
