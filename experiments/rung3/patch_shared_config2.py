"""Statically register pi0_libero_shared in _CONFIGS_DICT at import so tyro's cli() (used by
scripts/train.py) accepts it. Copy of pi0_libero_low_mem_finetune with extra_delta_transform=False."""
import sys

PATH = sys.argv[1]
src = open(PATH).read()
anchor = "_CONFIGS_DICT = {config.name: config for config in _CONFIGS}\n"
assert src.count(anchor) == 1, "expected one _CONFIGS_DICT definition"
if "_CONFIGS_DICT[\"pi0_libero_shared\"]" in src:
    print("already patched"); sys.exit(0)
block = anchor + (
    "import dataclasses as _dcshared\n"
    "_CONFIGS_DICT[\"pi0_libero_shared\"] = _dcshared.replace(\n"
    "    _CONFIGS_DICT[\"pi0_libero_low_mem_finetune\"], name=\"pi0_libero_shared\",\n"
    "    data=_dcshared.replace(_CONFIGS_DICT[\"pi0_libero_low_mem_finetune\"].data, extra_delta_transform=False))\n"
)
open(PATH, "w").write(src.replace(anchor, block))
print("patched config.py: pi0_libero_shared in _CONFIGS_DICT")
