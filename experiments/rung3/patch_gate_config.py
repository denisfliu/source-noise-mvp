"""Register pi0_gate: a copy of pi0_libero_shared (no-delta, LIBERO-style image+wrist+7DoF transforms)
pointed at the converted local/gate_nav dataset. Norm stats load from assets/pi0_gate/local/gate_nav/."""
import sys

PATH = sys.argv[1]
src = open(PATH).read()
if '"pi0_gate"' in src:
    print("already patched"); sys.exit(0)
anchor = ('    data=_dcshared.replace(_CONFIGS_DICT["pi0_libero_low_mem_finetune"].data, '
          'extra_delta_transform=False))\n')
assert anchor in src, "pi0_libero_shared registration anchor not found"
block = anchor + (
    '_CONFIGS_DICT["pi0_gate"] = _dcshared.replace(\n'
    '    _CONFIGS_DICT["pi0_libero_shared"], name="pi0_gate",\n'
    '    data=_dcshared.replace(_CONFIGS_DICT["pi0_libero_shared"].data, repo_id="local/gate_nav"))\n'
)
open(PATH, "w").write(src.replace(anchor, block))
print("patched config.py: registered pi0_gate -> local/gate_nav")
