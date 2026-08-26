"""Parameterize make_u_pca.py to build U under any config via SNMVP_CONFIG (default LIBERO)."""
import sys

p = sys.argv[1]
s = open(p).read()
old = 'config = _config.get_config("pi0_libero_low_mem_finetune")'
new = 'config = _config.get_config(os.environ.get("SNMVP_CONFIG", "pi0_libero_low_mem_finetune"))'
if new in s:
    print("already patched"); sys.exit(0)
assert old in s, "anchor not found"
open(p, "w").write(s.replace(old, new))
print("patched make_u_pca.py: SNMVP_CONFIG")
