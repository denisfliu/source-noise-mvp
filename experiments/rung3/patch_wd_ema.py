"""Idempotent patch: env-gated weight-decay / EMA override in get_config, so a run can
turn on regularization (SNMVP_WD sets AdamW weight_decay; SNMVP_EMA sets ema_decay).
Inserted just before get_config's `return _cfg` (added by the earlier init-ckpt patch)."""
import sys
PATH = sys.argv[1]
src = open(PATH).read()
if "SNMVP_WD" in src:
    print("already patched"); sys.exit(0)
anchor = "    return _cfg\n"
assert src.count(anchor) == 1, "expected exactly one 'return _cfg'"
block = (
    "    _wd = _os.environ.get(\"SNMVP_WD\"); _ema = _os.environ.get(\"SNMVP_EMA\")\n"
    "    if _wd or _ema:\n"
    "        import dataclasses as _dc2, logging as _lg2\n"
    "        if _wd:\n"
    "            _cfg = _dc2.replace(_cfg, optimizer=_dc2.replace(_cfg.optimizer, weight_decay=float(_wd)))\n"
    "        if _ema:\n"
    "            _cfg = _dc2.replace(_cfg, ema_decay=float(_ema))\n"
    "        _lg2.getLogger(\"openpi\").info(f\"SNMVP reg: wd={_wd} ema={_ema}\")\n"
    "    return _cfg\n"
)
open(PATH, "w").write(src.replace(anchor, block))
print("patched config.py: SNMVP_WD / SNMVP_EMA override")
