#!/usr/bin/env bash
# Score gate training patterns on held-out real via eval_offline_lang.py (offline action R^2 +
# pin-channel subspace R^2; the (state+which-gate)->c prior is fit on real, so pin checkpoints give
# the 'pin re-grounded on real' arm directly). Extracts real raw data if missing.
#   bash run_gate_eval.sh <GPU> <ckpt1> <ckpt2> ...
set -u
GPU=${1:-1}; shift; CKPTS="$*"
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_gate
PY=$HOME/code/openpi/.venv/bin/python
NORM=$HOME/code/openpi/assets/pi0_gate/local/gate_nav
U=$RD/pin_U_gate_k5.npy
cd "$HOME/code/openpi"
ST=$RD/gate_eval.status
[ -f "$RD/data_gate_real/meta.json" ] || { EPS=$RD/gate_real_eps.json OUT=$RD/data_gate_real $PY "$RD/gate_extract_raw.py" >> "$ST" 2>&1; }
for e in $CKPTS; do
  [ -d "$CK/$e/4999" ] || { echo "MISSING $e" >> "$ST"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 $PY "$RD/eval_offline_lang.py" \
    --config pi0_gate --U "$U" --ckpt "$CK/$e/4999" --norm "$NORM" \
    --raw_dir "$RD/data_gate_real" --out "$RD/gate_eval_$e.json" > "$RD/gate_eval_$e.log" 2>&1
  echo "EVAL_DONE $e rc=$? $(date -u +%H:%M:%S)" >> "$ST"
done
echo "GATE_EVAL_ALL_DONE $(date -u +%H:%M:%S)" >> "$ST"
