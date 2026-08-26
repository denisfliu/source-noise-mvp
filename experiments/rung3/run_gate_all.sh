#!/usr/bin/env bash
# Self-contained, idempotent, SSH-drop-safe master for the gate sim/real experiments. One launch:
#   setsid bash run_gate_all.sh </dev/null >gate_all_driver.log 2>&1 & disown
# Waits for the v3->v2 conversion, computes norm stats, builds U, splits real into train/held-out,
# and trains the three patterns (synth+pin, synth-scratch, synth+real+pin). Every stage skips if its
# output already exists, so relaunching after a drop resumes. All progress -> gate_all.status.
set -uo pipefail
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_gate
UV=$HOME/.local/bin/uv
PY=$HOME/code/openpi/.venv/bin/python
GPU=${SNMVP_GPU:-1}   # head-to-head runs on GPU0; gate pipeline defaults to GPU1
cd "$HOME/code/openpi"
ST=$RD/gate_all.status
echo "GATE_START $(date -u +%H:%M:%S)" >> "$ST"

# 0) wait for the conversion to finish
until grep -q DONE ~/gate_convert.status 2>/dev/null; do sleep 30; done
echo "CONVERT_DONE $(date -u +%H:%M:%S)" >> "$ST"

# 1) split real into train (80%) / held-out (20%); build combined synth+real_train episode lists
$PY - <<PY >> "$ST" 2>&1
import json, os
RD=os.path.expanduser("$RD")
synth=json.load(open(f"{RD}/gate_synth_eps.json")); real=json.load(open(f"{RD}/gate_real_eps.json"))
real=sorted(real); nh=max(1,len(real)//5); held=real[::len(real)//nh][:nh]; held=set(sorted(real)[-nh:])
real_train=[e for e in real if e not in held]; real_held=sorted(held)
json.dump(real_train, open(f"{RD}/gate_real_train.json","w"))
json.dump(real_held, open(f"{RD}/gate_real_held.json","w"))
json.dump(sorted(synth+real_train), open(f"{RD}/gate_synth_realtrain.json","w"))
print("SPLIT synth",len(synth),"real_train",len(real_train),"real_held",len(real_held))
PY

# 2) norm stats for pi0_gate
if [ ! -f assets/pi0_gate/local/gate_nav/norm_stats.json ]; then
  CUDA_VISIBLE_DEVICES=$GPU $UV run scripts/compute_norm_stats.py --config-name pi0_gate > "$RD/gate_norm.log" 2>&1
  echo "NORM rc=$? $(date -u +%H:%M:%S)" >> "$ST"
fi

# 3) gate U (K=5) in pi0_gate normalized action space (CPU)
if [ ! -f "$RD/pin_U_gate_k5.npy" ]; then
  SNMVP_CONFIG=pi0_gate SNMVP_K=5 SNMVP_NB=120 SNMVP_OUT="$RD/pin_U_gate_k5.npy" \
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 $UV run make_u_pca.py > "$RD/gate_u.log" 2>&1
  echo "U $(grep -o 'coverage=[0-9.]*' "$RD/gate_u.log" | tail -1) rc=$? $(date -u +%H:%M:%S)" >> "$ST"
fi

# 4) train the three patterns (idempotent)
train() {  # exp episodes pinU|""
  [ -d "$CK/$1/4999" ] && { echo "SKIP $1 (exists)" >> "$ST"; return; }
  if [ -n "$3" ]; then export SNMVP_PIN_U="$3"; else unset SNMVP_PIN_U; fi
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled SNMVP_EPISODES="$2" \
    $UV run scripts/train.py pi0_gate --exp-name="$1" --num-train-steps=5000 --save-interval=2500 --overwrite > "$RD/$1.log" 2>&1
  echo "TRAIN $1 rc=$? $(date -u +%H:%M:%S)" >> "$ST"
  unset SNMVP_PIN_U
}
train gate_synth_pin     "$RD/gate_synth_eps.json"        "$RD/pin_U_gate_k5.npy"
train gate_synth_scratch "$RD/gate_synth_eps.json"        ""
train gate_both_pin      "$RD/gate_synth_realtrain.json"  "$RD/pin_U_gate_k5.npy"
echo "GATE_TRAIN_ALL_DONE $(date -u +%H:%M:%S)" >> "$ST"
