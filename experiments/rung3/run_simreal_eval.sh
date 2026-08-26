#!/usr/bin/env bash
# After the shared-space source-pin flow finishes, run the offline arm matrix twice with the
# SAME faithful evaluator: (1) LIBERO control (in-distribution; the pin must reproduce known-good
# behavior -> pass-through low, oracle >> no_pin) to validate the evaluator; (2) the real Bridge
# sim->real test. Both use pi0_libero_shared + the shared-space U; each domain uses its own norm
# stats (per-domain standardization). GPU 0.
set -u
RD=$HOME/code/source-noise-mvp/experiments/rung3
CK=$HOME/code/openpi/checkpoints/pi0_libero_shared/snmvp_src_pin_shared/4999
UV=$HOME/code/openpi/.venv/bin/python
cd "$HOME/code/openpi"
rm -f "$RD/simreal_eval.status"

until grep -q "SRC_SHARED_DONE" "$RD/shared_src.status" 2>/dev/null; do sleep 30; done
[ -d "$CK" ] || CK=$HOME/code/openpi/checkpoints/pi0_libero_low_mem_finetune/snmvp_src_pin_shared/4999
echo "FLOW_READY ckpt=$CK $(date -u +%H:%M:%S)" >> "$RD/simreal_eval.status"

run() {  # raw_dir norm out n_train n_eval tag
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 $UV "$RD/eval_offline_action.py" \
    --config pi0_libero_shared --U "$RD/pin_U_pca_k5_shared.npy" --ckpt "$CK" \
    --raw_dir "$1" --norm "$2" --out "$3" --n_train "$4" --n_eval "$5" --offsets 2 \
    > "$RD/eval_$6.log" 2>&1
  echo "$6 done $(date -u +%H:%M:%S)" >> "$RD/simreal_eval.status"
}

run "$RD/data_libero_raw" "$RD/norm_shared_libero" "$RD/libero_validate_shared.json" 25 12 libero_control
run "$RD/data_bridge_raw" "$RD/bridge_norm" "$RD/simreal_offline_shared.json" 200 80 bridge_main
echo "SIMREAL_EVAL_DONE $(date -u +%H:%M:%S)" >> "$RD/simreal_eval.status"
