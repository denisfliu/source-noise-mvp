#!/usr/bin/env bash
# One Phase 1 training run + post-run evals. Resume-aware and idempotent:
# rerunning after a kill resumes from the last checkpoint; a run whose final
# eval JSON exists is skipped entirely.
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> phase1_run.sh <arm: A|B|C> <seed>
set -uo pipefail

ARM="$1"; SEED="$2"
REPO="$HOME/code/source-noise-mvp"
OPENPI="$HOME/code/openpi"
CKPT="$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
STEPS=15000
EXP="phase1_${ARM}_s${SEED}"
RESULTS="$REPO/experiments/phase1/results"
FINAL_STEP=$((STEPS - 1))
FINAL_EVAL="$RESULTS/evals/${EXP}_step${FINAL_STEP}.json"

export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
export HF_HOME="$HOME/code/hf-cache" HF_LEROBOT_HOME="$HOME/code/hf-cache/lerobot"

if [ -f "$FINAL_EVAL" ]; then
  echo "P1RUN: $EXP already complete (final eval exists), skipping"
  exit 0
fi

unset SNMVP_PIN_ALPHA SNMVP_COND_STATS SNMVP_PINNED_DIMS 2>/dev/null || true
case "$ARM" in
  A) ;;
  B) export SNMVP_COND_STATS="$REPO/experiments/phase1/invariant_stats.json" ;;
  C) export SNMVP_PIN_ALPHA=1.0 SNMVP_PINNED_DIMS=7 ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

# resume if the run already has checkpoints, else start fresh
MODE="--overwrite"
if ls "$OPENPI/checkpoints/pi0_libero/$EXP"/*/model.safetensors >/dev/null 2>&1; then
  MODE="--resume"
fi

# run manifest (config provenance per CLAUDE.md)
mkdir -p "$RESULTS/evals"
python3 - "$ARM" "$SEED" "$EXP" "$MODE" <<'EOF'
import json, subprocess, sys, os, pathlib
arm, seed, exp, mode = sys.argv[1:5]
commit = subprocess.check_output(["git", "-C", os.path.expanduser("~/code/openpi"), "rev-parse", "--short", "HEAD"]).decode().strip()
p = pathlib.Path(os.path.expanduser(f"~/code/source-noise-mvp/experiments/phase1/results/{exp}_manifest.json"))
manifests = json.loads(p.read_text()) if p.exists() else []
manifests.append({
    "exp": exp, "arm": arm, "seed": int(seed), "launch_mode": mode,
    "openpi_commit": commit, "torch": "2.7.1+cu128",
    "config": {"name": "pi0_libero", "num_train_steps": 15000,
               "lr_decay_steps": 15000, "batch_size": 32, "save_interval": 2500,
               "single_gpu": True, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")},
    "env": {k: os.environ.get(k) for k in
            ("SNMVP_PIN_ALPHA", "SNMVP_PINNED_DIMS", "SNMVP_COND_STATS")},
    "note": "resume reseeds data order; not bit-identical to unpaused run" if mode == "--resume" else None,
})
p.write_text(json.dumps(manifests, indent=2))
EOF

echo "P1RUN: $EXP starting ($MODE) on GPU ${CUDA_VISIBLE_DEVICES:-?} $(date -u +%H:%M:%S)"
( cd "$OPENPI" && uv run scripts/train_pytorch.py pi0_libero \
    --exp_name "$EXP" --seed "$SEED" \
    --pytorch-weight-path "$CKPT" \
    --num-train-steps $STEPS --lr-schedule.decay-steps $STEPS \
    --save-interval 2500 --no-wandb-enabled $MODE )
RC=$?
if [ $RC -ne 0 ]; then echo "P1RUN: $EXP TRAIN FAILED rc=$RC"; exit 1; fi

FINAL_CKPT="$OPENPI/checkpoints/pi0_libero/$EXP/$FINAL_STEP"
[ -d "$FINAL_CKPT" ] || { echo "P1RUN: $EXP final checkpoint missing"; exit 1; }

# success-rate eval (canonical states, 10 trials/task; server shares GPU 1)
bash "$REPO/scripts/eval_checkpoint.sh" "$FINAL_CKPT" 10 "$FINAL_EVAL" || echo "P1RUN: $EXP eval failed"

# arm C extra: offline wrong-invariant probe on this run's own GPU
if [ "$ARM" = "C" ]; then
  ( cd "$OPENPI" && uv run python "$REPO/scripts/wrong_invariant_probe.py" \
      --checkpoint "$FINAL_CKPT" --num-samples 16 --noise-draws 4 --seed 0 \
      --out "$RESULTS/${EXP}_wrong_invariant_probe.json" ) \
    || echo "P1RUN: $EXP probe failed"
fi
echo "P1RUN: $EXP DONE"
