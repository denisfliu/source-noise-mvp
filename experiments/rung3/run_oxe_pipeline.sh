#!/bin/bash
# Self-contained Open X-Embodiment offline transfer study. Streams action chunks
# from four robots and runs the subspace-transfer study. Launch detached with:
#   setsid bash run_oxe_pipeline.sh < /dev/null > oxe_pipeline.log 2>&1 & disown
set -u
cd ~/code/source-noise-mvp/experiments/rung3
UVR="$HOME/.local/bin/uv run --with tensorflow-cpu --with tensorflow-datasets --with numpy --python 3.11 python"
VENV=../../.venv/bin/python
echo "[oxe] start $(date -u +%H:%M:%S)"

for DS in berkeley_autolab_ur5 bridge toto viola; do
  if [ ! -s data_oxe/$DS.npz ]; then
    echo "[oxe] extracting $DS $(date -u +%H:%M:%S)"
    SNMVP_OXE_DS=$DS SNMVP_NEP=150 $UVR oxe_extract.py > oxe_extract_$DS.log 2>&1
    grep -h "episodes=" oxe_extract_$DS.log 2>/dev/null | tail -1
  fi
done
echo "[oxe] extraction done $(date -u +%H:%M:%S)"

SNMVP_OXE_ROBOTS=berkeley_autolab_ur5,bridge,toto,viola $VENV oxe_transfer.py > oxe_transfer.log 2>&1
echo "ALL_DONE $(date -u +%H:%M:%S)" > oxe_pipeline.status
echo "[oxe] finished $(date -u +%H:%M:%S)"
