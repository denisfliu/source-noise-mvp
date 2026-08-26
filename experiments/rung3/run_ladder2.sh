#!/bin/bash
set -u
RD=~/code/source-noise-mvp/experiments/rung3
PY=~/code/openpi/.venv/bin/python
ST=$RD/oxe_ladder2.status
echo "LADDER2_START $(date -u +%H:%M:%S)" >> $ST
until grep -q "OXE_VLM_DONE" $RD/oxe_vlm_bridge.log 2>/dev/null; do sleep 30; done
echo "BRIDGE_VLM_READY $(date -u +%H:%M:%S)" >> $ST
CUDA_VISIBLE_DEVICES= $PY $RD/oxe_shared_c.py > $RD/oxe_shared_c_v2.out 2>&1
echo "SHARED_C2 rc=$? $(date -u +%H:%M:%S)" >> $ST
CUDA_VISIBLE_DEVICES= $PY $RD/oxe_new_embodiment.py > $RD/oxe_new_emb_v2.out 2>&1
echo "NEW_EMB2 rc=$? $(date -u +%H:%M:%S)" >> $ST
echo "LADDER2_DONE $(date -u +%H:%M:%S)" >> $ST
