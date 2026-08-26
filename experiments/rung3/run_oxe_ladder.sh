#!/bin/bash
set -u
RD=~/code/source-noise-mvp/experiments/rung3
PY=~/code/openpi/.venv/bin/python
ST=$RD/oxe_ladder.status
echo "LADDER_START $(date -u +%H:%M:%S)" >> $ST
until grep -q "OXE_VLM_DONE" $RD/oxe_vlm.log 2>/dev/null; do sleep 30; done
echo "VLM_FEATS_READY $(date -u +%H:%M:%S)" >> $ST
CUDA_VISIBLE_DEVICES= $PY $RD/oxe_shared_c.py > $RD/oxe_shared_c.out 2>&1
echo "SHARED_C rc=$? $(date -u +%H:%M:%S)" >> $ST
CUDA_VISIBLE_DEVICES= $PY $RD/oxe_new_embodiment.py > $RD/oxe_new_emb.out 2>&1
echo "NEW_EMB rc=$? $(date -u +%H:%M:%S)" >> $ST
echo "LADDER_DONE $(date -u +%H:%M:%S)" >> $ST
