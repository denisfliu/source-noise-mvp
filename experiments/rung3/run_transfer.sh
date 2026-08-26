#!/bin/bash
cd ~/code/source-noise-mvp/experiments/rung3
SNMVP_SETA=Panda,IIWA,UR5e SNMVP_HELD=Jaco ../../.venv/bin/python transfer6d.py > transfer6d_run.log 2>&1
