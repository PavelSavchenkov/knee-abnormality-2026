#!/bin/bash
# Effusion pool labelling (pre-registered pick e_Ag_tshot) in a slot Experts opens: their hold line sets gpu_go 0 and touches
# logs/vlm_pool_go.flag; this runs the pass and touches logs/vlm_pool_done.flag (their hold line then sets gpu_go 1).
cd /home/pavel/Programming/knee-abnormality-2026
L=experiments/v3/oct16_experts/logs; D=work/vlm_label_2026-10-19
until [ -f $L/vlm_pool_go.flag ]; do sleep 30; done
echo "$(date +%H:%M) start pool_e_Ag_tshot"
systemd-run --user --scope -q -p MemoryMax=40G -E PATH="$PATH" -E HOME="$HOME" -E PYTHONPATH=.:work/vlm/pylib \
  conda run -n rsna --no-capture-output python -u $D/score.py Qwen/Qwen3.5-9B A $D/out/pool_e_Ag_tshot.csv --guide --findings Effusion \
  --shots $D/out/eshots.csv --studies $D/out/pool_studies_noshot.csv > $D/pool_e_Ag_tshot.log 2>&1
echo "$(date +%H:%M) end($?) pool_e_Ag_tshot"
touch $L/vlm_pool_done.flag
