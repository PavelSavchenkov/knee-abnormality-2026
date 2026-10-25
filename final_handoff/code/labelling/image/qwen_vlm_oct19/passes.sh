#!/bin/bash
# VLM passes 2+ on the local GPU, back to back, after Experts' oct20_stack_all run_local is done. Each line of passes.txt:
#   NAME MODEL VIEWSET [score.py options]   (lines starting with # are skipped; read fresh before each pass, so edit freely)
cd /home/pavel/Programming/knee-abnormality-2026
EV=experiments/v3/oct16_experts/logs/events.txt
until [ -f experiments/v3/oct16_experts/logs/vlm_gpu_go.flag ]; do sleep 30; done   # Experts touches it when ALL its local runs are done
D=work/vlm_label_2026-10-19
while true; do
  line=$(grep -v '^#' $D/passes.txt | grep -v '^\s*$' | while read n rest; do [ -f $D/out/$n.csv ] || [ -f $D/$n.started ] || { echo "$n $rest"; break; }; done)
  [ -z "$line" ] && { echo "$(date +%H:%M) passes queue empty"; break; }
  set -- $line; n=$1; shift
  touch $D/$n.started; echo "$(date +%H:%M) start $n $*"
  systemd-run --user --scope -q -p MemoryMax=40G -E PATH="$PATH" -E HOME="$HOME" -E PYTHONPATH=.:work/vlm/pylib \
    conda run -n rsna --no-capture-output python -u $D/score.py $1 $2 $D/out/$n.csv "${@:3}" > $D/$n.log 2>&1
  echo "$(date +%H:%M) end($?) $n"
done
