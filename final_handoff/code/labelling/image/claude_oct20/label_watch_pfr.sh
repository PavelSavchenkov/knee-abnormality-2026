#!/bin/bash
# Labelling watcher (5-min checks) for the Phase 3 PF batch, p3 sessions 60-72 (the PF RE-READ batch: pf_oa + depth record). One line per change + alerts. OAI: no.
cd /home/pavel/Programming/knee-abnormality-2026
D=work/claude_labels_oct20
run() { systemd-run --user --scope -q -p MemoryMax=4G -E PATH="$PATH" -E HOME="$HOME" -E PYTHONPATH=$PWD nice -n 19 conda run -n rsna --no-capture-output python $D/$1.py "${@:2}" > $D/$1.txt 2>&1; }
count() { ls $D/p3_s6[0-9]/*.json $D/p3_s7[0-2]/*.json 2>/dev/null | wc -l; }
prev=$(count)
while true; do
  sleep 300
  n=$(count)
  [ "$n" = "$prev" ] && continue
  run pf_depth_check 60-72
  run p3_monitor
  echo "$(date +%H:%M) PF RE-READ reads $prev -> $n | $(sed -n 2,3p $D/pf_depth_check.txt | tr '\n' ' ') | $(grep -E '^  s(6[0-9]|7[0-2]):' $D/p3_monitor.txt | grep -v ': 0 / ' | awk '{print $1 $2 $3 $4}' | tr '\n' ' ')"
  grep -h -E "ALERT|Traceback" $D/pf_depth_check.txt $D/p3_monitor.txt
  prev=$n
done
