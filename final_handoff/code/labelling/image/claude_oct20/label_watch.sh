#!/bin/bash
# Labelling watcher (5-min checks): p3 sessions 36-43 (the 'close the yardstick' batch: Contusion / Fracture on old-label positives
# without an Opus read). p3 24-35, p4 1-8 and g5 1-4 are complete. One line per change + alerts.
cd /home/pavel/Programming/knee-abnormality-2026
D=work/claude_labels_oct20
run() { systemd-run --user --scope -q -p MemoryMax=4G -E PATH="$PATH" -E HOME="$HOME" -E PYTHONPATH=$PWD nice -n 19 conda run -n rsna --no-capture-output python $D/$1.py > $D/$1.txt 2>&1; }
count() { ls $D/p3_s3[6-9]/*.json $D/p3_s4[0-3]/*.json 2>/dev/null | wc -l; }
prev=$(count)
while true; do
  sleep 300
  n=$(count)
  [ "$n" = "$prev" ] && continue
  run p3_monitor
  echo "$(date +%H:%M) close-batch reads $prev -> $n | p3 36-43: $(grep -E '^  s(3[6-9]|4[0-3]):' $D/p3_monitor.txt | grep -v ': 0 / ' | awk '{print $1 $2 $3 $4}' | tr '\n' ' ') | $(grep 'positives with' $D/p3_monitor.txt | sed 's/^ *positives with the 811 p2 reads: //') | rules: $(python3 $D/p3_rules_check.py --sessions 36-43 2>&1 | tail -1)"
  grep -h -E "ALERT|Traceback" $D/p3_monitor.txt
  prev=$n
done
