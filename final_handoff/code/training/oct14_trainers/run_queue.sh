#!/bin/bash
# Queue runner of experiments/v2/oct14_all_data. Queue lines (queue.txt, re-read every loop; '#' = comment; order = priority):
#   NAME KIND SRC FOLD [key=value ...]  -> scripts/launch.py --kind KIND --src SRC --name NAME --fold FOLD --set key=value ...
# (KIND main | basin | oa | bone | syn; SRC = the deployed source run's fold dir, whose config.json is the recipe; FOLD -1 = all data).
# Done = runs/NAME/foldF/done.txt. An arm started twice without finishing is skipped ("gave up").
# GPU gate: an arm starts only while gpu_go.txt holds 1, fewer than slots_max.txt arms run, and >= its need of GPU memory is free
# (queue token gb=N, default GB_FREE; an arm that does not fit is skipped for the next pending arm, so big arms wait for room).
# Logs: logs/NAME_fF.log; events: logs/events.txt. Stop this runner only by the PID in runner.pid (stopping it kills a running
# arm, its child): set gpu_go.txt to 0 to let running arms finish and start nothing new.
cd /home/pavel/Programming/knee-abnormality-2026
E=experiments/v2/oct14_all_data
Q=$E/queue.txt
L=$E/logs
GB_FREE=14
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
touch $L/events.txt
echo $$ > $E/runner.pid
is_done() { [ -f $E/runs/$1/fold$2/done.txt ]; }
running() { pgrep -f "^python $E/scripts/launch[.]py --kind [a-z]+ --src [^ ]+ --name $1 --fold $2( |$)" > /dev/null; }
n_running() { pgrep -af "^python $E/scripts/launch[.]py" | grep -oE -- "--name [^ ]+ --fold -?[0-9]+" | sort -u | wc -l; }   # DataLoader workers share the command line
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
go() { [ "$(cat $E/gpu_go.txt 2>/dev/null)" == "1" ]; }
while true; do
  started=0; pending=0
  while read -r n k src f sets; do
    [ -z "$n" ] && continue
    [[ $n == \#* ]] && continue
    is_done $n $f && continue
    running $n $f && { pending=1; continue; }
    if [ $(grep -c " start $n f$f " $L/events.txt) -ge 2 ]; then
      grep -q "gave up $n f$f$" $L/events.txt || echo "$(date +%H:%M) gave up $n f$f" >> $L/events.txt
      continue
    fi
    pending=1
    [ $started -eq 1 ] && break
    go && [ $(n_running) -lt $(cat $E/slots_max.txt 2>/dev/null || echo 1) ] || break
    need=$GB_FREE; args=""
    for t in $sets; do case $t in gb=*) need=${t#gb=} ;; *) args="$args $t" ;; esac; done
    [ $(free_gb) -ge $need ] || continue   # does not fit next to the running arms: try the next pending arm
    echo "$(date +%H:%M) start $n f$f $k $src$args (need ${need} GB)" >> $L/events.txt
    ( PYTHONPATH=. systemd-run --user --scope -q -p MemoryMax=36G /home/pavel/anaconda3/bin/conda run --no-capture-output -n rsna \
        python $E/scripts/launch.py --kind $k --src $src --name $n --fold $f --set $args > $L/${n}_f$f.log 2>&1 < /dev/null
      rc=$?; echo "$(date +%H:%M) end($rc) $n f$f" >> $L/events.txt ) &
    started=1
    break   # queue order is priority among arms that fit
  done < $Q
  [ $pending -eq 0 ] && [ $(n_running) -eq 0 ] && { grep -q "queue empty$" <(tail -1 $L/events.txt) || echo "$(date +%H:%M) queue empty" >> $L/events.txt; sleep 120; continue; }
  sleep $([ $started -eq 1 ] && echo 120 || echo 30)   # after a start, let it allocate before measuring free memory again
done
