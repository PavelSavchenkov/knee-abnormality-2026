#!/bin/bash
# Queue runner of experiments/v3/oct16_experts (copied from oct15_labels/run_queue.sh). Queue lines (queue.txt, re-read every loop; '#' = comment; order = priority):
#   NAME KIND SRC FOLD [key=value ...]  -> scripts/launch.py --kind KIND --src SRC --name NAME --fold FOLD --set key=value ...
# (KIND main | basin; SRC = a config dir relative to the experiment, e.g. refs/sub_b0/fold0 = the deployed main recipe).
# Done = runs/NAME/foldF/done.txt. An arm started twice without finishing is skipped ("gave up").
# GPU gate: an arm starts only while gpu_go.txt holds 1, fewer than slots_max.txt arms run, and >= its need of GPU memory is free
# (queue token gb=N, default GB_FREE; an arm that does not fit is skipped for the next pending arm, so big arms wait for room).
# Logs: logs/NAME_fF.log; events: logs/events.txt. Stop this runner only by the PID in runner.pid (stopping it kills a running
# arm, its child): set gpu_go.txt to 0 to let running arms finish and start nothing new.
cd /home/pavel/Programming/knee-abnormality-2026
E=experiments/v3/oct16_experts
Q=$E/queue.txt
L=$E/logs
GB_FREE=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
touch $L/events.txt
echo $$ > $E/runner.pid
is_done() { if [ "$3" == "dump" ]; then [ -f $E/runs/$1/fold$2/dumped.txt ]; else [ -f $E/runs/$1/fold$2/done.txt ]; fi; }
running() { [ -f /tmp/ev3_dump_$1_f$2.lock ] || [ -f /tmp/ev3_train_$1_f$2.lock ] || pgrep -f "^python $E/scripts/(launch[.]py --kind [a-z]+ --src [^ ]+ --name $1 --fold $2( |$)|dump_oa[.]py $1 $2 )" > /dev/null; }
n_running() { pgrep -af "^python $E/scripts/launch[.]py" | grep -oE -- "--name [^ ]+ --fold -?[0-9]+" | sort -u | wc -l; }   # DataLoader workers share the command line
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
go() { [ "$(cat $E/gpu_go.txt 2>/dev/null)" == "1" ]; }
while true; do
  started=0; pending=0
  while read -r n k src f sets; do
    [ -z "$n" ] && continue
    [[ $n == \#* ]] && continue
    is_done $n $f $k && continue
    if [ "$k" == "dump" ] && [ ! -f $E/runs/$n/fold$f/done.txt ]; then pending=1; continue; fi   # dump waits for its training
    running $n $f && { pending=1; continue; }
    tag=$n; [ "$k" == "dump" ] && tag=dump_$n
    if [ $(grep -c " start $tag f$f " $L/events.txt) -ge 2 ]; then
      grep -q "gave up $tag f$f$" $L/events.txt || echo "$(date +%H:%M) gave up $tag f$f" >> $L/events.txt
      continue
    fi
    pending=1
    [ $started -eq 1 ] && break
    go && [ $(n_running) -lt $(cat $E/slots_max.txt 2>/dev/null || echo 1) ] || break
    need=$GB_FREE; args=""
    for t in $sets; do case $t in gb=*) need=${t#gb=} ;; *) args="$args $t" ;; esac; done
    [ $(free_gb) -ge $need ] || continue   # does not fit next to the running arms: try the next pending arm
    echo "$(date +%H:%M) start $tag f$f $k $src$args (need ${need} GB)" >> $L/events.txt
    dep=19; [[ $src =~ ^[0-9]+$ ]] && dep=$src   # dump lines: SRC = the epoch to dump ('-' = 19)
    if [ "$k" == "dump" ]; then   # oct16_experts: per-crop (s, u) dumps of a finished OA expert run (oof + gold) for crop-fusion slot reads
    touch /tmp/ev3_dump_${n}_f${f}.lock   # 10-20 09:3x: marked running from launch across the oof -> gold gap (box double start 09:15)
    ( PYTHONPATH=. systemd-run --user --scope -q -p MemoryMax=36G conda run --no-capture-output -n rsna \
        python $E/scripts/dump_oa.py $n $f oof $dep > $L/${n}_f${f}_dump.log 2>&1 < /dev/null && PYTHONPATH=. systemd-run --user --scope -q -p MemoryMax=36G \
        conda run --no-capture-output -n rsna python $E/scripts/dump_oa.py $n $f gold $dep >> $L/${n}_f${f}_dump.log 2>&1 < /dev/null && touch $E/runs/$n/fold$f/dumped.txt
      rc=$?; rm -f /tmp/ev3_dump_${n}_f${f}.lock; echo "$(date +%H:%M) end($rc) $n f$f dump" >> $L/events.txt ) &
    started=1
    break
    fi
    touch /tmp/ev3_train_${n}_f${f}.lock   # 10-20 09:3x: marked running from launch
    ( PYTHONPATH=. systemd-run --user --scope -q -p MemoryMax=36G conda run --no-capture-output -n rsna \
        python $E/scripts/launch.py --kind $k --src $src --name $n --fold $f --set $args > $L/${n}_f$f.log 2>&1 < /dev/null
      rc=$?; rm -f /tmp/ev3_train_${n}_f${f}.lock; echo "$(date +%H:%M) end($rc) $n f$f" >> $L/events.txt ) &
    started=1
    break   # queue order is priority among arms that fit
  done < $Q
  [ $pending -eq 0 ] && [ $(n_running) -eq 0 ] && { grep -q "queue empty$" <(tail -1 $L/events.txt) || echo "$(date +%H:%M) queue empty" >> $L/events.txt; sleep 120; continue; }
  sleep $([ $started -eq 1 ] && echo 120 || echo 30)   # after a start, let it allocate before measuring free memory again
done
