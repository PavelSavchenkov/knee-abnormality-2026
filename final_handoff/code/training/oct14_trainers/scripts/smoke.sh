#!/bin/bash
# Smoke tests of the five trainer kinds on fold -1 (20 steps, eval + save at the end). Run from the repo root.
cd /home/pavel/Programming/knee-abnormality-2026
E=experiments/v2/oct14_all_data
run() { n=$1; shift
  PYTHONPATH=. systemd-run --user --scope -q -p MemoryMax=36G conda run --no-capture-output -n rsna python $E/scripts/launch.py "$@" --name $n --fold -1 > $E/logs/$n.log 2>&1 < /dev/null
  echo "$(date +%H:%M:%S) $n rc=$? $(ls $E/runs/$n/fold-1 2>/dev/null | tr '\n' ' ')"; }
run SMK_main --kind main --src experiments/v1/oct10_synovitis/runs/m_prob05/fold0 --set spe_ref=435 max_steps=20 eval_from=1 save_from=1
run SMK_basin --kind basin --src experiments/v1/oct10_synovitis/runs/n_prob05/fold0 --set spe_ref=435 stop_step=20 stop_epoch=0
run SMK_oa --kind oa --src experiments/v1/oct10_oa/runs/oa_crop_v2cor/fold0 --set spe_ref=434 max_steps=20 eval_from=1 save_from=1
run SMK_bone --kind bone --src experiments/v1/oct11_fracture_contusion/runs/b_v3/fold0 --set spe_ref=434 max_steps=20 eval_from=1 save_from=1
run SMK_syn --kind syn --src experiments/v1/oct11_synovitis_crops/runs/s_fs_nf/fold0 --set spe_ref=435 max_steps=20 eval_from=1 save_from=1
echo SMOKE DONE
