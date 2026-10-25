#!/bin/bash
# sequential downloads after Qwen3.5-9B: medical 7B, then the 27B dense (4-bit on load)
cd /home/pavel/Programming/knee-abnormality-2026
while pgrep -f "download.py Qwen/Qwen3.5-9B" > /dev/null; do sleep 30; done
for m in lingshu-medical-mllm/Lingshu-7B Qwen/Qwen3.5-27B; do
  n=$(echo $m | tr '/' '_')
  conda run -n rsna python work/vlm_label_2026-10-19/download.py $m > work/vlm_label_2026-10-19/dl_$n.log 2>&1
done
