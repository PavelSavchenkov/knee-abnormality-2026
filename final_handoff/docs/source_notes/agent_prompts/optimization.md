# Fast inference on Kaggle (4 CPUs, 2x T4): measured rules

Measured in Kaggle T4 kernels on 2026-10-07 (scripts and logs: `submissions/oct7_baseline_5_folds/speed/`).
A study = ~178 DICOM files, ~100 MB; the hidden test has ~1,300 studies. Reference code: `submissions/oct7_baseline_5_folds/knee_infer.py`.

## Method

- Measure on Kaggle, not locally: the local machine (28 cores, RTX 5090, NVMe) hides every Kaggle bottleneck.
- Time each stage alone first (data only, + GPU preprocessing, + models), each variant on unread studies (a second read is cached).
- Instrument the main loop: time waiting for data vs blocked on the GPU, per-stage worker time, system CPU busy % (`/proc/stat`).
  "Workers x worker time per study << wall time" means the consumer is the limit, not IO or CPU.
- Keep one-off costs (imports ~30 s, model load ~5 s, first CUDA batch ~12 s) out of per-study numbers; report them apart.
- Kaggle NFS speed varies between runs: the same code measured 0.32 and 0.55 s/study. Compare variants inside one kernel run, and
  repeat a measurement before believing a small change.
- Every speed change must keep outputs equal: byte-exact preprocessing where possible, else max pixel / probability diff on the check set.
- Test failure paths too (missing study, exception in a GPU thread): a hang or crash there costs the whole submission.

## Reading

- `/kaggle/input/competitions/...` is NFS (rsize 512 KB). Throughput grows with concurrent readers: 1 -> 18 MB/s, 8 -> 113, 32 -> 409,
  64 -> 614, 128 -> 658 MB/s; 8 processes x 16 threads reached ~950 MB/s. Read with >= 128 threads.
- Read whole files with `readinto` straight into the final buffer (one pinned uint8 tensor per study), file sizes from `os.scandir` stat.
- Thread pools must be per process (check `os.getpid()`): a forked child inherits the pool object without its threads; `map` hangs forever.

## Decoding

- pydicom `dcmread`: 0.28 ms/file locally, ~0.7 ms on Kaggle. A flat explicit-VR little-endian parser of the ~15 needed tags
  (`fast_read`) costs 0.036 ms/file and equals pydicom on 52,821 train files; anything unusual falls back to pydicom.
- Pixels: `np.frombuffer(buf, dtype, count, offset)` views, zero copy; on the GPU, int16 views of the uploaded study buffer at those offsets.

## CPU work and memory traffic

- Move pixel math to the GPU. cv2 `resize` is separable: build per-axis weights from cv2's formulas (`resize_matrix`: AREA =
  computeResizeAreaTab, LINEAR = 2-tap) and apply them as 2-4 shifted taps per axis (`resize_gpu`), not as dense matmuls (4x slower).
  AREA matches cv2 to 1e-7 of the range, LINEAR to 5e-5 (cv2 5.0 float LINEAR is not exact float math); after uint8: <= 1 grey level on
  0.003% of pixels, probabilities within 2e-4.
- Every extra pass over 100 MB/study costs: copying into fresh shared memory took 0.31 s/study per worker on Kaggle (page faults +
  contention); `share_memory_()` on a new tensor even copies its garbage first. Avoid worker processes for big arrays: threads in one process
  reading into pinned buffers need no shared memory, no pickling and no pin copy.
- numpy: in-place ops (`v /= hi; v *= 255; v += 0.5`), dtype + layout change in one copy (`np.array(view, dtype=np.float32, order="C")`).

## Moving data between stages

- Send compact data (uint8 slices, raw int16 frames), never float32; send each unique slice once plus an index, gather windows on the GPU.
- Bound memory in flight (studies queued x MB/study). 8 workers x prefetch 4 x 4 studies x 100 MB = 13 GB ended the Kaggle run in ERROR
  (15 GB /dev/shm) and hung locally.
- Pinned host memory + `non_blocking=True`. A blocking copy (`torch.tensor(list, device=dev)`, `.to(dev)` from pageable memory) waits for
  everything queued on that GPU: cache small constants per device and pin small batch tensors.

## GPU

- T4 has no bf16 (emulated, ~50x slower): backbone in fp16 (`.half()`), small head in fp32, channels_last.
- `cudnn.benchmark=True` costs ~35 s of autotuning per GPU and new batch shape on T4 (and every partial last batch) and was 5% slower
  afterwards: keep it off.
- effb0 at 256 px, 96 windows: 65 ms/study/model on one T4; 5 models on 2 T4s: 0.16 s/study.
- No gain on T4: BatchNorm folded into convs (1.02x), CUDA graphs, batch 768 instead of 384.
- torch.compile: default mode 1.29x (40 s compile for the first model, 0 s for more models of the same graph); max-autotune 1.6x (87 s).
  Worth it only when GPU-bound and the compile is not paid in the submission: AOTInductor package built once on a Kaggle T4 (needs
  `libcuda.so` on the link path: `-lcuda` fails by default), loaded with eager fallback.
- One Python thread cannot drive 2 GPUs: a batch of 5 effb0 forwards is ~1,500 kernel launches, the launch queue fills and blocks the thread
  on GPU 0 while GPU 1 idles (2 GPUs were only 7% faster than 1). Use one thread per GPU, fed by bounded queues, one batch in flight each.
- With other Python threads busy (readers, parsers), issuing eager ops is the bottleneck: the GPU waited on the CPU (issue time = GPU time,
  1 model as slow as 5); a shorter GIL switch interval did not help. CUDA graphs fix it: one `replay()` per model instead of ~250 dispatches
  (issue 0.26 -> 0.005 s/study, bit-identical outputs, 0.36 -> 0.29 s/study). Graph rules learned the hard way:
  - capture on a stream of the model's own device (`torch.cuda.graph(..., stream=side)`): the default capture stream is created once, on the
    first device used, and capturing cuda:1 work on it fails with "operation not permitted when stream is capturing";
  - one capture stream and one `graph_pool_handle()` per device for all models: a stream per graph stops pool reuse (5 x 2.7 GB vs 4.45 GB);
  - warm up eagerly on that stream before capture; static input shape: pad or run the last partial batch eagerly; clone outputs after replay;
  - capture before reader threads start (no concurrent CUDA calls such as pinned allocations).

## Python and reliability

- Cache per-shape constants (`functools.lru_cache` on resize taps keyed by size and device).
- Fault isolation: skip a broken series, give a broken study the prior, re-raise bugs from GPU threads in the main thread; log all of it.
- Fill fallbacks in numpy: pandas 3 refuses `df.loc[rows] = float64 values` on a float32 frame (pandas 2.3 on Kaggle only warns).
