# 4. Inference, packaging and Kaggle

## 4.1 Kernel structure

`code/inference/kernels/final_alldata_P1_inv1_a.ipynb` is the exact notebook of final P1. It has one code cell; P3 is the same
code with another config dataset. The fold-average kernels:
- `fold_v2_strict.ipynb`: the scored v2 fold package, 0.964;
- `fold_v2_all4_sxm_kit.ipynb`, `fold_v2_frac_v3c_fs_kit.ipynb`: v2 + one LB-won column change each.

The cell, in order:
1. **Monitor thread** (RSS, MemAvailable, GPU memory every 120 s) and `faulthandler`. An OOM kill prints nothing otherwise.
2. **Setup**:
   - mount the variant's datasets: frozen weight datasets + one small config dataset with code + `config.json`;
   - check every file's md5 against the locally audited build;
   - copy the weights to `/tmp/pkgw` (6.2 GB, md5 again while copying, ~20 s) instead of reading them from the gcsfuse mount
     (cold mount reads ran at ~3 MB/s);
   - build the WORK folder (code copied, weights symlinked, `main_sd/` rebuilt from `main_sd_config.json`);
   - assert where each imported module comes from, and that the code has the expected features.
3. **DEC constants** (post-hoc de-couplings) and `decouple()`.
4. **`fast.py`** written from embedded text: the two-GPU runner (below).
5. **Commit check** (runs in every commit and every scoring run):
   - predict the first 4 "edge" training studies (`check_studies.csv`) and write `check_pred.csv` / `check_parts.csv` /
     `check_timing.json`;
   - the gates in 4.3 compare these against the local reference.
6. **Test prediction**: `fast_predict_all` over the hidden test in chunks, `decouple()`, `submission.csv`.

STRICT mode (user rule: "we absolutely do not need any fallbacks; a robust pipeline works or fails loudly"):
- **Raise** on anything guaranteed for a valid study: series-csv coverage, finite outputs of every stage, every part covering every
  study, both GPUs present, any failed chunk.
- **Log only** what a valid study can legitimately hit: one unreadable series; a study with no coronal series gets a constant MCL
  zoom logit.

## 4.2 The FAST two-GPU runner (`code/inference/fast_runner/fast.py`)

- The parent splits the test set into chunks. Per chunk, one worker process per T4 runs the Basin pass and the crop experts (one
  worker does Basin then crops, the other crops then Basin, so the GPUs overlap). The parent runs the main and window passes.
  `ensemble.predict` then combines the faked stage outputs exactly as the sequential path does. It is bit-identical to sequential
  on Kaggle.
- **Read-ahead** (`code/inference/package_code/dicom_prefetch.py`):
  - a separate process per worker reads the next ~24 studies' DICOM with 64 threads into the page cache;
  - Basin's dataset publishes its position through `/dev/shm`;
  - outputs are bit-identical (gated locally on 88 studies and on Kaggle);
  - if the prefetcher dies, the run is only slower.
- Pad batches to a fixed size: on T4 every new batch shape costs ~9.5 s of cuDNN autotuning.
- fp16 on T4: keep fp32 weights + autocast + a NaN guard (NFNet / DINO-style layers overflow under `.half()`).

## 4.3 Packaging and gates (`code/inference/packaging/`)

| step | script | what it checks |
|---|---|---|
| stock / freeze | `build.py --stock`, `--freeze NAME` | every finished run -> a store (strip training-only heads; key-for-key against the shipped fold file) -> frozen flat weight datasets (`knee-oct21-fa-base`, `-d1`...). A file lives in ONE dataset. Kaggle `datasets create` skips subfolders, so build them flat |
| variant | `build.py --variant NAME` (`variants/*.json`: form switches, seeds per recipe, PL twins, routes) | config dataset (code + config.json + syn members + main_sd config); refuses unfrozen files |
| audit | `audit.py --variant NAME` | strict state-dict loads of every model, config consistency, routing structure |
| setup test | `test_kernel_setup.py NAME` | the kernel's setup region on the real datasets, on CPU, including a one-byte-changed negative |
| kernel | `make_kernel.py --variant NAME [--local-copy]` | renders the notebook (needs the audit PASS) |
| commit gate | `commit_gate.py` | the Kaggle commit's shared parts are bit-identical to the reference commit's, except the parts of changed recipes |
| local reference | `local_ref.py VARIANT KAGGLE_COMMIT_DIR` | local 5090 vs Kaggle T4 on the same 4 studies: every column and part <= 1e-3 (fp16 cross-GPU noise) |
| gold sanity | `code/labelling/pseudo_labels/best_gold_table.py` | hold a candidate if its gold-58 macro drops > 1 seed-swap sd (0.0014) or a column drops > 3 x its seed-noise sd |
| submit | `submit_once.py DIR KERNEL VERSION`, `poll_all.py`, `lb_order.py` | one synchronous submit per call; one poller every 2 min (4 pollers every 30 s hit HTTP 429); hidden-precision order of equal 3-decimal scores |

- The fold-average package: `fold_pkg.py` assembles every member's fold models from the probe kits into one folder, ready for
  `local_ref.py --pkg-dir`.
- `code/inference/v2_stack_assembly/`: how v2 was assembled and gated from the LB-won column probes (`run_local.py` gates G1-G5,
  `build.py` = the Q6 re-fit on the new Contusion).

## 4.4 Runtime (the 540-minute limit)

Scored runtimes (minutes from acceptance):

| submission | models | minutes |
|---|---|---|
| 0.961 fold package (folds 0-4 + fusion) | | 262 |
| v2 fold package 57416173 | | 205 |
| Contusion all-4 probe | v2 + 18 bone copies | 329 |
| Contusion sxm probe 57480087 | | 499 |
| all-data s5 (54 models) | 54 | 212 |
| folds + all-data (108) | 108 | 477 |
| safe3 57497980 | 81 | 190.5 |
| final118 57497935 | 118 | **timed out** |
| P1 57517691 | 93 | 176 |
| P3 57520062 | 117 | 315 |
| P5 57521751 | 130 | 526 |
| P2 57520594 / P4 57520478 | 87 / 120 | **timed out** |

What we learned:
- **The Kaggle machine, not the package, decides most of the runtime.**
  - The same 118-model package ran a 62-study check in 1,273 s on one VM and 2,383 s on another.
  - On slow VMs every study fully prefetched into RAM still waited in Basin. So the slow part is CPU (DICOM decode / resample),
    not storage, and read-ahead cut those waits by only ~20 %.
  - An 87-model package timed out on deadline day (P2), while a 117-model one finished in 315 min (P3).
- On a fast VM an extra model copy costs ~0.6 % of the base runtime (118 vs 81 models = 1.21x on the same VM).
- Short benches underestimate crop stages ~3x. Estimate from SCORED runs, and keep a margin of at least 2x to 540 min.
- Submit every final candidate EARLY. A timeout only costs that slot, and only scored submissions can be selected. On the last day
  the commit queue also waited 26-50 min before running.
- A submission needs a COMPLETE commit run of its kernel version (the API returns 400 otherwise). Kaggle runs at most 2 batch GPU
  sessions (commits) at once. Scoring runs don't count toward that limit; 4+ scored in parallel.
- Disk: `/kaggle/working` is 20 GB. A 100 GB write to `/tmp` killed the container (the exact limit is unknown). The 6.2 GB weight
  copy is proven.

## 4.5 Reading the public LB

- The public score shows 3 decimals. Kaggle's own sort (`competition_submissions(sort=SUBMISSION_SORT_BY_PUBLIC_SCORE)`) orders
  equal scores by the hidden full-precision value (`lb_order.py`).
- Downloading the team leaderboard and finding our team's position inside its 3-decimal block gives a band estimate.
- Noise: a single-class probe resolves about +-0.012 AUC for that class. A full model redraw moves public by +-0.0004-0.0005, so
  gaps between differently seeded packages under ~0.0005 are not readable.
- Probe one column at a time on a fixed base. Per-column gains add (LB = mean of 12 per-class AUCs), so merged packages were built
  from single-column LB wins.
