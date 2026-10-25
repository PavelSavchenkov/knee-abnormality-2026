# Kaggle submission pipeline (rsna-knee-abnormality-detection)

Toolchain: the newest submission folder, now `submissions/oct10_3pipe_3f/` (git-ignored, like all of `submissions/`: do not delete
past folders; `oct7_baseline_5_folds/` is the original). Start a new submission by copying its SCRIPTS (not `package/`, `kernel/`,
`kaggle_output/`, `local_check_*`, `*_nodet*`, `*.json`, `*.log`; do copy `basin/`) to `submissions/<date>_<name>/` (never import across folders), then replace
the old folder name and slug in every script (docstrings and argparse defaults of `build_package.py`, `make_kernel.py`, `run_local.py`,
`compare_kaggle.py`, `describe.py`, `submit_and_time.py`). Its `knee_infer.py` is the fast pipeline (~7-12 min est. for 5 effb0 models)
and supports image-head models (13 outputs, `synovitis_ext_w`) and mixed packages (Synovitis from the 13-output models, the other 11
findings from the 12-output models). Its `ensemble.py` (the package entry since oct10) blends pipelines: main = knee_infer, plus the
Basin-code pipelines of the 2nd / 3rd baselines (`basin/basin_predict.py`: NFNet-L0, RadImageNet-R50, dense position-aware view, Basin's
mirror detector) as sigmoid(mean of pipeline logits); a package without `--basin` models is main only. Speed notes: `prompts/optimization.md`.

| script (in the folder) | purpose |
|---|---|
| `knee_infer.py`, `model.py` | inference from raw DICOM; runs on Kaggle and locally; options from the package `config.json` |
| `ensemble.py`, `basin/*.py` | 3-pipeline entry (`ensemble.predict`) and the Basin-code inference (copied flat into the package) |
| `check_detector.py` | the mirror detector on fold 1 from DICOM, off vs on vs the runs' OOF (Basin's protocol) |
| `build_package.py RUN:FOLD ... [--syn-w W] [--syn-ext-w W] [--tta-draws N] [--basin PIPE=PATH ...] [--detector PATH]` | Kaggle dataset dir `package/`: code, weights, config (post-processing, image head, TTA), check list |
| `make_check_set.py` -> `check_studies.csv` | the 275 train check studies (58 gold, 100 random, 117 edge cases) |
| `check_preprocess.py` | CPU DICOM path == training cache `data/cache/fov160_px400`, byte for byte |
| `speed/check_gpu.py` | GPU preprocessing vs the CPU path (<= 1 grey level); only in `oct7_baseline_5_folds/speed/`, copy it when preprocessing changes |
| `run_local.py` | the package on the check studies locally -> `local_check_pred.csv` |
| `make_kernel.py [--keep F ...]` | submission notebook `kernel/`; its commit run also predicts the check studies; `--keep`: probe mode |
| `compare_kaggle.py` | Kaggle vs local predictions on the check studies + runtime estimate -> `kaggle_check.json` |
| `describe.py` | submission description in the strict format -> `description.txt` |
| `submit_and_time.py VERSION` | submits the checked kernel version; times Kaggle's acceptance -> scored (10 s polls) -> `submit_timing.json` |

Edit `FEATURES` in `describe.py` for each submission. Exact submitted code of each past submission: its folder's `package/`, `kernel/`.

## Steps (each GATE must pass before the next step)

0. Before any work on a package:
   - GATE: the models passed their experiment check (5-fold eval vs same-config references, gold veto) and it is in the ledger.
   - Write the folder `README.md`: the question this submission answers, the local evidence, and how to read the LB (which score
     means what), BEFORE submitting. A "replace one column" design (a past submission's models for the other findings, only the
     tested finding changed) reads that finding's change as 12 x (LB - past LB), +-0.012, and is a real LB entry; a probe
     (`--keep F`) reads one finding's absolute AUC +-0.006 but scores ~0.53.
   - GATE: the user's explicit go for this submission. Uploading a dataset, pushing a kernel and submitting are outward actions; a
     relayed message from another session is not the user's go. Check the daily quota (5 / day; `submit_and_time.py` prints it).
1. `build_package.py RUN:FOLD[:FILE] ... [--syn-w W] [--syn-ext-w W] [--tta-draws N]` -> `package/` (code, weights, `config.json`,
   `check_studies.csv`, `dataset-metadata.json`).
   - GATE: `config.json` has the intended `models` / `weights`, post-processing (`synovitis_effusion_w`, `synovitis_ext_w`, `tta`), and
     `dataset-metadata.json` id = `pavelsavchenkov/<slug>` with a NEW slug (never version a dataset with a pending submission).
2. Preprocessing changed? `check_preprocess.py --n 30` must print 0 differences (CPU DICOM path vs training cache `data/cache/fov160_px400`),
   then `speed/check_gpu.py` (GPU path vs CPU path: <= 1 grey level).
3. `run_local.py` (in `systemd-run --user --scope -p MemoryMax=40G`) -> `local_check_pred.csv`, BEFORE uploading.
   - GATE: 0 failed studies; vs the experiment's own predictions corr >= 0.9998, max |prob diff| <= ~0.02 (training cache JPEG + bf16 vs
     DICOM + fp16) and gold-58 equal within ~1e-3. Findings taken from a past submission's models: equal to that submission's
     `local_check_pred.csv` (0.0) and its Kaggle `check_pred.csv` (<= ~2e-4).
   - Then `rm -rf package/__pycache__` (run_local imports the package's code and creates it).
4. Upload: new dataset `kaggle datasets create -p package` (update: `kaggle datasets version -p package -m msg`).
   - GATE: `kaggle datasets status` = ready AND `api.dataset_list_files` shows every file with this upload's timestamp and the local
     size ("ready" can still refer to the previous version).
5. `make_kernel.py --slug S` -> `kernel/`.
   - GATE: `kernel-metadata.json` id `<owner>/<S>-sub`, `dataset_sources` = the dataset just uploaded, `is_private` true, T4,
     internet off; the notebook's `KEEP = []` for a real submission (non-empty only for a probe).
   - `kaggle kernels push -p kernel`; poll `kaggle kernels status` every 30 s. The commit run can wait ~25 min in Kaggle's GPU queue
     (QUEUED is not a submission). User rule: if still QUEUED after ~10 more minutes, `kaggle kernels delete owner/S-sub -y` and push
     again. The commit run sees the 3 sample test studies, so `if len(test) <= 3:` it also predicts the check studies from the
     competition's train DICOM and writes `check_pred.csv`, `check_timing.json`.
6. `compare_kaggle.py --kernel owner/S-sub` -> `kaggle_output/`, `kaggle_check.json`.
   - GATE: Kaggle vs local max |prob diff| < 1e-3 and equal gold-58 AUC; runtime estimate well inside the notebook time limit (today's
     packages: 7-12 min).
7. `describe.py` (edit `FEATURES`) -> `description.txt` (strict format below, <= 500 chars).
8. Submit and time it: `submit_and_time.py N` (N = the checked kernel version). A "system error" seconds after acceptance does not use
   the quota: resubmit. Run long waits (commit-run polling, the timer) detached, e.g. `setsid nohup bash -c '... | grep
   --line-buffered -v Warn > LOG' &`: a background job of a Claude session dies when the session ends (2026-10-08: a chain waiting on
   a commit run was killed before it submitted). After a restart, check `api.competition_submissions` and the folder's
   `kaggle_check.json` / `submit_timing.json` before redoing any step.
9. After the score: fill the folder README's Result table (submission ref, kernel version, dataset, public LB, the reading, scoring
   time, kaggle=local), add a row to "Results so far" below, a ledger entry, update the Claude memory (kaggle-submissions), and tell
   the sessions that depend on it.

## Description format (strict, <= 500 chars; Kaggle limit is 500)

`<submission folder> | OOF trusted <.4f> lookat <.4f> | gold <.4f> | folds <n>/5 (f<list>) ep <E> | <model features, <= 200 chars> | est <min> min | kaggle=local <max diff .0e>`

- OOF: `common.metrics.evaluate` on the pooled OOF of the submitted fold models at the submitted epoch, with the package's own
  post-processing and TTA applied (the experiment's `oof.parquet` alone has neither). Mixed packages: the OOF of the 12-output
  models, which give the 11 findings (trusted / lookat have no synovitis). A model trained on all data: report
  its parent fold models' OOF and write `folds full (parent 5/5)`.
- gold: gold-58 macro AUC of the Kaggle-side predictions of the exact package (gold is never trained on).
- est: notebook overhead + steady s/study of the gold + random check group x 1,300 studies. Noisy (Kaggle NFS speed varies ~1.7x between
  runs) and pessimistic (real scoring took ~0.6-0.9x of it).

## Facts that are easy to get wrong

- Kaggle GPU kernel (2026-10-07): 4 CPUs, 2x T4 (16 GB, no bf16), 15 GB /dev/shm, python 3.13, torch 2.11+cu128, timm 1.0.29,
  pydicom 3.0.2, cv2 4.14, numpy 2.1.3, pandas 2.3.3. Local: pandas 3, cv2 5, torch 2.14: never rely on version-specific behaviour.
  `kernel-metadata.json`: `"machine_shape": "NvidiaTeslaT4"`, `enable_internet: false`.
- Paths: competition data at `/kaggle/input/competitions/rsna-knee-abnormality-detection`, datasets at `/kaggle/input/datasets/<owner>/<slug>`.
  Find them with a depth-limited glob. `train_series/` + `train_series.csv` are mounted too: use them for Kaggle-side checks.
- A kernel slug must differ from the dataset slug (push fails with 409 Conflict): kernel = `<dataset slug>-sub`.
- The CLI has no cancel and shows no logs while a kernel runs. A hung kernel holds a GPU session for up to 12 h:
  `kaggle kernels delete owner/slug -y` stops it (the log is lost). Keep scratch kernels separate from submission kernels.
- T4: fp16 only. Backbone `.half()` + channels_last, head in fp32. fp16 vs local bf16 eval: corr 0.9999, gold equal.
- Fat suppression comes from `test_series.csv` (`Fat_Suppression`); every other input comes from the DICOM headers.
- Exact training preprocessing (rules in `knee_infer.py` docstring): side vote tie -> first series in (SeriesNumber, uid) order;
  duplicate series = same middle-frame pixels, checked before resampling; percentile on `vol[:, ::4, ::4]` of the padded 400 px canvas.
  Verified byte-identical on 275 train studies incl. edge cases (`make_check_set.py`).
- `kaggle datasets create -p DIR` skips subdirectories (default `--dir-mode skip`): keep packages flat.
- Basin-code models were evaluated in bf16; on the T4 they run fp16 (more precise): fp16 vs bf16 alone gives max |dp| 0.04-0.06 per
  model, so compare them with the experiment's predictions in bf16 (match within 0.005) and gate the final blend on gold-58.
- Probing one finding's public AUC: keep its column, set the other 11 to 0.5 (`make_kernel.py --keep F`): AUC = 12 x LB - 5.5, +-0.006.
- A submission that fails with "A system error" seconds after acceptance did not use the daily quota: resubmit.
- 5 submissions per day. Unverified whether scoring uses the dataset version of the commit or the latest one: never version a dataset
  while its submission is pending (use a new slug for experiments).
- RAM (125 GB, VS Code dies at 50% memory pressure): local checks read GBs of DICOM into the page cache. Run them in
  `systemd-run --user --scope -p MemoryMax=40G ...`, use small `--n`, and evict the files afterwards with `posix_fadvise(DONTNEED)`.
  Never read the raw 115 GB cache `fov160_px400/slices.u8` at scale.

## Results so far

| submission | models | public LB | OOF trusted / lookat | gold-58 | scoring time |
|---|---|---|---|---|---|
| oct7_baseline_one_fold (56913909) | effb0 R2 fold 3, ep 19 | 0.931 | 0.9742 / 0.9556 | 0.9134 | <= 15.5 min |
| oct7_baseline_5_folds (56914408) | effb0 5 folds, ep 19 | 0.938 | 0.9714 / 0.9520 | 0.9196 | <= 15.5 min |
| oct7_5_folds_synpp (56915803) | same + synovitis = logit mean(syn, eff), fast pipeline | 0.940 | 0.9714 / 0.9520 | 0.9262 | <= 10.6 min |
| oct7_5_folds_synpp_tta (56919730) | same + TTA: plain + 4 training-like random affine views (MCL plain) | 0.940 | 0.9726 / 0.9533 | 0.9231 | <= 20.5 min |
| oct7_synovitis_probe (56920502) | probe: 0.940's synovitis, other 11 = 0.5 -> public synovitis AUC 0.872 +- 0.006 (gold 0.772) | 0.531 | | | 7.0 min |
| oct8_imghead3_5f (56959328) | 0.940's 5 models for 11 findings + Synovitis from imghead3 x5 (image-read head alone) | 0.944 | 0.9714 / 0.9520 | 0.9329 | 9.9 min (accepted -> scored) |
| oct8_imghead3_single (56962458) | imghead3 x5 alone (11 findings from its main heads, Synovitis from its image head) | 0.942 | 0.9721 / 0.9532 | 0.9273 | 6.5 min (accepted -> scored) |
| oct9_probe_bakers (57018559) | probe: the 0.944 sub's Baker's, other 11 = 0.5 -> public Baker's AUC 0.968 +- 0.006 (gold 0.958) | 0.539 | | | 12.1 min |
| oct9_probe_lateral_oa (57018573) | probe: the 0.944 sub's Lateral OA, other 11 = 0.5 -> public Lateral OA AUC 0.884 +- 0.006 (gold 0.840) | 0.532 | | | 10.1 min |
| oct9_probe_pf_oa (57018624) | probe: the 0.944 sub's PF OA, other 11 = 0.5 -> public PF OA AUC 0.920 +- 0.006 (gold 0.870) | 0.535 | | | 9.9 min |
| oct9_probe_fracture (57018639) | probe: the 0.944 sub's Fracture, other 11 = 0.5 -> public Fracture AUC 0.908 +- 0.006 (gold 0.921) | 0.534 | | | 9.7 min |
| oct9_probe_lateral_meniscus (57018697) | probe: the 0.944 sub's Lateral Meniscus, other 11 = 0.5 -> public Lateral Meniscus AUC 0.944 +- 0.006 (gold 0.911) | 0.537 | | | 10.4 min |
| oct10_3pipe_3f (57051704) | 9 models: main effb0 (0.940's f0-2, syn step) + NFNet-L0 e14 + RadImageNet-R50 e20 (Basin code, mirror detector), folds 0-2, sigmoid(mean pipeline logit) | **0.948** | 0.9778 / 0.9584 (f0-2) | 0.9294 | 43.9 min (accepted -> scored; est 39.0) |
| oct11_oa_e1 (57075026) | 0.944 package, OA x3 = attention compartment-crop OA expert alone (effb0, localiser crops) 5 folds | 0.948 | 0.9748 / 0.9567 | 0.9357 | ~30 min (accepted -> scored; est 46.5) |
| oct11_oa_e2 (57074973) | 0.944 package, OA x3 = lse (instance-pooling) OA expert alone 5 folds | 0.947 | 0.9725 / 0.9536 | 0.9401 | ~30.5 min |
| oct11_oa_e3 (57080608) | 0.944 package, OA x3 = v2cor compartment-crop OA expert alone (coronal slices by tibial cartilage, localiser v2) 5 folds; submitted by the user via the Kaggle UI before the commit run finished | 0.949 | 0.9750 / 0.9568 | 0.9346 | ~29 min (accepted 18:04 -> scored by 18:33; est 45.1) |
| oct11_oa_3pipe (57090414) | 0.952 package for 9 findings + Synovitis; OA = 0.4 x the 0.952 OA + 0.6 x mean of v2cor compartment-crop OA experts (effb0 5f, NFNet-L0 f0-2), + de-coupling (recipe chosen on folds 0-2 OOF) | **0.955** | 0.9795 / 0.9603 (f0-2) | 0.9374 | 77.9 min (accepted -> scored; est 85.9) |
| oct12_syn_g (57112379) | the 0.955 package; Synovitis G = logit mean of image heads (effb0 m_prob05, NFNet n_prob05) + localiser-crop synovitis experts (s_fs, s_fc125 on effb0 + NFNet), f0-2, 18 models | **0.955** (above oct11_oa_3pipe by hidden precision: team LB 71/74 -> 54/83 in the 0.955 block; new best) | 0.9795 / 0.9603 (f0-2) | 0.9385 | 129.2 min (accepted -> scored; est 147) |
| oct13_pl_b0 (57143988) | inner-world B0: 9 honest inner models of outer fold 3 (main effb0 in_m_h3 + NFNet-L0 in_nf_h3 e14 + RadImageNet-R50 in_r50_h3(r) e20, inner folds 0-2, 60% data), 3x3 recipe, no experts | 0.943 | outer-f3 0.9760 / 0.9592 | 0.9259 | 57 min |
| oct13_pl_b0xf (57144850) | inner-world B0 with the clean-label stage xF on main (L6F_m: explicit v1 cells + T3pH synovitis, fracture / contusion all cells) and R50 (L6F_r50 e4); NFNet = B0's | 0.945 | outer-f3 0.9771 / 0.9596 | 0.9278 | 63 min |
| oct13_f5_win (57154196) | 0.957 package (oct12_global_best) + fold 3-4 checkpoints (3x3, bone) + Effusion img_eff blend (w 0.3) + window / crop-level fusion (relevance = mean attention, value = mean window prob, top-k gate): 3x3 Effusion / Baker's (15 models, geo|pm|g20), OA experts (g10), bone experts (g20), synovitis crop experts (per design, maxmix); fixed backbone batch 128 (Kaggle cuDNN) | **0.961** | 0.9809 / 0.9605 (f0-2) | 0.9479 | 262.4 min (est 258.8) |
| oct13_pl_k6 (57158313) | inner-world K6r: B0_xF with main -> PL students (honest 3-backbone teacher T3pH, ct blend) + teacher-confidence stage (L6Fc_ism e0), NFNet -> PL students without teacher synovitis (is_nf_T3pH_ctns e16), R50 = B0_xF's xF stage (L6F_r50 e4); epochs by median-of-best | 0.946 | outer-f3 0.9771 / 0.9603 | 0.9309 | 39.9 min (est 44.6) |
| oct14_pl_k12fkd (57171745) | inner-world K12fkd: K6r with main -> PL students trained with feature KD (feat_kd 0.1: cosine to the honest inner main teacher's window features) + teacher-confidence stage (L6Fc_fkd e0); NFNet / R50 as K6r | 0.946 (below K6r unrounded) | outer-f3 0.9776 / 0.9603 | 0.9325 | 47.3 min (est 40.4) |
| oct14_pl_hyb (57180247) | per-finding hybrid: Fracture / Contusion / Synovitis / Effusion from K6r (main L6Fc_ism e0), the other 8 from K12fkd (main L6Fc_fkd e0); NFNet / R50 shared; small dataset + md5-checked large files from the K12fkd / K6r datasets (slow uplink) | 0.946 (unrounded: K6r > hyb > K12fkd) | outer-f3 0.9776 / 0.9603 | 0.9325 | 39.9 min (est 50.5) |
| oct14_3x1_f0 (57194679) | the 0.948 3x3 with fold 0 only (main sub_b0 f0 + NFNet P_final f0 e14 + R50 F20 f0 e20, syn mix 0.5, detector): N = 1 control for the all-data 3x3 | 0.947 | OOF f0 0.9781 / 0.9564 | 0.9286 | 45.1 min (est 32.7) |
| oct14_ad3x3_A (57195542) | all-data 3x3: 1 model per pipeline (main effb0 sub_b0 recipe + NFNet-L0 + R50) trained on all 4,349 studies at the fold runs' step count; oct10_3pipe_3f recipe | 0.949 (above B by hidden score) | parent f0-2 0.9778 / 0.9584 | 0.9267 | 29.2 min (est 39.0) |
| oct14_ad3x3_B (57197100) | as A but at the fold runs' epoch count (~1.25x steps; NFNet stop e14 of 20 at 543 steps / epoch) | 0.949 (below A by hidden score) | parent f0-2 0.9778 / 0.9584 | 0.9278 | 30.0 min (est 38.6) |
| oct14_all_data_s1 (57213373) | the 0.961 package (oct13_f5_win) with every config's fold models replaced by ONE all-data model (seed 1, fold runs' step count; 14 models instead of 54); recipe constants / fusion specs unchanged | 0.960 | parent f0-2 0.9809 / 0.9605 | 0.9470 | 94.7 min (est 110) |
| oct15_all_data_s2 (57225427) | the 0.961 package with every config's fold models replaced by TWO all-data seeds (fold runs' step count; 28 models); recipe constants / fusion specs unchanged | **0.961** (above oct13_f5_win by the hidden score: new best) | parent f0-2 0.9809 / 0.9605 | 0.9513 | 143.0 min (est 224) |
| oct15_all_data_s3 (57227263) | the 0.961 package with every config's fold models replaced by THREE all-data seeds (fold runs' step count; 42 models); recipe constants / fusion specs unchanged | **0.962** (new best) | parent f0-2 0.9809 / 0.9605 | 0.9500 | 210.5 min (est 215) |
| oct15_all_data_s5 (57235072) | the 0.961 package with its fold models replaced by all-data seeds in the same counts (5 / 3 per config, 54 models, fold runs' step count); recipe constants / fusion specs unchanged | **0.962** (above the 3-seed package by the hidden score: new best) | parent f0-2 0.9809 / 0.9605 | 0.9491 | 211.8 min (est 258) |
| oct15_folds_plus_all_data (57239412) | the 0.961 package averaging its 54 fold models AND the 54 all-data models (equal weight per model per config; window / crop fusions over all 108) | 0.962 | parent f0-2 0.9809 / 0.9605 | 0.9505 | 477.0 min (est 426; limit 540) |
| oct16_probe_loa_splice (57273283) | the 0.961 package exact for 11 findings; Lateral OA from the same OA experts' crop fusion at k_eval 64 (2nd crop pass on the same bags, own align), de-coupled on the deployed MOA / PF (output-level splice) | 0.961 (above oct13_f5_win by hidden precision, API sort by public score: Lateral OA k_eval 64 > 32 on public, 0 < dAUC < +0.012) | 0.9809 / 0.9605 (f0-2) | 0.9489 | 252.9 min (est 305) |
