# oct14_all_data: retrain the 0.961 package on all training data

Started 2026-10-14 ~20:30 BST (user directive, user_directives.md). GPU master until "GPU BACK" is sent to the peer sessions.

## Status 10-16 02:40 (details: ledger.md)

- Step-count decision (user, hidden-score order of two LB probes): the fold runs' STEP count (A) > their epoch count (B, ~1.25x
  steps), both 0.949 displayed (all-data 3x3; fold-0 3x1 control 0.947, 9-model fold 3x3 0.948).
- Trained at the fold runs' step count:
  - A1-A3 = 3 all-data seeds of all 14 configs of the 0.961 package (42 models).
  - A4-A5 = seeds 4-5 of the six configs with 5 folds in the 0.961 package (12 models). 54 all-data models in total.
  - Plus 3x3 B1_* (the 1.25x-steps arm).
  - Every checkpoint + optimizer saved; scripts/audit.py PASS on all runs.
- Packages (the 0.961 fold package oct13_f5_win: 54 models, LB 0.961):

  | package | models | LB |
  |---|---|---|
  | submissions/oct14_all_data_s1 | 1 seed per config (14) | 0.960 (57213373) |
  | submissions/oct15_all_data_s2 | 2 seeds (28) | 0.961, above the fold package by the hidden score (57225427) |
  | submissions/oct15_all_data_s3 | 3 seeds (42) | 0.962 (57227263) |
  | submissions/oct15_all_data_s5 | 5 / 3 seeds = the fold package's counts (54) | **0.962**, above the 3-seed package by the hidden score: NEW BEST (57235072) |
  | submissions/oct15_folds_plus_all_data | 54 fold + 54 all-data (108) | 0.962 (57239412), 477 min scoring |

- GPU FREE sent to Master at 14:53 (runner / watchdog / memguard stopped, gate 0).

## Final analysis (10-16 02:40)

LB (public; "+" = above the same displayed score by the hidden score, user's Kaggle UI):

| package | models | LB | scoring |
|---|---|---|---|
| 0.961 fold package (oct13_f5_win) | 54 fold | 0.961 | 262 min |
| all-data 1 seed per config | 14 | 0.960 | 95 min |
| all-data 2 seeds | 28 | 0.961+ | 143 min |
| all-data 3 seeds | 42 | 0.962 | 211 min |
| **all-data 5 / 3 seeds (oct15_all_data_s5)** | 54 | **0.962+ (best)** | 212 min |
| fold + all-data (oct15_folds_plus_all_data) | 108 | 0.962 (order vs s5 open) | 477 min |

- At equal model count (54), all-data seeds beat fold models by ~+0.001, at the fold runs' step count; 1.25x steps did not help
  (3x3 probes A vs B).
- Adding seeds keeps helping in small steps (2 -> 3 -> 5 / 3 seeds), below the 3-decimal resolution after 3.
- The 108-model mix sits at the midpoint of its halves (275 check studies: |dlogit| 0.049 to each, halves 0.091 apart), scores =
  the all-data half at 3 decimals and costs 2.25x its scoring time (63 min under the 9 h limit).
- gold-58 did not rank these packages (2 seeds 0.9513 > 3 seeds 0.9500 on gold-58, LB the other way); use the LB.
- For future (e.g. label-changed) models: train all-data seeds at the fold runs' step count in the fold package's counts and deploy
  them alone; mixing in fold models showed no gain for double the runtime.

## Plan (changed 20:55-21:05 by the user, ledger L7)

Tonight: all-data 3x3 without experts (main effb0 / NFNet-L0 / RadImageNet R50, the oct10_3pipe_3f recipe) at two step counts, one
training job at a time.
- A = the fold runs' step count (A1_*).
- B = the fold runs' epoch count at 100% data, ~1.25x steps (B1_*).
Then 3 user-confirmed submissions before 01:00: review 29's fold-0-only 3x1 control of the 0.948 package, submissions/oct14_ad3x3_A, submissions/oct14_ad3x3_B.
The experts' all-data runs (step 1 below) and the fold-model screens (step 2 below) are on hold / dropped.

Original plan (20:30):

1. **Step 1:** one all-data model for every model config of the 0.961 package (submissions/oct13_f5_win). 14 configs, each at
   the fold runs' step count (schedule in the fold runs' step units). Then build the package with each config's fold models
   replaced by its one all-data model, gate it locally and on Kaggle (prompts/submission.md), and make ONE official submission.
2. **Step 2:** screens on more encoders: "does more data need more training steps, or is the same step count enough?". If more
   steps clearly give no gain or lose, keep the fold runs' step count.
3. **Step 3:** 2 more all-data seeds of every config (3 more if step 2 selects more steps).

Background: work/seeds_vs_folds/README.md. Fair inner-world read: one all-data model per backbone ~ 3-4 fold models; estimated
LB gain for 3 all-data seeds vs folds 0-2 ~ +0.002. The PL data x steps test (main, fold 3, 1 seed per cell) found no detected
gain from more steps at fixed data, with lookat / gold leaning positive within noise.

## Configs (source = the deployed fold runs; recipe = the source config.json, copied exactly by scripts/launch.py)

| config | kind | source run | deployed folds / epoch | spe_ref (fold runs' mean steps/epoch) | all-data stop |
|---|---|---|---|---|---|
| main sub_b0 | main | v0/oct7_baseline/runs/sub_b0 | f0-4, last (e19) | 434 | 20 x 434 = 8,680 steps |
| main m_prob05 (synovitis head) | main | v1/oct10_synovitis/runs/m_prob05 | f0-2, last | 435 | 8,700 |
| NFNet | basin | v0/oct9_baseline_2nd/runs/P_final (+ N_pos20 f1, P_base f2 / f4: same recipe) | f0-4, e14 of 20 | 434 | 14 x 434 = 6,076 (cosine over 8,680) |
| R50 | basin | v0/oct10_baseline_3rd/runs/F20_rad50m_5e-4 | f0-4, e20 | 434 | 8,680 |
| NFNet n_prob05 (synovitis head) | basin | v1/oct10_synovitis/runs/n_prob05 | f0-2, e14 of 20 | 435 | 6,090 (cosine over 8,700) |
| OA v2cor effb0 | oa | v1/oct10_oa/runs/oa_crop_v2cor | f0-4, ep19 | 434 | 8,680 |
| OA v2cor NFNet | oa | v1/oct10_oa/runs/oa_crop_v2cor_nfnet | f0-2, ep19 | 435 | 8,700 |
| bone b_v3 / b_lr3 | bone | v1/oct11_fracture_contusion/runs/b_v3, b_lr3 | f0-4, ep19 | 434 | 8,680 |
| bone b_nf | bone | v1/oct11_fracture_contusion/runs/b_nf | f0-2, ep19 | 435 | 8,700 |
| synovitis crops s_fs / s_fc125 / s_fs_nf / s_fc125_nf | syn | v1/oct11_synovitis_crops/runs/* | f0-2, ep19 | 435 | 8,700 |

All folds share one split: 4,349 training studies, fold training sets 3,460-3,494 (432-436 steps / epoch at batch 8). An
all-data run has 543 steps / epoch; with spe_ref its warmup, cosine and epoch-based stop are in the fold runs' step units, so it
takes the same step count and LR curve as a fold run, over 1.25x more studies (each seen ~16 times instead of 20).

## Layout

- main/, basin/, oa/, bone/, syn/: trainer copies (sources in each docstring) with three patches. Fold -1 = all-data run
  (every non-gold study trains; validation = gold-58). spe_ref (+ basin --stop_step) = equal-step schedule. Optimizer state is
  saved with every checkpoint (basin already saved EMA, raw and opt).
- scripts/launch.py: runs one arm from a source config (exact recipe + path remaps + overrides). scripts/smoke.sh: 20-step tests.
- run_queue.sh (systemd unit alldata-runner, restart_runner.sh), queue.txt, gpu_go.txt, slots_max.txt, watchdog.sh, memguard.sh,
  status.sh.
- data/, loc/: copies of the targets / localiser outputs the trainers read; cache -> v0/oct8_baseline_v2/cache (Basin slices);
  weights/ (RadImageNet R50); runs/_bags: copied bag caches.
- ledger.md: decisions and results in time order.

Checkpoints: every eval epoch (every 2 epochs, from epoch 1) saves weights and optimizer state (opt_ep*.pt / opt_e*.pt).
Basin also saves raw weights (model_raw_e*.pt) next to the EMA weights.
