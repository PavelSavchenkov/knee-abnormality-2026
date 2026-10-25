# oct9_baseline_2nd: a second baseline, different from the main one

Goal (user, 2026-10-09 22:45, 14 h):
- Get as much quality as possible from a pipeline whose errors are not correlated with the main baseline
  (experiments/v0/oct7_baseline).
- The main criterion is blend potential with the main baseline on at least some classes.
- If this model is clearly better on its own, low correlation matters less, but it is still preferred.

## Start here (map for later sessions, written 2026-10-10)

History:
- 2026-10-09 ~16:00-20:00 triage. Why was Basin (experiments/v0/oct8_baseline_v2, DINOv3 ConvNeXt-T) behind the main
  baseline? T* runs on fold 3; see notes/triage_gap_reasons.md and notes/attention_findings.md.
- 20:00-22:40: encoder pair, frozen-feature encoder bench, NFNet regime (BB_*, N_e12, N_lr15, N_base); see
  notes/ledger_until_2240.md and notes/plan_2026-10-09.md.
- 22:45 -> 2026-10-10 ~12:00 (user goal, user_directives.md): tune the NFNet-L0 pipeline. Logged in ledger.md:
  - clean-fold controls;
  - TTA density;
  - random triplet positions (N_pos, N_pos20, P_base);
  - the lever screens on the pos base (P_*);
  - the fold-0 hold-out (P_final);
  - the start of the encoder grid (S6_*).
- 2026-10-10 12:50: work with other backbones moved to experiments/v0/oct10_baseline_3rd (user), which found the 3rd
  baseline (RadImageNet R50, centre slice) and holds the 3-pipeline ensemble analysis. This folder stays the
  NFNet-L0 baseline.

Reading order:
1. **learnings.md**: consolidated findings and the per-class blend table. Read it first.
2. **This README**: final recipe, protocol, safety.
3. **ledger.md**: every arm with numbers and decisions, in time order (sections start "## HH:MM").
4. **ideas.md**: all ideas with priority / status.
5. **user_directives.md**: every user instruction for this goal.
6. **notes/**: the pre-goal triage.

Runs (runs/<arm>/fold<k>/), all NFNet-L0 unless noted:
- N_base f1-4 (grid recipe, e10, TTA oof.parquet): clean-fold controls.
- N_pos (f3), N_pos20 (f1), P_base (f2, f4): the random-positions recipe.
- P_final f0: the hold-out. P_final f3 (2026-10-10 18:36, run by the 3rd baseline's runner): the last fold, so the final
  recipe has all 5 folds at e14 (P_final f0 / f3, N_pos20 f1, P_base f2 / f4). 5-fold OOF 0.9737 / 0.9543.
- P_* (mir, sub, T12, dp2, ra15, hyb, l1, cut, stack, 8slot, lr45, lr2, bs16): lever screens on the pos base. Each has a
  same-fold control: N_pos20 f1, P_base f2, P_base f4.
- BB_nfnet, N_e12, N_lr15, N_base_rawW (f3): regime runs. S6_nf_3e-4 f1: 6-ep screen anchor.
- T0-T9 (f3): triage arms. Backbone DINOv3 ConvNeXt-T, except T5 / T6 (EfficientNet-B0).
- Moved to oct10_baseline_3rd on 2026-10-10: BB_v2s, S6_v2s_5e-4, S6b_v2s_5e-4 (EfficientNetV2-S).

## Rules

- Labels: only the v1 text labels the main baseline used (v1 statuses / probabilities).
- No image-read labels. No pseudo-labels. No anatomy masks (user, 22:50).
- No ensembling, except blend checks against the main baseline.
- Save every evaluated checkpoint. With EMA, save both the EMA weights (model_e*.pt) and the raw weights
  (model_raw_e*.pt).

## Code and data

src/:
- A copy of Basin's pipeline (experiments/v0/oct8_baseline_v2/src), via the triage copy in work/basin_triage/src.
- Pipeline: crops for coronal / axial, region attention readout (battn), 8-triplet grid, EMA.
- Triage additions:
  - head `hattn` = battn + a global per-finding attention path;
  - head `gattn`;
  - `--stop_epoch` (stop part-way through a schedule);
  - `--eval_raw` (also evaluate the raw weights);
  - raw-weight checkpoints.

Data links (data, not code):
- cache/ -> experiments/v0/oct8_baseline_v2/cache (s320 / s352 / s448 caches, folds_k5_s0.parquet).
- weights/ -> Basin's weights folder.

scripts/:
- report_arms.py: the standard per-arm report:
  - trusted / lookat / gold;
  - per-finding and group deltas;
  - Spearman correlation with our runs;
  - blend gain of ours + arm over ours + another of our seeds.
- main_ref.py: read-only access to the main baseline's out-of-fold predictions.
- encoder_bench.py: encoder invariance bench on frozen / fine-tuned weights.
- basin_memorise.py: backbone vs head memorisation.
- basin_attention_probe.py, attention_compare.py: readout attribution.
- diversity_vs_quality.py, early_predict.py.

Runner and records:
- run_queue.sh: one arm at a time (slots_max.txt); re-reads queue.txt every loop; logs to logs/events.txt.
  - Never stop it while an arm runs: stopping it kills the arm.
  - D_NAME FOLD EPOCH lines run the 5-view slice-offset TTA (src/deliver.py).
- runs/<arm>/fold<k>/: config, metrics (curve, EMA and raw_*), preds(.parquet / _raw), checkpoints, TTA oof.parquet.
- notes/: triage findings (why Basin lost), attention probes, the plan, the ledger up to 22:40.
- ledger.md: running record of arms, results and decisions.

Run:

    PYTHONPATH=. conda run -n rsna python experiments/v0/oct9_baseline_2nd/src/train.py --name X --fold 3 --seed 1 <BASE flags> ...
    bash experiments/v0/oct9_baseline_2nd/run_queue.sh

## Final NFNet recipe (2026-10-10)

- Basin best regime: battn + crops, 5 slots, RandAugment N 2 per triplet, affine 15 / 0.15 / 0.08, warmup 3, EMA 0.999.
- Encoder: ECA-NFNet-L0 (eca_nfnet_l0.ra2_in1k), lr 3e-4, head lr 1e-3, batch 8.
- Sampling: random triplet positions every epoch (`--sample pos --uniform 1 --T_eval 24`); dense 24-position eval.
- Schedule: the 20-epoch cosine stopped at e14. That is the median-of-best epoch; it was pre-registered and confirmed on
  the fold-0 hold-out.
- 4 clean folds at e14 vs the main baseline at its fixed e19:
  - +0.0024 trusted / +0.0033 lookat;
  - corr 0.862 (our seeds: 0.94);
  - blend sum10 +32.8. The per-class blend table is in learnings.md.
- Screened on top of it and not adopted (see learnings.md):
  - mirror, series swap, T 12, drop_path, RandAugment M 1.5;
  - cutout + hybrid readout;
  - 8 series, 1 transformer layer;
  - lr 2e-4 / 4.5e-4, batch 16.
- Queue line: `P_final F 1 --backbone eca_nfnet_l0.ra2_in1k --grad_ckpt 1 --eval_raw 1 --save all --eval_every 2 --lr 3e-4
  --sample pos --uniform 1 --T_eval 24 --stop_epoch 16`

## Safety (RAM / VS Code)

- Every training and TTA job runs inside `systemd-run --user --scope -p MemoryMax=24G`, one arm at a time.
- Ad-hoc heavy jobs (smoke tests, bench, analysis) also run in a scope (MemoryMax=16G).
- memguard.sh (background): if MemAvailable < 10 GB or memory PSI full > 20%, it kills this folder's python jobs (only these)
  and logs to events.txt.
- Basin's loader drops each cache file from the page cache after reading.

## Measuring progress (apples-to-apples)

- scripts/compare_pipelines.py ARM --selected <folds where ARM's recipe was chosen>.
- Main baseline: its 3 reference runs at their median-of-best epoch over 5 folds. That is the last epoch for all three,
  so it is not selection-inflated.
- 2nd baseline: 5-offset TTA (eval density equal to the main baseline's 96 windows), at an epoch fixed in advance or its
  own median-of-best over its folds.
- Only folds not used to choose a recipe count as evidence that it is on par or better.
- Ledgers: ledger.md (runs), ideas.md (ideas with priority / status; updated on every user message), learnings.md.

## Fold use

- Fold 3: initial choices (encoder, lr, stop point).
- Folds 1 / 2 / 4: screening blocks, each with its own control.
- Fold 0: never screened. It is held out for the final confirmation of the chosen recipe.

## Screening protocol

- Screens on the grid recipe (until ~01:00 on 10-10): the e10 prefix of the 20-epoch schedule. On the pos base: the e14
  prefix (eval at e7 / e14, ~26 min). EMA + raw eval, all checkpoints saved.
- Screening folds rotate by block, and each block has its own control (N_base on that fold).
- Winners are confirmed on other folds.
- Noise: two runs of N_base differ by 0.0014 trusted / 0.0069 lookat at e10 (lookat sd across seeds ~0.0034).
  - Single-screen differences under ~0.003 / ~0.005 are noise.
  - Use the per-finding pattern, group means and stacking.
