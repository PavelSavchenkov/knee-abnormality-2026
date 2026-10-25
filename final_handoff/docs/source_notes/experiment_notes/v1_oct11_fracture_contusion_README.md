# oct11_fracture_contusion: Fracture / Contusion on pipeline #1

Started 2026-10-11 22:10 (user goal, 24 h; user_directives.md). Time-ordered results: ledger.md; ideas and priorities: ideas.md;
Kaggle probe designs: probes.md (no submission without the user's go).

## Question

How to raise pipeline #1's (main effb0 R2, LB 0.940) Fracture and Contusion AUC on the expert-labelled test? Public Fracture
0.908 (2nd weakest finding); Contusion unprobed (~0.96).

## Metric (scripts/fc_eval.py, scripts/compare.py)

Pooled out-of-fold, epoch 19, gold = mean logit of the fold models. Cell sets: text (v1 meets vs denied / normal), tn (vs all
trusted negatives), hard (contusion: traumatic vs degenerative oedema; fracture: fracture vs oedema without fracture), plain
(plain / avulsion fractures), v3 (v3 qualifying vs negative), gold-58 (18 F / 19 C positives; direction). Single models are
compared with the 3 R2 references on the same folds and studies; an N-member blend with every N-subset blend of the references
(--blend). Noise (3 folds, near-R2 runs): sd F text 0.005, F plain 0.0075, F gold 0.0085, C text 0.003, C gold 0.004.

## Layout

- main/: pipeline #1 copy (train.py / data.py / model.py / labels.py with tgt_csv); bone expert (bone_data.py, bone_train.py).
- scripts/: fc_eval.py, compare.py, scan_runs.py, gold_errors.py, site_patterns.py, make_targets.py, bag_viz.py,
  loc_data.py / loc_train.py / loc_infer.py (localiser v3 = v2 + femur / tibia), run_queue.sh (queue.txt, slots_max.txt), watch.sh.
- data/: tgt_{typed,v3,mix}.csv (F / C target variants), loc_128 (localiser windows), loc_pred_v3 (localiser predictions).
- refs/: OOF + gold of the 3 R2 references; out/: tables and images; runs/, logs/.

## Final state (2026-10-12 18:56; supersedes the candidate sections below; details ledger.md 14:40-18:55)

**Final candidate** = submissions/oct12_fc_3pipe (package built and checked locally; no Kaggle step, user 17:12): the 0.955 package
(oct11_oa_3pipe) with only Fracture / Contusion changed by the bone experts b_v3 (effb0, v3 targets), b_lr3 (effb0, v1 targets),
b_nf (ECA-NFNet-L0, v3 targets, lr 3e-5), folds 0-2, 64 windows per bone; recipe out/fc_recipe.json (scripts/make_fc_recipe.py):
Fracture = 0.1 x the 0.955's F logit + 0.9 x the rescaled expert mean, Contusion = the rescaled expert mean (replace).
Folds 0-2 OOF vs the 0.955's own F / C: F text 0.9621 -> 0.9710 (+8.9e-3 [+0.6, +20.1]), C text 0.9596 -> 0.9763 (+16.7e-3 [+9.6,
+24.2]), C hard +18.7e-3 [+8.8, +29.2], F hard +12.5e-3 (n.s.); leave-one-fold-out gains positive on every fold. Local gates: 10 other
columns identical, members vs experiments corr >= 0.9998, 0 failed studies; stage time 0.68 s / study locally (Kaggle est. up to
~5 s / study = ~115 min for 1,300 studies, unmeasured; k_eval 32 or no NFNet are cheaper variants).

What 2026-10-12 added (moderate evidence unless noted):
- Eval-time window budget: k_eval 64 per bone instead of 32 (training k 8) = free contusion gain (+1.7e-3 C text in D, CI above 0;
  5 folds); all windows (~k 128) no better.
- Encoders: ECA-NFNet-L0 needs lr 3e-5 in this expert (3e-4 / 1e-4 stay at the prior, also fp32 / 224 px; OA's NFNet used 3e-4);
  then it is the best single expert on fold 0 and adds in the blend (pre-registered rule, folds 0-2). RadImageNet R50 (mono, lr 3e-4,
  fold 0): no gain in D. V2-S / ConvNeXt-T / NFNet-L1 screens queued but not run (GPU master OA Pseudo: no gap before 22:10).
- Member cuts: the two effb0 experts carry the gain; m_fs adds only fracture 'hard'; one dual-head expert (b_dual, v1 + v3 heads)
  + m_fs ~ D (4 folds) = the 2-model option; a single model loses ~0.005-0.007 fracture text.
- Closed (no gain detected): avgmax pooling (4 folds), soft v3 targets (1 fold), snapshot averaging (5 folds), longer training
  (flat curves), Z1 = R2 on FS-only windows as an m_fs stand-in (5 folds, text down).

How to continue: run the queued encoder screens (queue.txt #AFTER_GAP; ~25 min each) and extend a winner to folds 1-2, then rerun
scripts/make_fc_recipe.py (add its set to SETS) and submissions/oct12_fc_3pipe/build.sh + run_local.py. For the combined package:
copy fc_stage.py, the "fc" config block, the fc_* weights and fc_loc.pt; the stage needs knee_infer.py / model.py / basin_flipdet.py
and flipdet.pt of this toolchain. New scripts: make_fc_recipe.py, copy_pkg955_refs.py (refs/pkg955), bone_eval_k.py +
auto_eval_k.sh (out/eval_k), reduce_members.py, dual_report.py, split_heads.py, screen_report.py.

## Earlier candidates (06:32; details ledger.md 05:02 / 05:28 / 05:56; probe designs probes.md, no submission without the user's go)

Members (pipeline-1 effb0, 20 epochs, epoch 19, 5 folds each): R2 (the 0.944 package's F / C models), m_fs = main trained and
evaluated on fat-suppressed series only, b_v3 = bone-crop expert (localiser v3 femur / tibia / patella crops ~0.4 mm/px, v3 targets,
lr 3e-4), b_lr3 = the same expert with v1 targets. Blends = z-scored logit means, equal weights (out/fc_config_*.json).

vs the 0.944 package's F / C columns (5-fold OOF, gold test-like):

| | F text | F hard | F gold | C text | C hard | C gold |
|---|---|---|---|---|---|---|
| package (R2) | 0.957 | 0.817 | 0.917 | 0.960 | 0.892 | 0.953 |
| A = R2 + b_v3 | 0.965 | 0.818 | 0.932 | 0.970 | 0.913 | 0.953 |
| B2 = m_fs + b_v3 | 0.9705 | 0.829 | 0.9375 | 0.970 | 0.916 | 0.954 |
| **D = m_fs + b_v3 + b_lr3 (recommended)** | **0.972** | **0.846** | 0.929 | **0.972** | 0.914 | **0.9555** |
| C = R2 + m_fs + b_v3 | 0.969 | 0.832 | 0.929 | 0.970 | 0.912 | 0.957 |

Diversity check (vs same-size blends of R2 seeds, 5 folds): B2 F text +0.007 / gold +0.025, C text +0.006 / gold -0.001; held-out
folds 3-4 keep the signs. D vs 3-seed R2 blends: 5 folds F text +0.008 / hard +0.021 / gold +0.019, C text +0.0075 / gold
-0.001; held-out folds 3-4 F text +0.003 / hard +0.040 / gold +0.032, C text +0.004 / gold +0.008 (most robust candidate). Test-time paths verified on gold-58 (deploy/bone_stage.py: expert corr 0.9994 /
0.9997, FS-only main corr 0.9999). Per location the expert fixes R2's patella weakness (patellar fractures 0.81 -> 0.85-0.94).

Caveat: the v3 labels' prompt tables were written after seeing gold-58 summary statistics, so gold-58 gains of v3-trained members
(b_v3 and every candidate using it) may be optimistic; the v1 text cells are independent of that.

Significance of D (paired study bootstrap, 5 folds, x1e-3 [95%]): vs the package F text +14.9 [+5.8, +25.1], C text +12.8 [+6.7,
+19.4], C hard +21.9 [+11.9, +32.7]; vs the 3-seed R2 blend F text +7.7 [+2.9, +13.1], C text +7.5 [+1.8, +13.7]; F hard and
gold-58 positive but not significant. Site-robust (F wins 8 / 10 sites, C 12 / 18; site-centred AUC up). deploy/README.md lists the
package recipe (weights, constants out/fc_config_D.json, fc_combine.py, gates).

Not adopted (3 folds): v3 labels as a plain replacement in the main (m_v3: F gold +0.002, C text -0.003); expert at lr 1e-3
(broke the pretrained encoder). Weak / mixed: post-hoc FS-only inference of R2 (gold +, fracture text -).

## Results so far (23:45; details and evidence strength in ledger.md)

No GPU training yet (the GPU belongs to the Synovitis session; slots expected ~00:30-01:00). CPU work:

| finding | evidence (strength) |
|---|---|
| Baseline R2 (3 refs, 5 folds) | F text 0.956-0.966, gold 0.904-0.924; C text 0.958-0.962, gold 0.947-0.958 (solid) |
| Weakest cell sets | F 'hard' (fracture vs oedema without fracture) 0.82; patellar fractures ~0.87; patella-only contusions ~0.81 (solid, post-hoc) |
| Gold-58 review | readers count impaction / SONK reports as contusion, fold bruises into plain fractures ~half the time, add fractures to silent pivot-shift knees (posterolateral plateau impaction, seen on zoomed crops) (descriptive, 18 / 19 positives) |
| Sites | Greek sites report "microtrabecular insufficiency fractures" for oedema (2x fracture rate, lowest model AUC); English Aera / Achieva / Ambition X leave bruises unsaid; small patellar / plateau contusions are the low-scored positives (descriptive) |
| Labels | v3 vs gold as predictions: F 0.913 / C 0.856 (v1 0.811 / 0.803; optimistic); old runs: steven labels F plain +0.022, C text +0.013, gold + (1-2 folds, weak); soft > sharp / hard targets for C |
| Old runs scan | fracture likes capacity / detail (effb2 f0 gold +0.062, CoAtNet-288, sattn, sidefix; single folds, weak); erase mildly hurts C (12-epoch screens, weak); noise / k48 / pooling variants hurt |
| Post-hoc de-coupling | hurts (research session; solid) |
| Localiser v3 | femur / tibia / patella series boxes, median error 1.5-3.6 mm (validation fold 0) |
| Bone expert | code + bags ready; CPU proxy learns (3 epochs, tiny model); test-time path verified on gold-58 (corr 0.9993-0.9994) |

GPU queue (queue.txt): m_v3 (v3 baseline) and b_base (bone expert) interleaved on folds 0-2, b_v3, m_v3_s1 (v3 seed pair),
m_typed, then one-fold expert screens (FS-only contusion heads, fracture-site heads, erase 0, knee context bag, avgmax, init).

## How to continue

- GPU arms: add lines to queue.txt (`main NAME FOLD key=value ...` -> main/train.py; `bone NAME FOLD key=value ...` ->
  main/bone_train.py), set slots_max.txt to 1 once the GPU owner allows, run `bash scripts/run_queue.sh` (re-reads queue.txt; never
  stop it while an arm runs). Pending: b_v3pat f1 (pat_margin 16), b_v3knee f2, b_v3amax f0.
- Scoring: `scripts/batch_report.py --single RUN --pair A+B --z` (vs the R2 refs or `--base` replicates), `scripts/blend_table.py
  MEMBERS --max 3 [--folds ...]`, `scripts/subgroups.py`, `scripts/site_robust.py out/fc_config_X.json`, `scripts/make_fc_config.py
  NAME MEMBERS` (5-fold constants + package-level panel).
- Deployment: deploy/README.md (stage, verify.py, fc_combine.py). Report page: https://claude.ai/artifact/9dJ3qX9CgbYAXv39aJFArS.
