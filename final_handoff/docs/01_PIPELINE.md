# 1. The final pipeline (best fold-average form)

This is the package our best submissions run. In its **fold-average** form every member is the average of its cross-validation fold
models (3 or 5 folds). The two selected finals (P1 57517691, P3 57520062, both public 0.965) run the same form with every member
retrained on all training data (several seeds instead of folds; see section 6).

- Package code: `code/inference/package_code/` (exactly the files the Kaggle kernel imports).
- Package config: `code/inference/package_config/config.json` (+ `syn_members*.json`, `main_sd/config.json`, the Basin member
  `*.json`, `dec_refit.json` = the post-hoc de-coupling constants).
- Local fold-average build: `code/inference/packaging/fold_pkg.py` (assembles every member's fold models + code + config into one
  folder; needs the trained checkpoints, which are not in this handoff).
- Kaggle kernels: `code/inference/kernels/` (see `04_INFERENCE_AND_KAGGLE.md`).

## 1.1 Inputs every stage shares

- Raw DICOM of one study, all series. Series type (sagittal / coronal / axial, fat-suppressed or not) and position come from DICOM
  headers.
- **Side correction.** Images are put in a canonical orientation (`code/common/geometry.py`). A small flip detector (`flipdet.pt`,
  trained by `code/localiser/flipdet.py`) mirrors back series whose left / right side the header rule gets wrong.
- **Localiser.** A MobileNetV3 slice localiser (`oa_loc.pt` "v2cor", `fc_loc.pt` "v3"; trained by `code/localiser/loc_train.py`,
  distilled from nnU-Net Knee boxes) finds anatomical structures per slice (compartments, cartilage plates, menisci, notch, MCL,
  suprapatellar pouch, Hoffa's fat pad, bones). Crop experts cut their crops ("bags") from its boxes.

## 1.2 Stages

| stage | module | models (fold-average form) | what it produces |
|---|---|---|---|
| main pass | `knee_infer.py` | `sub_b0` x5 (12 findings), `m_prob05` x3 (13 outputs: + image-read Synovitis head) | 2.5D effb0 over 3-slice windows of every series, per-finding attention head |
| main SD pass | `knee_infer.py` (second model set) | `sd1b` x5 | per-series-group heads, used for MCL only |
| Basin pass | `basin_predict.py`, `basin_model.py`, `slots.py` | NFNet-L0 `nfnet` x5, RadImageNet R50 `r50` x5, NFNet image head `n_prob05` x3, NFNet SD `bsd_nf` x5 | a second and third generalist architecture (series slots, centre-slice crops) |
| windows | `win_stage.py` | main + Basin models on every window | per-window exports for the Effusion / Baker's window fusion |
| OA crops | `oa_stage.py` | `oa_v2cor` x5, `oa_v2cor_nf` x3, `oasep` x5, `oasep_nf` x3, `silsep` x5 | compartment crop experts: Medial / Lateral / PF OA + meniscus heads |
| bone crops | `fc_stage.py` | `b3c_rep` x5, `b3l_rep` x5, `b3n_rep` x3, `bfs_rep` x5 (FS-only) | femur / tibia / patella crop experts: Fracture, Contusion |
| synovitis-form crops | `syn_stage.py` | `cs_fs` x3, `cs_fc125` x3 (ConvNeXt-T), `s_fs_nf` x3, `s_fc125_nf` x3 (NFNet), `ef3*` x3 each, `w5b_mcl` x5, `z1_acl` x5, `b3g` x5, `sxm` x3 | crops of pouch / Hoffa / posterior recess (Synovitis, Effusion), MCL zoom, notch zoom (ACL), a 9-bag multi-area model (b3g), bone-bag ConvNeXt (sxm) |
| ensemble | `ensemble.py` | | per-column recipes below, then the post-hoc de-couplings (DEC) |

The FAST runner (`code/inference/fast_runner/fast.py`, embedded in the kernels) runs the same stages on both Kaggle T4 GPUs in
parallel worker processes and is bit-identical to the sequential path (`ensemble.predict`).

## 1.3 Column recipes

Logit = log-odds. "z-align" = rescale a member's logit to the base logit's OOF mean / sd before blending (`mu` / `sd` in config).
"crop-level fusion" = every fold model's logits over the shared crops, a top-k relevance gate, then a mean (`fc_stage.fuse_crops`).

| column | recipe |
|---|---|
| ACL | 3x3 generalist base (mean over pipelines main / NFNet / R50 of the pipeline-mean logits) -> splice the ACL notch-zoom member `z1_acl` (w 0.4, z-aligned) -> splice `b3g`'s ACL head (w 0.25) |
| MCL | base = mean of main-SD (`sd1b`), NFNet-SD (`bsd_nf`) and R50 MCL logits -> splice the coronal MCL-zoom member `w5b_mcl` (w 0.25) -> splice `b3g`'s MCL head (w 0.25) -> post-hoc de-coupling from Contusion (Q6, b 0.2, constants re-fitted on the final Contusion) |
| Medial / Lateral Meniscus | 3x3 base blended with the OA experts' meniscus heads (`men`: w 0.6, z-aligned; members oa_v2cor + oa_v2cor_nf) -> LM only: post-hoc de-coupling from ACL (Q8, b 0.1; constants re-fitted on the final ACL = Q22) |

Post-hoc de-coupling (`dec_refit.json`, `decouple()` in the kernel): `z_out = z - b * (c0 + c * z_context)` on logits. It removes
part of a column that a co-reported finding explains (reports couple MCL to bone bruises and LM tears to ACL tears; the image-based test labels
do not). Constants `c0`, `c` are fitted label-free on non-gold OOF folds 0-2; `b` was chosen on gold and confirmed on the LB. The
exact formula is `decouple()` in the kernel notebook.
| Medial OA, PF OA | 3x3 base blended with the OA expert term (w 0.6) + per-column de-coupling constants (`oa.recipe.dec`); expert term = crop-level fusion (geo pm g10 wmean) of the OA experts |
| Lateral OA | as above, but the expert term is the fusion over `oasep`, `oasep_nf`, `silsep` only (`loa_set`, LB-won Q15): per-series SD heads + the calibrated "v4b" silent-cell LOA labels (silsep) |
| Effusion | window fusion of the 3x3 (geo pm g20 wmean over every window of main + Basin models) -> blended with the EF3 crop experts' image-read Effusion head (`eff`, w 0.3) |
| Synovitis | w 1/3 x image-head logit (mean over m_prob05 + n_prob05) + 2/3 x synovitis crop-expert term (crop-level fusion per design, mean pm g20 maxmix; ConvNeXt `cs_*` in the main slot, NFNet `s_*_nf` in the NFNet slot) |
| Baker's | window fusion of the 3x3 (as Effusion, no image-read blend) |
| Contusion | "all-4" bone member set (b3c_rep, b3l_rep, b3n_rep, bfs_rep on Opus image+report targets, crop-level fusion) blended into the 3x3 base -> FS-only member `bfs_rep` splice (w 0.25) -> ConvNeXt `sxm` image-read Contusion splice (w 0.3) |
| Fracture | the same four bone members (`frac_allmem`) -> FS-only `bfs_rep` splice (w 0.75) |

The old bone members (b_v3, b_lr3, b_nf) are switched off (`fc_shipped_off`): the v3c members replaced them.

Every blend weight, alignment constant and de-coupling constant was fitted on out-of-fold predictions of folds 0-2 (non-gold
studies), never on gold, and every column form was LB-tested one column at a time on the fold package (section 5 of
`05_LEARNINGS.md`, `06_LB_HISTORY.md`). When a member changed, the coupled terms were re-fitted on the new members
(`docs/source_notes/experiment_notes/oct16_experts_PROBES.md`, "FINAL-ASSEMBLY RE-FIT RULES").

## 1.4 Members by family

| family | recipes | trainer |
|---|---|---|
| main 2.5D effb0 | sub_b0, m_prob05 (image-read Synovitis head), sd1b (per-series-group heads) | `code/training/oct14_trainers/main/train.py`, `oct16_trainers/mainp/train.py` |
| Basin (NFNet-L0, RadImageNet R50) | nfnet, r50, n_prob05, bsd_nf | `code/training/oct14_trainers/basin/train.py`, `oct16_trainers/basin/train.py` |
| OA compartment experts | oa_v2cor, oa_v2cor_nf, oasep, oasep_nf, silsep | `code/training/oct14_trainers/oa/oa_train.py`, `oct16_trainers/oa/oa_train.py` |
| bone experts | b3c_rep, b3l_rep, b3n_rep, bfs_rep | `code/training/oct16_trainers/bone/bone_train.py` |
| synovitis-form crop experts | s_fs_nf, s_fc125_nf, cs_fs, cs_fc125, ef3, ef3fc, ef3nf, ef3nffc, w5b_mcl, z1_acl, b3g, sxm | `code/training/oct14_trainers/syn/syn_train.py`, `oct16_trainers/syn/syn_train.py` |

Recipe configs (the exact `config.json` of each member's fold-0 run): `code/training/recipes/<recipe>.json`; summary table in
`02_TRAINING.md`. NFNet fold models come from three runs of the same recipe family (`basin_sources` in config.json: P_final f0/f3,
N_pos20 f1, P_base f2/f4).

## 1.5 Runtime (Kaggle, 2 x T4)

- Fold-average v2 package: 205 min scored. All-data finals: P1 (93 models) 176 min, P3 (117) 315 min.
- Scoring time depends far more on the Kaggle machine than on the package. See `04_INFERENCE_AND_KAGGLE.md` before adding
  members.

## 1.6 All-data version (the submitted finals)

- Same code and config form. Every member's fold models are replaced by all-data runs (fold -1 = all 4,349 non-gold training
  studies, the fold runs' step count via `spe_ref`), 3-9 seeds per recipe.
- P1 = safe3 (3 seeds per recipe) + w5b_mcl 9 seeds, bfs_rep / oa_v2cor / sd1b 5 seeds (93 models).
- P3 = safe3 + w5b 9 seeds, with plain + pseudo-label twins side by side for the OA experts, z1 / w5b / sd1b and the generalists,
  pseudo-label twins only for b3g / bsd_nf / EF3 / synovitis crops, and plain bone + Lateral-OA extras
  (`code/inference/packaging/variants/best_b6.json`, 117 models).
- Builder: `code/inference/packaging/build.py` (`--variant`), kernel renderer `make_kernel.py`.
