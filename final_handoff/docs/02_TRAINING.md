# 2. Training: data preparation, trainers, recipes

Everything here was run on one local RTX 5090 (32 GB) plus up to two rented RTX 5090s. Commands are run from the repo root with
`PYTHONPATH=.`; our environment was the conda env `rsna`.

## 2.1 Environment

| package | training (local) | Kaggle scoring image |
|---|---|---|
| python | 3.12.14 | 3.13.15 |
| torch / torchvision | 2.14.1+cu130 / 0.29.1 | 2.11.0+cu128 |
| timm | 1.0.30 | 1.0.29 |
| pydicom | 3.0.2 | 3.0.2 |
| numpy / pandas | 2.5.4 / 3.0.6 | 2.1.3 / 2.3.3 |
| opencv-python-headless | 5.0.0 | 4.14.0 |
| also | blosc2 4.14.1, scikit-learn 1.9.1, scipy 1.18.1, pyarrow 25.0.1, iterative-stratification, nbformat, kaggle 2.2.4 | |

Training uses bf16 autocast and channels_last. The trainers call `torch.compile` where enabled in the config. Kaggle inference
runs fp16 on T4: keep fp32 weights, and guard against NaN for NFNet / DINO-style backbones.

## 2.2 Data preparation (once)

1. **Competition data** -> `data/original_data/` (train / test DICOM, `train.csv`, `train_series.csv`, reports). Not included here.
2. **Metadata + frames**: `code/data_prep/build_data.py` -> `data/meta/{studies,series,frames}.csv` and `data/frames/*.b2nd`
   (field levels and normalisation: `code/data_prep/META_README.md`).
3. **Canonical slice cache** (all trainers read it): `python -m common.cache build --fov 160 --px 400`, then
   `python -m common.cache jpeg --src fov160_px400 --quality 95` -> `fov160_px400_q95` (0.4 mm/px, 160 mm FOV, ~30 GB). Train
   from the JPEG copy: the raw uint8 cache (115 GB) thrashes RAM.
4. **Knee side per series**: `code/data_prep/side_check/` -> `series_side_final.csv` (included in `label_data/derived/`). About
   100 studies had a wrong side in the DICOM header; 405 series in 101 studies are flipped.
5. **Structure boxes** (for the localiser): nnU-Net Knee (`aagatti/nnunet_knee`, MIT) segments one sagittal fat-suppressed series
   per study. Its boxes are projected into every series (`code/data_prep/nnunet_boxes/`) -> `slice_boxes.parquet` (included).
   The public nnU-Net model is used offline only, as a teacher. It never ships.
6. **Localisers** (MobileNetV3, 128 px, CPU-fast): `code/localiser/loc_data.py` -> `loc_train.py` -> `loc_infer.py`.
   - v2 / v2cor: compartments, cartilage plates, menisci, notch, MCL, suprapatellar pouch, Hoffa, posterior recess.
   - v3 (`code/localiser/v3_bone/`): v2 + femur / tibia / patella, for the bone experts.
   - `loc_infer.py` writes `data/loc_pred*` (60-113 MB each; not included, regenerate).
7. **Side-flip detector**: `code/localiser/flipdet.py` (small CNN; mirrors back series the header rule mis-sides at test time).
8. **Folds**: `code/common/folds.py` (`get_folds(k=5, seed=0)`, multilabel-stratified over site + text positives + gold flag;
   duplicate reports share a fold). The 58 gold studies are never trained on. "Fold -1" = all 4,349 non-gold studies.

## 2.3 Labels the trainers read

Details and provenance: `03_LABELLING.md`.

| label | file (in `label_data/`) | used by |
|---|---|---|
| text v1 statuses -> probabilities ("v1prob") | `text_labels/v1.json` (read by `common/labels.py`) | every recipe (default target) |
| Codex synovitis / effusion image reads | `training_targets/img_syn_prob.csv` (from `derived/synovitis_labels_codex.csv`) | image heads: m_prob05, n_prob05, sd1b, syn-form experts |
| EF3 effusion image target (Codex + Qwen VLM z-aligned, + Opus on its 384 reads) | `training_targets/ieff_codex_vlm_opus.csv` | ef3, ef3fc, ef3nf, ef3nffc (ef3m / e3bn: effusion-form probe) |
| bone targets v3c (Opus images + report where read, old target elsewhere) | `training_targets/v3c/tgt_bone_v3m.csv`, `tgt_p2_bone.csv`, `ext_p2_main.csv` | b3c_rep, b3n_rep, bfs_rep / b3l_rep |
| Opus raw bone reads (masked image heads) | `training_targets/v3f/opus_raw.csv` | sxm |
| v4b calibrated Lateral-OA labels ("silent cells low") | `training_targets/targets/v4b_f{0..4}.csv`, `v4b_fall.csv` (all-data) | silsep |
| site table | `training_targets/site_train.csv` | bone v3c, EF3, sxm (site offsets) |
| bilateral partner studies | `derived/partners.csv` | main / OA trainers (pairs the two knees of one patient) |

The trainers expect these under their own experiment folder (for example `<EXP>/data/img_syn_prob.csv`) and the side table next
to the trainer (`main/series_side_final.csv`, `oa/series_side_final.csv`, ...). Recreate that layout with copies or symlinks before
training.

## 2.4 Trainers and launching

Two trainer sets, each a self-contained copy (experiments never imported code from each other):

- `code/training/oct14_trainers/` = the deployed recipes of the 0.961 package: `main/train.py` (sub_b0, m_prob05),
  `basin/train.py` (NFNet, R50, n_prob05), `oa/oa_train.py` (oa_v2cor, oa_v2cor_nf), `syn/syn_train.py` (s_fs_nf, s_fc125_nf),
  `bone/bone_train.py` (b_v3, b_lr3, b_nf).
- `code/training/oct16_trainers/` = the v3 "experts" trainers with every later option: `mainp/train.py` (sd1b: per-series heads),
  `basin/train.py` (bsd_nf), `oa/oa_train.py` (oasep*, silsep), `bone/bone_train.py` (b3*_rep, bfs_rep), `syn/syn_train.py`
  (cs_*, ef3*, w5b_mcl, z1_acl, b3g, sxm).

Launch one run from a recipe config (the launcher applies trainer DEFAULTS <- recipe config <- overrides):

```bash
# recipe config of a deployed member -> a "source run" folder
mkdir -p refs/w5b_mcl/fold0 && cp final_handoff/code/training/recipes/w5b_mcl.json refs/w5b_mcl/fold0/config.json
# one fold
PYTHONPATH=. python <EXP>/scripts/launch.py --kind syn --src refs/w5b_mcl/fold0 --name w5b_mcl --fold 0
# all data, new seed, the fold run's step count
PYTHONPATH=. python <EXP>/scripts/launch.py --kind syn --src refs/w5b_mcl/fold0 --name w5b_mcl_all_s101 --fold -1 \
    --set seed=101 spe_ref=434
```

- Kinds: oct14 `main | basin | oa | bone | syn`; oct16 adds `mainp`. Outputs: `runs/NAME/foldF/` (config.json, metrics, oof /
  gold predictions per epoch, checkpoints, `done.txt`).
- `spe_ref` keeps the all-data run at the fold runs' optimiser step count: 434 steps per epoch for recipes trained on folds 0-4,
  435 for recipes trained on folds 0-2.
- Queue runner (`run_queue.sh`): one queue line per run, `NAME KIND SRC FOLD key=value ... gb=N`. It starts a run only while
  `gpu_go.txt` holds 1, fewer than `slots_max.txt` runs are live, and N GB of GPU memory is free. Lines that don't fit are skipped
  until memory frees up. Stop it only by the PID in `runner.pid`.
- `scripts/audit.py` / `audit_configs.py` diff a finished run's config against its recipe. Run it before trusting a run.

## 2.5 Recipes in the final form

Generated from `code/training/recipes/*.json`. "folds" = how many fold runs have a local config; the fold package uses folds 0-4
or 0-2 per recipe (section 1.4).

| recipe | kind | backbone | input | epochs / lr | target / labels | extras | folds |
|---|---|---|---|---|---|---|---|
| sub_b0 | main | efficientnet_b0.ra_in1k | img 256, crop_mm 140, k 24, k_eval 96 | 20 / 0.001 | labels v1prob |  | 5 |
| m_prob05 | main | efficientnet_b0.ra_in1k | img 256, crop_mm 140, k 24, k_eval 96 | 20 / 0.001 | labels v1prob, ext_labels img_syn_prob.csv, ext_weight 0.05 |  | 3 |
| nfnet | basin | eca_nfnet_l0.ra2_in1k | size 256, fov 1.0, k 12, k_eval 24, T 8, T_eval 24 | 20 / 0.0003 | target v1 | seed 1, drop_path 0.1 | 2 |
| r50 | basin | radimagenet_resnet50 | size 256, fov 1.0, k 12, k_eval 24, T 8, T_eval 24 | 20 / 0.0005 | target v1 | seed 1, drop_path 0.1 | 5 |
| n_prob05 | basin | eca_nfnet_l0.ra2_in1k | size 256, fov 1.0, k 12, k_eval 24, T 8, T_eval 24 | 20 / 0.0003 | target v1, img_csv img_syn_prob.csv | seed 1, drop_path 0.1 | 3 |
| oa_v2cor | oa | efficientnet_b0.ra_in1k | px 224, k 8, k_eval 32, loc_version v2cor | 20 / 0.001 | labels v1prob |  | 5 |
| oa_v2cor_nf | oa | eca_nfnet_l0.ra2_in1k | px 224, k 8, k_eval 32, loc_version v2cor | 20 / 0.0003 | labels v1prob |  | 3 |
| s_fs_nf | syn | eca_nfnet_l0.ra2_in1k | px 224, k 8, k_eval 32, bags pouch,hoffa,post, outs img_syn,img_eff,syn,eff,bakers | 20 / 0.0003 | labels v1prob, img_csv img_syn_prob.csv, fs_only True | w_img 1.0, w_ieff 0.5, w_txt 0.5 | 3 |
| s_fc125_nf | syn | eca_nfnet_l0.ra2_in1k | px 224, k 8, k_eval 32, bags pouch,hoffa,post, outs img_syn,img_eff,syn,eff,bakers | 20 / 0.0003 | labels v1prob, img_csv img_syn_prob.csv, fs_only True | w_img 1.0, w_ieff 0.5, w_txt 0.5 | 3 |
| cs_fs | syn | convnext_tiny.in12k_ft_in1k | px 224, k 8, k_eval 32, bags pouch,hoffa,post, outs img_syn,img_eff,syn,eff,bakers | 20 / 5e-05 | labels v1prob, img_csv img_syn_prob.csv, fs_only True | seed 5, drop_path 0.1, llrd 0.9, w_img 1.0, w_ieff 0.5, w_txt 0.5 | 5 |
| cs_fc125 | syn | convnext_tiny.in12k_ft_in1k | px 224, k 8, k_eval 32, bags pouch,hoffa,post, outs img_syn,img_eff,syn,eff,bakers | 20 / 5e-05 | labels v1prob, img_csv img_syn_prob.csv, fs_only True | seed 5, drop_path 0.1, llrd 0.9, w_img 1.0, w_ieff 0.5, w_txt 0.5 | 3 |
| b_v3 | bone | efficientnet_b0.ra_in1k | px 256, k 8, k_eval 32, bags femur,tibia,patella, series all, outs frac,cont,c_lfc,c_mfc,c_fem,c_ltp,c_mtp,c_tib,c_fib,c_pat | 20 / 0.0003 | labels v1prob, tgt_csv tgt_v3.csv |  | 5 |
| b_lr3 | bone | efficientnet_b0.ra_in1k | px 256, k 8, k_eval 32, bags femur,tibia,patella, series all, outs frac,cont,c_lfc,c_mfc,c_fem,c_ltp,c_mtp,c_tib,c_fib,c_pat | 20 / 0.0003 | labels v1prob |  | 5 |
| b_nf | bone | eca_nfnet_l0 | px 256, k 8, k_eval 32, bags femur,tibia,patella, series all, outs frac,cont,c_lfc,c_mfc,c_fem,c_ltp,c_mtp,c_tib,c_fib,c_pat | 20 / 3e-05 | labels v1prob, tgt_csv tgt_v3.csv | hn_w 1.0 | 3 |
| sd1b | mainp | efficientnet_b0.ra_in1k | img 256, crop_mm 140, k 24, k_eval 96 | 20 / 0.001 | labels v1prob, ext_labels img_syn_prob.csv | series_heads sep, sd_weight 1.0, seed 11 | 5 |
| bsd_nf | basin | eca_nfnet_l0.ra2_in1k | size 256, fov 1.0, k 12, k_eval 24, T 8, T_eval 24 | 20 / 0.0003 | target v1 | series_heads 2, sd_weight 0.5, seed 1, drop_path 0.1 | 5 |
| oasep | oa | efficientnet_b0.ra_in1k | px 224, k 8, k_eval 32, loc_version v2cor, series all | 20 / 0.001 | labels v1prob | series_heads sep, sd_weight 1.0 | 5 |
| oasep_nf | oa | eca_nfnet_l0.ra2_in1k | px 224, k 8, k_eval 32, loc_version v2cor, series all | 20 / 0.0003 | labels v1prob | series_heads sep, sd_weight 1.0 | 3 |
| silsep | oa | efficientnet_b0.ra_in1k | px 224, k 8, k_eval 32, loc_version v2cor, series all | 20 / 0.001 | labels v1prob, tgt_csv v4b_f0.csv | series_heads sep, sd_weight 1.0, seed 18 | 5 |
| b3c_rep | bone | efficientnet_b0.ra_in1k | px 256, k 8, k_eval 32, bags femur,tibia,patella, series all, outs frac,cont,c_lfc,c_mfc,c_fem,c_ltp,c_mtp,c_tib,c_fib,c_pat | 20 / 0.0003 | labels v1prob, tgt_csv tgt_bone_v3m.csv, opus_csv ext_p2_main.csv | sd_weight 0.5, seed 1, hn_w 1.0, w_img 1.0 | 5 |
| b3l_rep | bone | efficientnet_b0.ra_in1k | px 256, k 8, k_eval 32, bags femur,tibia,patella, series all, outs frac,cont,c_lfc,c_mfc,c_fem,c_ltp,c_mtp,c_tib,c_fib,c_pat | 20 / 0.0003 | labels v1prob, tgt_csv tgt_p2_bone.csv, opus_csv ext_p2_main.csv | sd_weight 0.5, seed 1, hn_w 1.0, w_img 1.0 | 5 |
| b3n_rep | bone | eca_nfnet_l0 | px 256, k 8, k_eval 32, bags femur,tibia,patella, series all, outs frac,cont,c_lfc,c_mfc,c_fem,c_ltp,c_mtp,c_tib,c_fib,c_pat | 20 / 3e-05 | labels v1prob, tgt_csv tgt_bone_v3m.csv, opus_csv ext_p2_main.csv | sd_weight 0.5, seed 1, hn_w 1.0, w_img 1.0 | 3 |
| bfs_rep | bone | efficientnet_b0.ra_in1k | px 256, k 8, k_eval 32, bags femur,tibia,patella, series fs, outs frac,cont,c_lfc,c_mfc,c_fem,c_ltp,c_mtp,c_tib,c_fib,c_pat | 20 / 0.0003 | labels v1prob, tgt_csv tgt_bone_v3m.csv, opus_csv ext_p2_main.csv | sd_weight 0.5, seed 1, hn_w 1.0, w_img 1.0 | 5 |
| w5b_mcl | syn | efficientnet_b0.ra_in1k | px 224, k 8, k_eval 32, bags mclz, outs mcl | 20 / 0.001 | labels v1prob, img_csv img_syn_prob.csv | w_img 1.0, w_ieff 0.5, w_txt 0.5 | 5 |
| z1_acl | syn | efficientnet_b0.ra_in1k | px 224, k 8, k_eval 32, bags notch40, outs acl | 20 / 0.001 | labels v1prob, img_csv img_syn_prob.csv | w_img 1.0, w_ieff 0.5, w_txt 1.0 | 5 |
| b3g | syn | efficientnet_b0.ra_in1k | px 224, k 8, k_eval 32, bags knee,medc,latc,pfl,notch,pouch,hoffa,post,mcl, outs img_syn,img_eff,syn,eff,bakers,acl,mcl,mm,lm,moa,loa,pf,cont,frac | 20 / 0.001 | labels v1prob, img_csv img_syn_prob.csv | w_img 1.0, w_ieff 0.5, w_txt 1.0 | 5 |
| ef3 | syn | convnext_tiny.in12k_ft_in1k | px 224, k 8, k_eval 32, bags pouch,hoffa,post, outs img_syn,img_eff,syn,eff,bakers | 20 / 5e-05 | labels v1prob, img_csv img_syn_prob.csv, ieff_csv ieff_codex_vlm_opus.csv, fs_only True | seed 5, drop_path 0.1, llrd 0.9, w_img 1.0, w_ieff 0.5, w_txt 0.5 | 5 |
| ef3fc | syn | convnext_tiny.in12k_ft_in1k | px 224, k 8, k_eval 32, bags pouch,hoffa,post, outs img_syn,img_eff,syn,eff,bakers | 20 / 5e-05 | labels v1prob, img_csv img_syn_prob.csv, ieff_csv ieff_codex_vlm_opus.csv, fs_only True | seed 5, drop_path 0.1, llrd 0.9, w_img 1.0, w_ieff 0.5, w_txt 0.5 | 5 |
| ef3nf | syn | eca_nfnet_l0.ra2_in1k | px 224, k 8, k_eval 32, bags pouch,hoffa,post, outs img_syn,img_eff,syn,eff,bakers | 20 / 0.0003 | labels v1prob, img_csv img_syn_prob.csv, ieff_csv ieff_codex_vlm_opus.csv, fs_only True | w_img 1.0, w_ieff 0.5, w_txt 0.5 | 5 |
| ef3nffc | syn | eca_nfnet_l0.ra2_in1k | px 224, k 8, k_eval 32, bags pouch,hoffa,post, outs img_syn,img_eff,syn,eff,bakers | 20 / 0.0003 | labels v1prob, img_csv img_syn_prob.csv, ieff_csv ieff_codex_vlm_opus.csv, fs_only True | w_img 1.0, w_ieff 0.5, w_txt 0.5 | 5 |
| sxm | syn | convnext_tiny.in12k_ft_in1k | px 224, k 8, k_eval 32, bags medc,latc,pfl, outs img_cont,img_frac,cont,frac | 20 / 5e-05 | labels v1prob, img_csv img_syn_prob.csv, ibone_csv opus_raw.csv, fs_only True | seed 5, drop_path 0.1, llrd 0.9, w_img 1.0, w_ieff 0.5, w_txt 0.5 | 3 |
| ef3m | main | efficientnet_b0.ra_in1k | img 256, crop_mm 140, k 24, k_eval 96 | 20 / 0.001 | labels v1prob, tgt_csv ieff_codex_vlm_opus.csv |  | 5 |
| e3bn | basin | eca_nfnet_l0.ra2_in1k | size 256, fov 1.0, k 12, k_eval 24, T 8, T_eval 24 | 20 / 0.0003 | target v1, tgt_csv ieff_codex_vlm_opus.csv | sd_weight 0.5, seed 1, drop_path 0.1 | 5 |

Main / Basin inputs: the main effb0 sees 3-slice windows of every series (140 mm crop -> 256 px; 24 windows per study in training,
96 at eval) with series-type + position embeddings and a per-finding attention head. Basin (NFNet-L0 / RadImageNet R50) uses
series "slots" and slice samples (T 8 train / 24 eval). Crop experts sample k crops per bag in training (8) and k_eval (32-64) at
eval. They attend over the crops with bag-restricted attention.

## 2.6 Training cost (all-data run, one RTX 5090)

| recipe family | minutes per run | GPU memory |
|---|---|---|
| sub_b0, m_prob05, sd1b | 12-14 | ~12 GB |
| NFNet / R50 Basin (nfnet, r50, n_prob05, bsd_nf) | 28-33 (R50 ~47 when sharing the card) | 13 / 22-26 GB |
| OA experts (effb0 / NFNet) | 9 / 15 | 9 GB |
| bone experts (effb0 / NFNet) | 12 / 19 | 12-16 GB |
| synovitis-form experts (ConvNeXt / NFNet) | 15-16 | 9-16 GB |
| w5b_mcl, z1_acl | 4 | 12 GB |
| b3g (9 bags) | 26 | 24 GB |

Fold runs take about as long: the all-data runs keep the fold runs' step count (`spe_ref`).

## 2.7 Reproducing the members

- Fold-average package: train each recipe on its folds (0-4 or 0-2) with `--fold F`, then build the package (section 4).
- All-data finals: `--fold -1 --set seed=S spe_ref=434|435` for 3+ seeds per recipe (see `05_LEARNINGS.md` on how many seeds per
  recipe and which recipes are noisy).
- Pseudo-label twins (P3 only): the same recipes with the teacher-v1 targets in `label_data/pseudo_labels/pl2_<recipe>_v1.csv`
  (`tgt_csv` / `img_csv` overrides; see `03_LABELLING.md` section 3.5).
