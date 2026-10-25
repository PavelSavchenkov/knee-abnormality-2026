# oct14_backbone_sweep: image backbones we never properly tried, inserted into our best pipeline with an LR sweep

Started 2026-10-14 ~01:35 (user brief relayed by the review session knee-abnormality-2026-29). GPU: after the PL session
(knee-abnormality-2026-2e) seals experiments/v2/oct12_pseudo and hands it over; 24 h from then, or until exhausted. CPU-only prep
before that. Then the GPU goes to the VLM session (experiments/v2/oct14_vlm).

Files: user_directives.md (every instruction), history.md (every backbone ever tried + public evidence: the line-by-line read of old
experiments), ideas.md (ranked ideas ledger), ledger.md (numbered entries: runs, reads, decisions, reviews), queue.txt / run_queue.sh
(GPU runner), src/ (trainer = copy of experiments/v0/oct10_baseline_3rd/src, the Basin-code pipeline of NFNet / R50), scripts/ (reads),
refs/ (copies of reference predictions, no weights), runs/ (this sweep), out/ (read outputs).

Vehicles: src/ (Basin-code final recipe, primary), main/ (main pipeline R2 copy), oa/ (OA / meniscus crop expert copy; ideas E1),
bone/ (F / C bone-expert copy). See Status -> How to resume for the full tool list.
Tools: scripts/read_arm.py (standard read of one run), a2a.py (apples-to-apples per-finding single-model read vs the same-recipe reference; decides), pkg.py (read inside the expert-routed 0.957 package; information only), lr_board.py (LR sweep side by side), gold_flips.py (manual error analysis),
prior_members.py (earlier encoders as 4th members), cpu_check.py (recipe model forward / backward on CPU), fp16_check.py (Kaggle T4
precision), smoke_gpu.sh (tiny GPU smokes); status.sh (check-in snapshot); run_queue.sh + watchdog.sh + memguard.sh (runner).

## Goal

Find backbones that add value to the cross-backbone package (main effb0 + NFNet-L0 + RadImageNet R50 generalists, plus the experts
built on them). "Properly tried" = in our best pipeline + an LR sweep at full length. Required: ConvNeXt-T and DINOv2-S. Image-pretrained
only (no VLMs). Never use other participants' weights.

## Vehicle: the Basin final recipe (2nd / 3rd baselines)

Why this pipeline:
- It is the best single pipeline (5-fold OOF NFNet 0.9737 / 0.9543, R50 0.9736 / 0.9545 vs main effb0 0.9721 / 0.9523) and holds 2 of
  the package's 3 generalist slots;
- it is where the last encoder search ran (3rd baseline), so every new candidate is directly comparable with that table and with
  the true NFNet seed-2 yardsticks (folds 1 / 2 / 4);
- its regime (random triplet positions, EMA 0.999, RandAugment per triplet, 20-ep cosine, warmup 3) is the strongest regularisation we
  have, which bigger modern encoders need; it already supports layer / stage LR decay, LP-FT, output-bias init, ViTs
  (dynamic image size) and plain BN.
Recipe (queue BASE + R, as the 3rd baseline): `--compile 1 --lr_head 1e-3 --wd 0.02 --bs 8 --clip 3 --ema 0.999 --comp 1 --head battn
--layers 0 --asym 0.5 --syn_eff 1 --fov 0.875 (all planes) --cor_yshift 0.08 --epochs 20 --warmup 3 --ra_n 2 --aug_per_image 1 --rot 15
--scale 0.15 --shift 0.08 --sample pos --uniform 1 --T_eval 24 --grad_ckpt 0 --save all --eval_every 2` + `--backbone X --lr Y`
(+ per-family settings: layer / stage decay, bias init, image size).
The main pipeline (oct7 R2) is the second vehicle for the top candidates only if the Basin read is promising but the slot logic
suggests main's structure (cheaper, the public ConvNeXt / CoAtNet / DINOv2 readers look like it).

## Validation regime

Reads use scripts/read_arm.py (lib.py = adapted copies of the PL / review read tools). Every number is AUC or x1e-3.

1. Own learning curve per fold: trusted / lookat_m / text / gold / train-probe trusted at every evaluated epoch (every 2 epochs).
   Epoch rule (protocol): median-of-best = each fold's best-trusted epoch, median over the candidate's folds; a single screening fold
   reports its best epoch AND the last epoch (e20), and the decision uses the last epoch unless the curve clearly peaks earlier
   (memorisation, as NFNet's e14). Slow families (ViTs) are judged on their own curve, never at equal steps (memory
   judge-by-own-learning-curve).
2. Single model vs the package members on the same fold, with the NFNet true-seed row (REF_nf20_s2, folds 1 / 2 / 4) and the main
   seed row (sub_b0_s1) as model noise. Metrics: trusted, lookat, lookat_m, text (core, w/o MCL / synovitis), sum10, thr, un, sg,
   gold-58 (11 findings; gold_nofc; gold_unc = report-uncertain cells), gold synovitis separately, Codex effusion / synovitis
   image reads (all and report-uncertain cells).
3. Correlation: mean per-finding Spearman vs main / NFNet / R50 (seed pairs 0.94-0.95 for scale; cross-pipeline 0.85-0.89).
4. Package value (the decision metric; user: ensemble = cross-backbone):
   - ADD: base (plain logit mean of main + NFNet + R50 on the fold) + candidate vs base + NFNet true seed (the yardstick a 4th model
     must beat; 3rd-baseline rule) and vs base + main seed;
   - SWAP: candidate in each slot vs that slot's seed swap (noise row);
   - logit spread ratio sd(cand) / sd(slot) as a re-weighting flag;
   - paired study bootstrap of (base + cand) - (base + NFNet seed) on trusted and lookat_m; gold-58 rank-free direction (align);
   - transfer views: _xe = without Effusion (the 0.961 package already fixes Effusion with window fusion + an image head), tw_ =
     weighted by the generalists' share of each finding in the package (ACL / MCL / Effusion / Baker's 1, OA / menisci 0.4,
     fracture 0.1, contusion / synovitis 0: experts own those).
5. Manual error analysis per finished candidate (scripts/gold_flips.py): the gold cells the ADD step moves most, with v1 status and the
   report text; gains / losses by v1 status x gold label; small rank moves on rare-positive findings are noise.
6. Noise model (memory noise-model-r2): per-run sd trusted 0.0014, lookat 0.0034 across seeds, gold 0.0032; one fold resolves
   ~0.003 trusted single-model differences; package ADD differences are smaller (fold-1 seed-yardstick spread ~ +-0.5 trusted).

Pre-registered rules (written before any run exists):
- LR sweep (per backbone, fold 1, full 20-ep schedule): pick the LR with the best mean of trusted + lookat_m at its median-of-best /
  last epoch; within 0.002 of each other -> prefer the one with the larger package ADD; a monotone curve at the bracket end extends
  the bracket by one point (x2 or x0.5).
- Screen pass (fold 1): single model within 0.003 trusted of the NFNet member on the fold AND package ADD (trusted, lookat_m, sum10)
  above the NFNet-seed ADD row. Fail on every read -> at most one confirmation fold (memory prioritise-roi), then drop.
- Confirmation: folds 2 and 4 (true NFNet seed rows exist) at the pre-registered epoch; promoted as a package candidate if ADD beats the
  NFNet-seed ADD on >= 2 of 3 folds with a positive mean on trusted and lookat_m, and gold-11 not below -0.005 (veto). Then folds 0 / 3
  (5-fold OOF) and the 5-fold package table with gold-58 test-like averaging.
- Stop rules for screens (ledger L11): broken = trusted < 0.93 at e6 or no learning; clearly worse = at e12 below NFNet's e12 by
  > 0.008 trusted and > 0.01 lookat_m unless still rising >= +0.002 per eval.
- A slot swap is a candidate only if it beats the slot's seed swap on >= 2 of 3 folds on trusted and lookat_m.
- LB-probe gate (review 29 / 213 after K12fkd, corrected by the user 10:25: "reliable gain only on some classes also counts"):
  a candidate qualifies either on the macro OR on specific findings, used then only in those columns (per-finding routing). Either
  way the gain must be reliable: (a) per finding beyond its multi-seed noise (per-finding paired bootstrap vs the seed yardstick, z
  vs the 3-ref sd) and in the same direction on >= 2 of 3 folds; (b) not contradicted by that finding's gold-58 and Codex reads
  (gold12 / Codex effusion / synovitis checked, not only gold_nofc / _xe); (c) clearly beyond best-of-N selection noise (package
  gold sd ~2e-3 per arm) or confirmed on folds not used to select it; (d) apples-to-apples (user ~11:20): decided single-model vs
  the same-recipe non-expert reference (Basin -> NFNet / R50, main -> effb0 sub_b0; noise = that reference's seed replicate;
  scripts/a2a.py) and by ADD into the non-expert 3x3 base vs the seed ADD; reads against the expert-routed package (scripts/pkg.py)
  are information only. Breadth is necessary for a macro claim, not sufficient.
- Nothing is submitted; a Kaggle package needs the user's explicit go for each submission.

## 10-min check-in routine (user 01:37, carried over from the PL / review sessions)

1. `bash status.sh`: RAM / PSI, GPU apps, runner / watchdog / memguard, running arms' last epochs, events; stuck or over budget first.
2. Every finished run: `scripts/read_arm.py RUN --folds F` (curve + median-of-best, single vs members with seed rows, Spearman, package
   ADD / SWAP vs seed yardsticks, bootstrap, gold direction, per finding; _xe / tw_ views) and `scripts/gold_flips.py` (manual read).
3. Code check before a new kind of arm: diff vs the recipe, assertions, fold / epoch mapping, identity reads (candidate == member -> 0).
4. Pre-register rule + forecast in the ledger before the run; compare after; correct wrong claims at once.
5. Re-rank ideas.md after every run / insight; state evidence strength; numbered ledger entries; tell dependent sessions.

## Cost model (RTX 5090, from the Basin speed tables)

R50 ~87 s / epoch, NFNet ~95-108 s; ConvNeXt-T ~1.25x R50 (bench 1447 vs 1794 img / s at 256), DINOv2-S at 252 ~ R50 (1870 img / s),
at 336 ~1.8x. A full 20-ep fold: ~30-40 min (ViT-S at 336 ~55 min). 24 h ~ 35-45 full runs.

## Status: SEALED 2026-10-15 11:20 (user: "seal current learnings, next plans and all documentation ... then quit")

Nothing of this folder runs: runner / watchdog / memguard services stopped, gpu_go.txt = 0, no process on the GPU or CPU. The GPU
belongs to the VLM session (user 10-14 ~16:05; All Data borrowed and returned it). Details: ledger.md L16-L51, history.md "This
sweep's results", ideas.md (next plans), user_directives.md (every instruction).

### Learnings (answers to the brief: new backbones strong alone / adding to the 3-pipeline ensemble?)

| backbone | vehicle / LR sweep | verdict (vs the same-recipe reference + its seed row) | inside the deployed package (information) |
|---|---|---|---|
| ConvNeXt-T | main: 5e-5 / 1e-4 / **2e-4** / 4e-4 (f1); 2e-4 on f1 / f2 / f4 | **beats effb0: trusted +3.8 [+1.6, +5.6] pooled (seed +1.2), 3 / 3 folds**; 3-pipeline ADD +0.7 [+0.2, +1.1] over a seed | ADD +0.2 [-0.1, +0.4], SWAP +0.2 over a seed -> LB forecast ~0, no probe |
| ConvNeXt-T | Basin: 1e-4 (f1) | -4.3 vs NFNet (seed +0.7) | - |
| ConvNeXt-T | bone expert: 2e-4 / 1e-4 / **5e-5** / 2.5e-5 (f1, k64) | contusion +7.9 [-0.1, +15.3] vs b_v3 (seed +1.5), fracture -5.5 | contusion swap +2.1 (seed +0.7) |
| ConvNeXt-T | OA v2cor expert: 2e-4 (f1) | OA grades = effb0; medial meniscus +6.3 [+2.8, +10.4] (seed -0.2) | meniscus columns +1.5..+2.1 over a seed |
| ConvNeXt-T everywhere | 0.957 package (f1) | - | upper bound +0.3 [-0.8, +1.2] over seeds in the same slots |
| DINOv2-S | Basin 252: 5e-5 / 1e-4; main 252: 5e-5; Basin 336: 5e-5 (stopped e15) | -9.1 vs NFNet (252), -9.4 (336, e14); ADD -1.5 [-2.5, -0.6] | - |
| RadImageNet DenseNet-121 | Basin 5e-4 (stopped e7), 1e-3 6-ep screen | e6 0.9357 / 0.9323 vs NFNet 0.9647 / anchor 0.9623 | - |
| **NFNet-L0 output stride 16** | Basin 6-ep screen 3e-4 (f1) | **+2.8 / +3.3 vs the stride-32 anchor; ACL +8.5, LM +6.2, contusion +8.6 (CI > 0)** | not read (screen) |

- Answer: ConvNeXt-T is strong on its own in the main recipe (3 clean folds) but adds ~0 to the deployed package: generalist-slot
  gains are diluted (OA / meniscus 0.6 from experts; F / C / synovitis do not read the base). Nothing proposed for a probe.
- Most promising open lead: the detail axis (NFNet-L0 at output stride 16), one 6-ep screen only.
- Natural-image SSL ViTs tie NFNet on diffuse findings, lose OA / focal (the VLM's SigLIP2 profile) -> domain-pretrained ViTs
  (OrthoFoundation-L, MRI-CORE) are prepared against their natural controls.

### Next plans (in order; all lines paused in queue.txt, the user decides GPU time)

1. NFNet-L0 stride 16: 2nd LR screen SC_nfos16_1.5e-4 (#PAUSED_ALLDATA, ~30 min), then the full 20-ep run at the better LR on
   fold 1 (~75 min) -> a2a vs N_pos20 + NFNet seed, SWAP of the NFNet slot vs the seed swap; folds 2 / 4 if it holds.
2. Frozen domain-ViT screens: `bash scripts/frozen_screens.sh` (GPU phase ~30 min: finishes feats/ofl256 with --resume_from 4300 and
   extracts dv3l256; mricore256 / dv3b256 are done), then CPU probes (fp32 now, ~15 min / seed, run with CUDA_VISIBLE_DEVICES="");
   reads: `scripts/a2a.py probe_mricore256 --fold 1 --ref probe_dv3b256 --noise probe_dv3b256_s345` (and ofl vs dv3l), per class
   group. Then tier-2 fine-tune screens of the better domain model (SC_ofl_2e-5 / SC_mricore_5e-5 vs their controls).
3. Fine-detail queue (L47): HRNet-W18, effb0 / ConvNeXt-T stride 16 in main, HRNet-W32, SAM ViT-B, DINO ViT-S/8 (GPU-smoke first).
4. Breadth screens (L45): NFNet-L1, EfficientNetV2-S (compile 0), CAFormer-S18, B3-NS, MambaOut-T, R50-a1, RadImageNet
   InceptionV3; CoAtNet / MaxViT need a smoke with a larger cap. Per the user's 15:55 rule: the best screen gets the full run +
   one more LR, then the next best.
5. Parked (ROI): ConvNeXt-T main folds 0 / 3, OA / bone ConvNeXt folds 2 / 4 (upper bound +0.3 in the package).
6. User decisions: MedSigLIP-448 (gated HAI-DEF terms), Curia (research-only), DINOv3-family licence for any deployment.

### How to resume

- Un-pause queue lines (remove the #PAUSED_* prefix) in priority order, then `GB_FREE=29 bash restart_runner.sh` (only with no arm
  of this folder running; it sets gpu_go 1). `bash scripts/return_gpu.sh` releases the GPU at once (kills this folder's python jobs,
  stops the runner, gate 0). Watchdog / memguard (stopped at the seal): from this folder, `systemd-run --user --unit bsweep-watchdog -p WorkingDirectory=$PWD
  --setenv=PATH="$PATH" --setenv=HOME="$HOME" /usr/bin/bash $PWD/watchdog.sh` (same with bsweep-memguard / memguard.sh).
- Vehicles: src/ (Basin; --output_stride, local loaders orthofoundation_l / mri_core_vitb / radimagenet_* / samvit), main/
  (output_stride=), oa/ (OA v2cor expert), bone/ (F / C expert + bone_eval_k.py); runner kinds Basin / main / oa / bone.
- Reads: a2a.py (single + ensemble + class groups; --ref / --noise), a2a_pool.py (multi-fold, L34 rules), pkg.py / pkg_pool.py /
  pkg_combo.py (0.957 package, information), bone_read.py / bone_vs_expert.py, oa_read.py, lr_board.py, read_arm.py.
- Weights (links in weights/): radimagenet, ortho_foundation, mri_core (MRI_CORE_vitb.pth, sha256 0ca90aeb...8276).

Nothing is submitted; any Kaggle step needs the user's explicit go for each one.
