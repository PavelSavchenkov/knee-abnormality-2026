# oct10_baseline_3rd: the 3rd baseline (RadImageNet ResNet-50, centre-slice input) — SEALED 2026-10-10 ~20:45

This folder is finished; nothing runs here any more.
- Created 2026-10-10 ~12:50 on user request: a third pipeline that adds ensemble value on top of the two existing
  baselines and is fast to train.
- It also hosts the encoder search that found it and the ensemble measurements behind the decision.
- Detail: ledger.md (every run, in time order), learnings.md (consolidated findings), ideas.md (backlog),
  user_directives.md (the user's instructions).

## The three baselines (final state, 2026-10-10)

| | folder | pipeline | encoder / input | eval epoch | 5-fold OOF trusted / lookat | fold checkpoints |
|---|---|---|---|---|---|---|
| 1 main | oct7_baseline | 2.5D windows from all series + per-finding attention (R2) | EfficientNet-B0, triplets | e19 (last) | 0.9721 / 0.9523 (mean of 3 refs) | runs/sub_b0/fold0-4/model.pt (also sub_b0_s1, effb0_lr1e-3_R) |
| 2 NFNet | oct9_baseline_2nd | Basin code: 5 series slots, cor / ax crops, battn readout, random positions, EMA, 20-ep cosine | ECA-NFNet-L0, triplets | e14 (stop) | 0.9737 / 0.9543 | runs/P_final/fold0, fold3; N_pos20/fold1; P_base/fold2, fold4 (model_e14.pt EMA, model_raw_e14.pt raw) |
| 3 RadImageNet | here | the same Basin code as 2 | RadImageNet ResNet-50, **centre slice x3** (`--mono 1`), plain BN | e20 (last; 5-fold median-of-best) | 0.9736 / 0.9545 | runs/F20_rad50m_5e-4/fold0-4/model_e20.pt (EMA; model_raw_e20.pt raw) |

Blends (OOF, single view, equal-weight mean of z-scored logits; plain logit or prob mean gives the same within 0.0003):

| blend | 5-fold trusted / lookat |
|---|---|
| main + NFNet | 0.9762 / 0.9579 |
| **main + NFNet + RadImageNet** | **0.9775 / 0.9595** |
| main + NFNet + another main seed | 0.9763 / 0.9581 |

- Per fold, sum10 x1e-3 vs main + NFNet:

  | fold | + RadImageNet | + main seed | + a true NFNet seed |
  |---|---|---|---|
  | 0 | +17.6 | +0.6 | - |
  | 1 | +17.5 | +2.0 | +13.0 |
  | 2 | +16.4 | +0.8 | -3.0 |
  | 3 | +8.4 | +4.3 | - |
  | 4 | +4.9 | +0.4 | -0.8 |

- Pairwise rho: RadImageNet vs NFNet 0.90, vs main 0.84-0.85; NFNet vs main 0.86. Seed pairs: NFNet 0.949, main 0.935.
- Gold-58 (every model predicts every study, like the test), 3 pipelines, k models per pipeline: k1 0.9281, k2 0.9291,
  k3 0.9293, k4 0.9294, k5 0.9292. **It saturates at 2-3 models per pipeline.**
  - A second seed of the same fold adds about 2/3 of what a second fold adds (scripts/seeds_vs_folds.py, NFNet folds 1 / 2 / 4:
    alone +3.1 vs +4.6; inside the 3-pipeline blend +0.6 vs +0.8).
  - So plan ~3 all-data seeds per pipeline. Gold-58 is noisy (~+-0.003).
- Submission: a 3 pipelines x 3 folds package (folds 0-2) is built by another session in submissions/oct10_3pipe_3f/
  (see its README).
  - That session also ported the test-time mirror detector to the dense (random-position) models: Basin's predict.py had
    it switched off for them, which cost ~0.001 trusted per model on mis-sided studies.

### 3rd-baseline recipe (queue line)

`F20_rad50m_5e-4 FOLD 1 --backbone radimagenet_resnet50 --grad_ckpt 0 --save all --eval_every 2 --sample pos --uniform 1
--T_eval 24 --lr 5e-4 --mono 1` on top of run_queue.sh's BASE / R:
- BASE: compile, battn, crops, asym 0.5, syn_eff, EMA 0.999, fov 0.875, bs 8, lr_head 1e-3, wd 0.02, clip 3;
- R: 20 ep, warmup 3, RandAugment N 2 per triplet, rot 15 / scale 0.15 / shift 0.08.
- About 87 s per epoch on the 5090, ~31 min per fold (NFNet: ~95 s, ~25 min). Same cost class as NFNet, not faster.
- Weights: weights/radimagenet/ResNet50.pt (loader in src/model.py create_encoder). Checkpoint keys carry
  "enc._orig_mod." (torch.compile); strip it for an uncompiled model. Outputs are 14 logits: use [:12].

## What was learned (positive)

- **Radiology pretraining is what decorrelates.**
  - ImageNet CNN swaps in this pipeline land at rho ~0.92 vs NFNet (ResNet-34d, SE-ResNeXt26ts, NF-RegNet-B1).
  - RadImageNet lands at 0.89 with either input. ImageNet-pretrained MobileNetV4 reached 0.79 too, but only because it
    collapsed.
- **Input contract.** RadImageNet was pretrained on grayscale x3: the centre slice x3 gives +0.008 trusted / +0.008
  lookat at e6 over 3 adjacent slices.
  - The same on NFNet (ImageNet) costs -0.006 / -0.007 and adds no diversity, so pipelines 1 / 2 keep triplets.
- A 3rd member must be close to the others in quality. ResNet-34d (0.004 weaker, rho 0.92) loses in the 3-way blend at
  every weight.
- **BatchNorm:**
  - plain BN on the normal crop path works and beat Basin's bn_single workaround (ResNet-34d e6 0.9568 vs 0.9532, 1.6x
    faster);
  - frozen ImageNet BN fails (EfficientNetV2-S learned nothing, train AUC 0.59).
- **Speed:**
  - gradient checkpointing costs small encoders 20-48% per step (memory is fine without it: 10-24 GB);
  - one GPU sync per step was removed from the training loop;
  - no concurrent benchmarking while an arm trains (an OOM nearly hit an arm).
- **Ensembling rules (OOF):**
  - the plain average beats per-class weights / routing out of fold (scripts/class_weights.py);
  - logit ~ z ~ prob mean; rank mean slightly worse;
  - the bar for a 3rd member is "+ a seed of the stronger pipeline", not "+ a main seed".

## What did not work (each a specific regime; fold 1, 6-ep screens at e6 unless noted; NFNet seeds 0.9623-0.9640 / 0.9419-0.9424)

| encoder / change | e6 trusted / lookat | rho vs NFNet | verdict |
|---|---|---|---|
| EfficientNetV2-S, frozen BN | 0.5868 at e3 (no learning) | - | frozen BN fails |
| EfficientNetV2-S, bn_single | 0.9553 at e3 (ahead of NFNet's 0.9467) | - | cancelled: user, slow (261 s / epoch) |
| NF-RegNet-B1 lr 5e-4 | 0.9562 / 0.9352 | 0.93-0.94 | weaker, seed-like; dropped (user) |
| ResNet-34d plain BN 5e-4 / 1e-3 / bn_single | 0.9568 / 0.9531 / 0.9532 | 0.92 | full 20-ep: peak e16 0.9691 (-0.004); 3-way -16 |
| MobileNetV4-conv-medium 1e-3 / 3e-3 | 0.9099 / 0.8190 | 0.79 | collapses (pc12 0.86-0.89) |
| EdgeNeXt-S (LN) 2e-4 | 0.9489 / 0.9318 | 0.92 | weaker (ACL -34) |
| SE-ResNeXt26ts | 0.9544 / 0.9341 | 0.92 | weaker |
| RadImageNet R50, triplet input | 0.9540 / 0.9306 | 0.885 | diverse but weaker -> mono fixed it |
| RadImageNet R50 mono, lr 2.5e-4 / 5e-4 / 1e-3 | 0.9603 / 0.9623 / 0.9654 | 0.89 | 5e-4 chosen for the 5 folds; 1e-3 full run is backlog |
| NFNet mono | 0.9565 / 0.9352 | 0.94-0.95 | worse, no diversity |
| no single NFNet pipeline knob (13 lever screens: mirror, cutout, T 12, hybrid readout, 8 series, transformer layer, LR, BS, ...) | - | 0.934-0.972 | none decorrelates beyond a seed (scripts/knob_diversity.py) |

- Too slow without checkpointing for "fast": ConvNeXt-V2-nano, CAFormer-S18, ConvFormer-S18, ResNet-50-GN.
- Do not build with our kwargs: EfficientViT, DenseNet, DLA, InceptionNeXt, RepViT, FastViT.
- Out of memory without checkpointing: RegNetZ-B16 / C16, EfficientNetV2-T.
- Speed tables: out/speed_bench_*.txt.

## Runs in runs/

- Copies without weights (NFNet references for self-contained reports): S6_nf_3e-4, N_pos20, P_base, P_final.
  P_final/fold3 here is a copy; the real one lives in oct9_baseline_2nd.
- Moved here from oct9_baseline_2nd: BB_v2s, S6_v2s_5e-4, S6b_v2s_5e-4.
- This session's runs:
  - 6-ep screens: S6_nfrb1_5e-4, REF_nf6_s2 (NFNet seed 2), S6p_r34d_5e-4, S6b_r34d_5e-4, S6p_r34d_1e-3,
    S6p_mnv4c_1e-3 / 3e-3, S6_edgs_2e-4, S6p_rad50_5e-4, S6p_srx26ts_5e-4, S6p_rad50m_{2.5e-4, 5e-4, 1e-3}, S6_nfm_3e-4.
  - Full runs: F20_r34d_5e-4 f1; F20_rad50m_5e-4 f0-4; REF_nf20_s2 f1 / f2 / f4 (NFNet seed 2, e14).
- Every evaluated epoch is saved (EMA model_e*.pt + raw model_raw_e*.pt + opt_e*.pt). preds.parquet holds sets oof /
  gold / train.

## Code (src/ = a copy of oct9_baseline_2nd/src, with these changes)

- model.py: projection width = head_hidden_size when set (MobileNetV4 / HGNetV2); bn_single asserts no head conv.
- train.py:
  - `--mono` (centre slice x3, train + eval);
  - loss / grad norm accumulated on the GPU (no per-step sync; backups in out/train.py.bak_*).
- run_queue.sh: counts arms of both folders (oct9 + oct10), so only one arm uses the GPU. slots_max.txt = 0 pauses it.
- watchdog.sh: kills an arm silent for 15 min and retries it once with --compile 0. memguard.sh guards RAM.
- scripts/:
  - final_eval.py: the 5-fold table above;
  - blend3.py: 3-way value vs yardsticks;
  - pair_diversity.py: pair rho / blend;
  - knob_diversity.py;
  - class_weights.py;
  - fold_count.py: test-time averaging on gold-58;
  - seeds_vs_folds.py;
  - speed_bench.py;
  - report_arms.py, main_ref.py, compare_pipelines.py (from the 2nd baseline).

## Measurement conventions and caveats

- Fold 1 screens; the 3rd baseline was confirmed on folds 2 / 4 / 0 / 3 with a fixed epoch (e20 = its 5-fold
  median-of-best).
- Main at e19, NFNet at e14 (its pre-registered median-of-best).
- 6-ep screens favour fast learners and higher LR. Encoders are judged on a full run before blend tests.
- OOF blends include 1 -> N model averaging, which the 5-fold test mean already has. So "+ a seed of an existing
  pipeline" is the yardstick, not the 2-model base.
- Synovitis is not measurable on OOF (3-8 text negatives per fold); sum10 leaves out MCL / synovitis.
- Laterality: pipelines 2 / 3 train on side-corrected caches. At test time the sides come from metadata plus the mirror
  detector (now ported for dense models by the submission session).

## Backlog (not run; ideas.md)

- RadImageNet mono full run at lr 1e-3 (better at e6: 0.9654 vs 0.9623).
- A 30-ep schedule (5e-4 still rose to e20 on folds 0-2).
- RadImageNet DenseNet-121 (weights present, needs a loader).
- Mono on cor / ax only (to win back ACL / MCL / fracture).
- Higher resolution with this encoder.
- All-data seed models (~3 per pipeline) for the final submission.
