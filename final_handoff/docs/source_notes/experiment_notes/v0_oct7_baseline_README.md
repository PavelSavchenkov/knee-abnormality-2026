# oct7_baseline: the main baseline (R2) and everything built around it

What this folder is:
- The first pipeline: a 2.5D CNN over 3-slice windows from every series, with a per-finding attention head.
- It ran as the "Experiments" session from 2026-10-07 01:00 to ~2026-10-09 01:00. A parallel "Ideas" session reviewed it and
  researched ideas (ideas_archive/).
- It produced the public LB 0.940 submission (R2, 5 folds + synovitis/effusion logit mix) and 0.944 (the same models, with
  synovitis taken from the imghead3 image-read head).
- It is finished. Other experiments use it as the reference: "ours" / "main baseline" in oct8_baseline_v2 (Basin),
  oct9_baseline_2nd (2nd baseline, NFNet-L0) and oct10_baseline_3rd (3rd baseline, RadImageNet R50; its README has the
  3-pipeline summary).

Written 2026-10-10 as a map for later sessions. Details are in the files named below; this README only summarises them.

## Reading order

1. **ledger.md** (708 lines): the full results log.
   - Setup, noise floor, speed table, the 02:16 OOM incident.
   - The results table (rows 1-21: aug, epochs, backbones, labels), then dated notes.
   - Decisions (newest first): LB probes, ensembling final, image labels, reference bias, fold rotation.
2. **ENSEMBLE_LEARNINGS.md**: what ensembling showed and how to measure it honestly.
3. **IMAGE_LABELS.md**: the image-read synovitis labels (Codex reads): are they better, does a model learn from them,
   dose-response. IMAGE_HEAD_INFERENCE.md is the old imghead2 contract (superseded by submissions/oct8_imghead3_5f/README.md).
4. **ideas.md**: the ideas ledger with the priority table, plus sections on labels, validation, model / input, architecture
   levers, BirdCLEF lessons, submission, data.
5. **analyses/*/README.md**: finished side analyses (eda, levers, postproc, pseudo_labels, pipeline_review, text_gold,
   text_embed, attention_probe).
6. **ideas_archive/**: the Ideas session's notes.
   - ideas_loop/summary.md is the best single summary of that session.
   - Also in ideas_loop/: sent.md, research_log.md, reviews.md, error_analysis_2026-10-09.md (main vs Basin errors), and the
     basin_hint*.md hints passed to Basin.
   - ideas/: topic docs: pretrained_zoo, knee_encoders, roi_two_stage, birdclef_lessons, public_recipe_diff, vlm_design,
     in_domain_ssl, report_quote_alignment.

Private files, not to be shared:
- **ideas_archive/private/** holds the pseudo-labelling notes. The user's rule is that PL stays private unless they say
  otherwise.
- **BASIN_ORACLE.md** holds the oracle rules for answering the Basin session. Never show it to a Basin session.

## Final recipe R2 (submitted)

- Backbone: efficientnet_b0.ra_in1k, lr 1e-3 (head 1e-3), wd 0.01, warmup 1, cosine to 0, 20 epochs, bf16 + channels_last, batch
  8 studies.
- Input:
  - canonical cache `data/cache/fov160_px400_q95` (0.4 mm/px, JPEG q95), 140 mm crop resized to 256 px;
  - windows of 3 adjacent slices: k 24 per study in training, 96 at eval.
- Aug (aug_ge): rotation 15, scale 0.15, shift 0.05, erase p 0.3.
- Labels: v1prob (text v1 statuses as probabilities), not_mentioned target 0.15.
- Head: per-finding attention over all windows, with series-type + position embeddings ("attn").
- Score: the epoch is fixed at 19, which is the median-of-best epoch.
- On 5 folds: trusted 0.9722, lookat 0.9520.
- Reference runs (3 per fold, used as "ours" elsewhere): **effb0_lr1e-3_R, sub_b0, sub_b0_s1**.
  - Their oof.parquet / gold.parquet have an `epoch` column; metrics.json has the curve.
- Submission checkpoints: sub_b0 folds 0-2 + effb0_lr1e-3_R folds 3-4, model.pt = epoch 19. **DO NOT DELETE.**
  - sub_b0 also keeps model_ep13-19.
  - imghead3_b0 folds 0-4 = the synovitis image head in the 0.944 package.
- Packages: `submissions/oct7_baseline_5_folds/` (knee_infer.py = the fastest validated inference) and
  `submissions/oct8_imghead3_5f/`.

## Measurement (rules that later experiments inherited)

- Metrics (common/metrics.py):
  - **trusted**: 7 findings on hand-checked cells (primary);
  - **lookat**: 8 findings, trusted + look-at cells;
  - **text**: 12 findings on v1 statuses; text_core leaves out synovitis (3-8 negatives per fold) and MCL (6-13 positives);
  - **gold**: gold-58, the same 58 studies on every fold.
- Noise per run (same config, other seed): trusted 0.0014, lookat 0.0034 (0.0007 with an identical random stream), text_core
  0.0017, gold 0.0032.
  - The study bootstrap ignores training noise; report.py adds it.
  - Compare with same-config replicates (3 references per fold), not with the spread of other configs.
- Screens were run on folds 0-1 and the gate on 2-4. Screen gains shrank 2-4x on unseen folds, from selection luck. Fold 0 got
  73 screens, so it is the most selected fold for this pipeline. After 00:55 on 10-08, screens rotated folds.
- The references were ~+0.005 lucky on gold (the selected config's replicate landed high). Typical variants sit ~0.005 below
  on gold.
- Gold-58 is weak for synovitis: the public probe gave 0.872 where gold-58 said 0.772. Gold gains of a label recipe also
  repeat across folds (corr ~0.5). Use gold only as a veto.
- Per-finding public AUC probes (LB = (AUC + 11 x 0.5) / 12):
  - lateral OA 0.884, fracture 0.908, PF OA 0.920, lateral meniscus 0.944, Baker's 0.968;
  - synovitis 0.872 -> ~0.92 with the image head.

## What worked

- Augmentation + longer training together: aug_ge alone +0.004 (12 ep); 20 epochs + aug_ge +0.011 over base.
- Backbone EfficientNet-B0 at lr 1e-3: +0.0063 trusted over ResNet-34 on folds 0-1, +0.0075 on the gate.
  - B2 = B0 at 1.4x cost; ResNet-18 / 50 behind; LR 2e-3 too high, 1.5e-4 much worse.
  - ConvNeXt-nano (LayerNorm) did not train at 4e-4 and was slow at 2e-4.
- Per-finding attention pooling over windows. Mean / max / LSE pooling lost 0.014-0.018 on gold.
- Synovitis = mean(logit synovitis, logit effusion) at inference: LB 0.938 -> 0.940.
- Image-read synovitis labels via a separate head (imghead3, weight 0.25), synovitis taken from that head alone: LB 0.944.
- Eval windows: flat beyond 96.

## What did not work (each was a specific regime)

- Text-label recipes (~25 variants: sharpen, hard, status heads, site silence, external LLM label sets, kNN, severity): all
  within noise on the report metrics. Soft targets beat hard ones (lab_hard -0.015).
- EMA (cosine to 0 already smooths the end); SWA / checkpoint averaging; AdaBN (BN re-estimated on eval windows: ~0).
- Gaussian noise aug collapsed eval: BN statistics on noisy inputs didn't fit clean ones.
- Mirror aug with M/L swap; window dropout; crop 120 / 160 mm; img 320; k 48; 30 epochs. All flat or worse.
- Sequence models over windows (transformer, bi-GRU, even with identity init): worse or neutral.
- Structure priors from nnU-Net boxes (prior_pred): lookat +, gold - on 5 / 5 folds. Not adopted; a possible member.
- Foundation encoders:
  - DINOv2-L LoRA under-fit (-0.023 trusted);
  - OrthoDiffusion frozen probe weak (0.86);
  - Qwen3-VL-2B LoRA didn't learn in epoch 0 (train_vlm.py);
  - CoAtNet-rmlp-2 at parity with effb0, seed-like as a member.
- Multi-contrast windows (FS + co-registered non-FS slice), soft AUC loss, hard / easy-example weighting: neutral or worse.
  SAM (rho 0.05): report metrics +, gold -, 2x cost.
- Ensembling (ENSEMBLE_LEARNINGS.md):
  - same-pipeline members are seed-like (corr 0.92-0.96 with the references);
  - same-recipe averaging saturates at ~5 models;
  - only prior_pred beats a seed partner on all 5 folds, by a small margin.
  - Diversity has to come from structure (input, labels, model family). That is why oct8_baseline_v2 (Basin) and
    oct9_baseline_2nd were built.

## Files

| file | purpose |
|---|---|
| train.py | train one config on folds: `train.py --name NAME --folds 0 --set key=value ...`; writes runs/NAME/foldF/ and a row in runs/results.csv |
| run_queue.py, queue.txt, queue_slots.txt, queue_done.txt | queue runner (`NAME \| FOLDS \| key=value ...`, N slots) |
| data.py | study dataset over the canonical slice cache (windows, aug, multi-contrast partners, side fixes) |
| model.py | KneeNet: timm window encoder + pooling heads (attn, sattn, pooling variants, seq layers, extra heads) |
| labels.py | training targets from text labels (recipes: v1prob, nm, sharpen, ext:NAME, mix:A+B, ...) |
| predict.py | re-score a saved fold model with other inference settings (k_eval, side fix, AdaBN, ext_avg) |
| report.py, screen.py | run summaries; each run at its median-of-best epoch vs a reference (paired bootstrap + training noise) |
| calib.py | calibration per run (ECE / Brier on trusted cells and gold) |
| ensemble.py, ensemble_rank.py, ensemble_forward.py, member_vs_seed.py | cross-fitted ensembles, seed yardstick, forward selection, the member test beyond a seed |
| img_label_eval.py, img_syn_*.csv, img_effsyn_rank.csv, r2_gold_synovitis.csv | image-label test inputs / outputs |
| partners.py, partners.csv | co-registered contrast partners for multi-contrast windows |
| probe_od.py | frozen OrthoDiffusion probe |
| train_vlm.py | Qwen3-VL / Qwen3.5 VLM classifier (private packages in work/vlm/pylib) |
| series_side_final.csv (+ _v1) | knee-side flip table copy (the live one is data/derived/series_side_final.csv) |
| watch.sh | RAM / pressure / queue-liveness watcher |
| runs/ (106 runs, 1.1 GB), logs/ | per-run outputs and logs; runs/results.csv = one row per run x fold |

Shared code and data: `common/` (folds, labels, metrics, cache, geometry), `data/cache/`, `data/meta/`, `data/derived/`
(side table, nnU-Net slice boxes, Codex synovitis reads).
