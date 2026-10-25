# oct10_synovitis: synovitis from the Codex image reads in the 3 pipelines

Started 2026-10-10 ~21:00 (user goal in user_directives.md). Status: DONE 2026-10-11 10:15. Pipeline #1 finalised, transferred to
NFNet / R50; final package submissions/oct10_syn3x3 = public 0.952 (0.948 3x3 package + 3-pipeline image-head synovitis). GPU handed
to the OA session at 08:08. Details: ledger.md.

## Question

What is the best way to use the image-based synovitis labels (Codex reads, 3,566 non-gold studies) in each of the 3 baseline pipelines
(main effb0 R2, NFNet-L0, RadImageNet R50), and does a 3-pipeline x 3-fold ensemble of image-head synovitis beat the 0.944
submission's synovitis (imghead3: main effb0, 700 reads, 5 folds)?

Time budget: the goal allowed ~3 h of tuning; at 22:00 the user lifted it ("work for 24h ... until you reasonably exhaust levers
on pipeline #1"), and later set the order: exhaust pipeline #1 (labels first), transfer to #2 / #3, check vs 0.948 on Kaggle, hand the
GPU to the OA session (user_directives.md).

Analysis of experiments/v0 and its 3 baselines (the first step of the goal): v0_analysis.md.

## Layout

- main/: main pipeline (copy of experiments/v0/oct7_baseline train / data / model / labels; DEFAULTS = R2; cls_w; img_* columns saved).
- basin/: Basin-code pipelines 2 / 3 (copy of experiments/v0/oct10_baseline_3rd/src; --img_w / --img_csv / --cls_w; img_Synovitis saved).
- data/: export snapshot (synovitis_labels_codex.csv, 2026-10-10 07:18, 3,574 studies), image targets img_syn_blockrank.csv,
  img_syn_prob.csv, img_syneff_blockrank.csv.
- refs/: predictions of the v0 reference runs (R2 refs, imghead3, NFNet, RadImageNet), copied for self-contained comparisons.
- scripts/: run_queue.sh (queue.txt: `main|nf|rad NAME FOLD ARGS`), eval_syn.py (synovitis metrics per run), blend_syn.py (blends),
  memguard.sh, watchdog.sh.
- runs/: this folder's runs (m_* main, n_* NFNet, r_* RadImageNet); logs/: per-arm logs, events.txt.
- cache -> experiments/v0/oct8_baseline_v2/cache (Basin s320 slice cache); weights/radimagenet/ResNet50.pt.

## Metric

Held-out reads: AUC of a synovitis score vs the Codex read (p >= 0.5) on validation-fold studies (~725 per fold, never trained on in
that fold). Gold-58 synovitis (27 positives, SE ~0.06) is a veto only. The public LB is the final judge (0.944 = imghead3 image head;
synovitis change = 12 x LB difference, +-0.012).

## Results (2026-10-11 08:20; details in ledger.md)

Pipeline #1 (main effb0), image head alone (blending with the text synovitis head or the effusion mix always hurt), folds 0-2 pooled:

| synovitis from | held-out reads | clear-cut reads | gold-58 (test-like) | gold eff+ | public (replace-column probes) |
|---|---|---|---|---|---|
| R2 + effusion mix (0.940) | 0.870 | 0.966 | 0.784 | 0.612 | 0.940 package |
| 0.948 package's synovitis (main mix + NFNet + R50 text heads) | 0.877 | 0.972 | 0.802 | 0.643 | 0.948 package |
| imghead3 (700 reads, block rank, w 0.25) | 0.900 | 0.983 | 0.859 | 0.783 | **0.944** (f0-2 on the 0.944 base; 0.944 with 5 folds) |
| all 3,566 reads, block rank, w 0.05 | 0.905 | 0.986 | 0.877 | 0.850 | **0.945** |
| all reads, raw read probability, w 0.05 (prob05) | 0.914 | 0.988 | 0.875 | 0.825 | **0.945**; on the 0.948 base: pending |
| all reads, chat-calibrated probability (cal05) | 0.914 | 0.989 | 0.870 | 0.811 | - |
| all reads, raw attribute score (attrp05) | 0.913 | 0.988 | 0.872 | 0.836 | - |

Transfer (same recipe: raw read probability, image weight 0.05), folds 0-2 pooled, image head alone:

| pipeline | held-out reads | clear-cut | gold-58 (test-like) | vs its own text synovitis (reads / gold) |
|---|---|---|---|---|
| main effb0 (m_prob05) | 0.914 | 0.988 | 0.875 | +0.08 / +0.13 (R2 main head) |
| NFNet-L0 (n_prob05, e14) | 0.920 | 0.990 | 0.872 | +0.046..+0.054 / +0.068..+0.082 per fold |
| RadImageNet R50 (r_prob05, e20) | 0.911 | 0.987 | 0.851 | +0.035..+0.041 / +0.019..+0.057 (folds 0-1) |
| 3 pipelines, equal logit mean | 0.920 | 0.990 | 0.875 | vs the 0.948 package's synovitis: +0.043 [+0.035,+0.052] / +0.073 [+0.004,+0.157] |

Public LB: oct10_3x3_synprob05 (0.948 package + main image head) **0.952**; oct10_syn3x3 (0.948 package + 3-pipeline image heads)
**0.952** (synovitis +0.048 +-0.012 vs the 0.948 package; the 3-pipeline gain over the main head is below LB resolution).

Ensembling and more-data gains (folds 0-2 pooled, paired study bootstrap vs the main image head alone, x1e-3 [95%]):

| synovitis from | held-out reads | d reads vs main only | gold-58 | d gold |
|---|---|---|---|---|
| main image head, 700 reads (imghead3) | 0.900 | -14.3 [-20.1,-8.6] | 0.859 | -15.5 [-64,+28] |
| main image head, all reads (0.952 package) | 0.914 | | 0.875 | |
| main + a second main seed | 0.916 | +2.6 [+0.3,+4.9] | 0.875 | 0 |
| main + NFNet | 0.921 | +7.1 [+4.1,+10.3] | 0.879 | +4.8 [-27,+36] |
| **3 pipelines (final package)** | **0.920** | **+6.0 [+2.3,+9.7]** | 0.875 | 0 [-36,+33] |

More reads: +14.3 on held-out reads (significant); ensembling the 3 pipelines: +6.0 (significant, more than a seed's +2.6); gold-58
cannot resolve either. On the LB both 3x3 packages score 0.952: a +0.006 synovitis gain moves the 12-class LB by 0.0005, below its
3-decimal resolution, so the LB neither confirms nor contradicts the ensembling gain. Main + NFNet alone is within noise of the 3-way;
the 3-way was kept as the user's design.

Findings:
- The training reads are noisy vs the host labels: training-style Codex re-reads of gold score 0.81 AUC (careful reads 0.878).
  Stronger image weight copies the reader (held-out reads up, gold down); weight 1.0 also hurts the other findings; dropping the text
  main loss (image head only) hurts gold (0.785 vs 0.88): multi-task text targets denoise the reads.
- Block ranks (per-chat percentile targets) are not comparable across chats with different positive rates (b1-8 55%, b19-28 13%):
  weak reads in low-prevalence chats got near-positive targets. Raw targets fix it (+9 reads vs block rank, +14 vs imghead3); chat
  calibration and synovial attributes add nothing measurable on top. Public: all reads +0.012 +-0.012 synovitis vs imghead3.
- One LB probe resolves only +-0.012 synovitis AUC: label variants within ~0.01 are not separable on the LB.
- Gold false positives are effusion-positive knees without synovial thickening / debris (host labels follow synovial changes, not
  effusion size); a target with thickening / debris / Hoffa did not beat the raw probability over 3 folds.
- No concentrated harmful-read subset: strong model-read contradictions are 17 of 2,150, spread over chats; late-chat reads are fine.
- Text synovitis is a site reporting style: mention rate 0.5-28% by site, uncorrelated with read positives; precision vs reads
  0.38-1.0 by site; templates co-report effusion and synovitis at some sites. Test pending: drop the text synovitis task (m_nosyn05).
