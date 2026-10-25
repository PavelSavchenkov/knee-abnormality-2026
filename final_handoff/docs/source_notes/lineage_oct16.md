# Submission lineage, 0.938 -> 0.961 (2026-10-16)

Public LB on 10-16 15:58 UTC: we are 26th of 6,095 at 0.962. Public notebooks sit at 0.958 (411 teams; 0.954 on 10-12).

## Steps

| # | Step | Submission | LB | Gain | Classes changed |
|---|---|---|---|---|---|
| 0 | effb0, 5 folds | oct7_baseline_5_folds | 0.938 | | all |
| 1 | Synovitis = logit mean with Effusion | oct7_5_folds_synpp | 0.940 | +0.002 | Synovitis (+0.024 AUC) |
| 2 | Synovitis from a model trained on Codex image reads | oct8_imghead3_5f | 0.944 | +0.004 | Synovitis (+0.048) |
| 3 | 3 backbones (effb0, NFNet-L0, R50) x folds 0-2 | oct10_3pipe_3f | 0.948 | +0.008 vs 0.940 | all |
| 4 | Step 2's Synovitis approach on the 3-backbone package | oct10_3x3_synprob05 | 0.952 | +0.004 | Synovitis |
| 5 | OA crop experts (60% share) + Lateral OA de-coupling | oct11_oa_3pipe | 0.955 | +0.003 | OA classes (summed AUC +0.036) |
| 6 | Synovitis blend G (2 image models + 4 crop experts) | oct12_syn_g | 0.955 (higher hidden score) | <= +0.001 | Synovitis |
| 7 | Menisci from the OA experts' meniscus heads (60% share) + Fracture / Contusion bone experts | oct12_global_best | 0.957 | +0.002 | both menisci, Fracture, Contusion (not separable) |
| 8 | Window / crop fusion + Effusion image-read model (30% weight) + folds 3-4 | oct13_f5_win | 0.961 | +0.004 | Effusion, Baker's, OA, bone, Synovitis crops; folds 3-4 ~0 |
| 9 | Retrain everything on all data (3-5 seeds) | oct15_all_data_s3 / _s5 | 0.962 | +0.001 | all |

By lever:
- Backbone ensemble: +0.008, about a third of the total.
- Image-read labels: +0.006 for Synovitis (steps 1-2), plus part of step 8 through the Effusion image-read model.
- Experts (OA, menisci, bone, Synovitis crops): about +0.006.
- Fusion: about +0.003.
- All data: +0.001.

Gave about nothing:
- TTA;
- folds 3-4;
- every text label set;
- feature KD;
- label-diverse ensembles;
- a ConvNeXt main model (+0.0002 in the package);
- pseudo-labels: inner-world gains of +0.003, never ported.

## Public AUC per class, as far as we know

Every measurement comes from a single-class probe on an older package, +-0.006 each (oct7_synovitis_probe, oct9_probe_*,
oct11_synbase_pair). No class has been measured on the 0.961 package.

"Rough now" is an estimate: the later steps applied to each class, constrained to add up to the 0.961 total. Each value could be
off by 0.01-0.02.

| Class | Measured | On package | Changed since | Rough now |
|---|---|---|---|---|
| Lateral OA | 0.884 | 0.944 | 3 backbones, OA experts, de-coupling, OA crop fusion | 0.90-0.92 |
| Fracture | 0.908 | 0.944 | 3 backbones, bone expert, bone fusion | 0.92-0.94 |
| PF OA | 0.920 | 0.944 | as Lateral OA | 0.93-0.95 |
| Synovitis | 0.848 -> 0.872 -> 0.932 | 0.938 / 0.940 / image-model pair | G, crop fusion (G scored above the single image model) | 0.93-0.94 |
| Lateral meniscus | 0.944 | 0.944 | 3 backbones, OA experts' meniscus heads, crop fusion | 0.95-0.96 |
| Baker's | 0.968 | 0.944 | 3 backbones, window fusion | 0.97-0.98 |
| ACL, MCL, medial meniscus, Medial OA, Effusion, Contusion | about 0.964 on average (not probed one by one; worked out from the LB) | 0.944 | 3 backbones, experts, Effusion image-read model, fusion | about 0.98 on average |

- The 0.938 Synovitis value is derived: 0.872 minus 12 x the step-1 gain.
- The Lateral OA splice probe (57273283: 0.961 with Lateral OA from the OA experts at 64 eval windows instead of 32) scored 0.961,
  above the 0.961 package by hidden precision: Lateral OA gains between 0 and +0.012 AUC. It measures the change, not Lateral OA's level.
  The same change on the 0.962 all-data package (57283796) scored below that package. So the 64-window effect is about 0, and its sign
  depends on the package.
- Lateral OA with one extra OA-expert member trained on calibrated labels for report-silent cells (silL, 57284893) scored above
  0.961, above the 64-window splice and above the ConvNeXt Synovitis probe. Locally it raised gold and left the report reads flat:
  on the LB, Lateral OA follows gold.
- Synovitis probes on the 0.961 package: ConvNeXt in both Synovitis crop slots (57283240) scored above 0.961 by hidden precision, a
  Synovitis gain between 0 and +0.012 AUC. Fat-sat-only image heads (57279368) scored below 0.961.
- Single-class probes on the 0.961 package would replace these estimates. That costs one slot per class, and needs the user's go.
