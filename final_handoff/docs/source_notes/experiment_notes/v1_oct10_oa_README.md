# oct10_oa: better Medial OA, Lateral OA and PF OA — SEALED 2026-10-11 ~19:10 (paused)

Started 2026-10-10 23:50 (user goal, user_directives.md). Paused by the user on 2026-10-11 ~18:58: nothing runs here now; the user
will give instructions for running the OA approach on more pipelines. Every result: ledger.md (time order); dataset analysis:
analysis.md; ideas and priorities: ideas.md; Kaggle probes: probes.md.

## Final state (2026-10-11 19:10)

- **3-pipeline OA package (2026-10-12, submissions/oct11_oa_3pipe): LB 0.955 (best; the 0.952 package +0.003 = OA AUC sum +0.036
  +-0.012).** All other findings = the 0.952 package; OA = 0.4 x its 3-pipeline OA + 0.6 x the mean of the v2cor experts on effb0
  (5 folds) and ECA-NFNet-L0 (folds 0-2), + de-coupling (make_oa_recipe.py, chosen on folds 0-2 OOF; blend beat replace on 3 / 3
  folds). The R50 v2cor expert was trained (folds 0-2) and left out: no gain on top of effb0 + NFNet.
- **Best OA setting, LB-tested (E3, LB 0.949):** the 0.944 package (main effb0 5 folds for the other findings) with the 3 OA
  columns from the **v2cor compartment-crop expert** alone (runs/oa_crop_v2cor folds 0-4, epoch 19; submissions/oct11_oa_e3).
  LB ladder, only OA changed: main effb0 0.944 -> E1 attention expert 0.948 -> E2 lse expert 0.947 -> E3 v2cor expert 0.949.
- **v2cor recipe** (main/oa_train.py `loc_version=v2cor`, all other defaults): effb0 224 px, localiser crops (out/loc_v2_e2.pt,
  16 structures) in medial / lateral / PF bags, lateral coronal / axial crops mirrored, each OA output attends only to its own bag,
  coronal medial / lateral bags = slices with that compartment's tibial cartilage, centred on it; k 8 train / 32 eval windows per
  bag; meniscus auxiliary heads (men_weight 0.5); 20 epochs, eval epoch 19; ~10 min per fold alone on the GPU.
- **Combining with the main:** OA replaced by the expert (E3 recipe) = blend w 0.5 + de-coupling locally (within 0.0002). Best
  2-member ensemble: a main + the v2cor expert (scripts/two_member.py).
- **Do not stack on v2cor** (GPU, 20 epochs, seed 0; folds noted): clean labels (folds 0-2), init from the main encoder (folds 0-2),
  isolated-cell weighting x3, OA-case upsampling x2, OA-only loss, enhanced main (compartment crops as extra windows in the main,
  free / OA-restricted; = an extra seed once the expert is in the stack) (fold 1 each).
- **Closed earlier:** de-averaging (lse / top-k; LB E2), 288 / 384 px, augmentation changes, residual / PoE (expert and main),
  localiser attention prior, per-site calibration, mirror TTA (details in the Results table and ledger.md).
- **Transfer to the other pipelines' encoders (started 18:35, fold 1 only, not LB-tested):** v2cor expert on RadImageNet R50 (mono, lr 5e-4)
  and ECA-NFNet-L0 (lr 3e-4); text3 / md3 alone: effb0 0.9730 / 0.9723, R50 0.9709 / 0.9721, NFNet 0.9726 / 0.9733 (LOA 0.968 /
  0.975 vs effb0 0.965). In C3 (3-pipeline blend + expert(s) + de-coupling): effb0 alone 0.9756 / 0.9766; + R50 0.9764 / 0.9784;
  + NFNet 0.9769 / 0.9783; all three 0.9771 / 0.9789 (seed pair of the E1 expert 0.9745 / 0.9762; blend noise ~0.0006). Spearman
  vs effb0 v2cor: R50 0.91-0.93, NFNet 0.93-0.95 (seeds 0.95-0.97). Diverse experts add; folds 0 / 2 (and 3 / 4) not run yet.
- **How to continue on more pipelines:** queue lines `oa oa_crop_v2cor_<enc> FOLD backbone=... loc_version=v2cor eval_every=4
  eval_from=15` in queue.txt, run by scripts/gap_runner.sh (yields the GPU to the Synovitis session); evaluate with
  scripts/reeval.py / screen_f1.py (experts) and scripts/member_eval.py / two_member.py (members). Deployment: the Kaggle OA stage
  (submissions/oct11_oa_e3/oa_stage.py) runs effb0 experts; build_package.py refuses mono=True and oa_stage builds the backbone with
  timm.create_model(name), so the R50 expert needs the mono input path plus the radimagenet_resnet50 -> timm resnet50 (ImageNet
  mean / std) mapping (~30 min); a timm encoder such as eca_nfnet_l0 needs no code change.
- **Leads outside the OA goal:** the expert's meniscus heads beat the main on 5-fold OOF (Medial Meniscus text 0.9717 vs 0.9673,
  Lateral 0.9492-0.9513 vs 0.9434; shared with the research session); the enhanced main (crops_free) Contusion +0.012 text (fold 1).

## Question

How to make the three OA outputs better on the expert-labelled test? The public probes put Lateral OA (0.884) and PF OA (0.920)
among the weakest findings. The diagnosis from earlier sessions (work/oa_research/README.md): the OA heads score "degenerative
knee" and cannot tell which compartment has the OA (Lateral OA on lateral-only vs medial-only knees: AUC 0.55).

## Constraints

- GPU: the Synovitis session first; GPU arms only in its gaps. Most work so far runs on CPU (under nice 19, RAM-capped scopes).
- Baseline for experiments: main effb0 R2 (fastest, ~12 min per fold alone on the GPU), copied into main/.
- No Kaggle submission without the user's go: candidates are described for the user in the ledger.

## Metric (scripts/oa_eval.py)

Per OA finding: text (v1 meets vs denied / structure_normal), trusted / lookat (Medial OA, PF OA), strat (AUC inside the
strata of the other compartments' status), comp (compartment test), thr (meets vs below), gold (gold-58, veto only).
Text / trusted share the training labels' co-occurrence and punish de-coupling by construction, so strat / comp / thr / gold
carry the decision for de-coupling methods. Same-config noise (3 refs, 5 folds): text3 sd 0.001, strat3 0.005, comp3 0.004.

## Results so far (details in ledger.md)

| what | result |
|---|---|
| Post-hoc de-coupling of the 3-pipeline blend's OA logits, per-finding strength (C1': F - b_F x linear fit of F on the other two OA logits; b = 0.3 / 0.4 / 0.2 for Medial / Lateral / PF), chosen by nested cross-fitting on text3 + strat3 | held-out: text3 +0.0006, strat3 +0.007, gold3 +0.0035, test-like wtext3 +0.0041 (Lateral OA +0.012); holds for every single pipeline too. Per-site variants lose held-out. Candidate C1' (needs the user's go) |
| CPU localiser (MobileNetV3, 128 px, distilled from nnU-Net boxes) | series box centre error median 2-4.5 mm vs 7-10 mm for fixed crops; checked by eye |
| Label analysis | low-scored OA "meets" are mostly small focal defects; v1 extent / depth attributes separate image evidence inside a status; in gold "meets" depth (full vs deep) matters more than extent |
| Coupling sources | OA outputs agree across pipelines (Spearman 0.88-0.94); tissue brightness falls with effusion (per-series scaling); effusion explains little of OA AUC; degenerative marrow oedema location is a strong compartment cue (LOA meets 55% vs 9%) |
| Pipeline weights for OA | equal mean stays (leave-one-fold-out weights: noise) |
| Compartment-crop expert (CPU, MobileNetV3 160 px, each output sees only its compartment's crops; folds 0-2) | vs blend3 + a 4th main seed: wtext3 / text3 / thr3 up on 3 / 3 folds; pooled +0.0021 / +0.0013 / +0.0021, strat3 +0.0041. Candidate C2 = blend3 + expert (w 0.25) + C1': wtext3 0.9555 vs 0.9490, text3 0.9617 vs 0.9599, strat3 0.9337 vs 0.9174, gold3 0.9116 vs 0.9060 (pooled folds 0-2). Test-time path verified (scripts/oa_expert_infer.py) |
| **GPU compartment expert (effb0 224 px; folds 0-2)** | alone matches the 3-pipeline blend on text3 (0.9598 vs 0.9599) and beats it on strat3 / thr3; blend3 + expert w 0.5 vs blend3 + a 4th main seed: all four metrics up on 3 / 3 folds (text3 +0.0029..+0.0056). **Candidate C3** = blend3 + expert (w 0.5, z-scored) + per-finding de-coupling (b 0.2 / 0.4 / 0.2, nested): held-out wtext3 0.9589 vs 0.9490, text3 0.9646 vs 0.9599, strat3 0.938 vs 0.917, thr3 0.846 vs 0.832, gold3 0.913 vs 0.906 (folds 0-2) |
| Overall OA gain (09:30, pooled folds 0-2) | C3 vs the LB 0.948 package's blend: text3 0.9599 -> 0.9647 (MOA +0.000, LOA +0.007, PF +0.008), wtext3 +0.010, trusted M / P +0.004, gold3 0.906 -> 0.913 (gold LOA -0.006, noise). vs main alone: text3 +0.0135, gold3 +0.018. Expected LB: +0.001..0.003 (3 of 12 findings) |
| GPU expert screens (C3 text3 = blend3 + expert + de-coupling; fold-1 seeds 0.9743 / 0.9731) | split: +0.0007 after de-coupling (3 folds); bml, crop_norm, oa_grade, v2 bags, px288: within noise on f1; px384: LOA alone 0.968 (best) but PF -0.008, net level; lse (instance pooling): worse alone on text, better on gold (2 / 2 folds), and as a 2nd member +0.0003 / +0.0010 text, +0.008..0.014 gold3, because it lifts gold LOA positives the report calls below / silent / normal (A67) |
| Other checks | two arms at once: no throughput gain; CPU expert as an extra member: no gain; crops centred right but 23-44% of coronal / sagittal TF windows and 72% of axial ones lack the target cartilage (A60); attention ~uniform (A57); the expert memorises (train 0.992 vs val 0.968, A61); a 3-slice lesion misses 59% of k-8 training bags (A64) |
| **LB probes (14:45, 18:33)** | on the 0.944 package, only OA columns changed: OA from the attention compartment expert alone 0.948 (OA AUC sum +0.048 +-0.012 vs main effb0), from the lse expert 0.947 (-0.012 vs attention), from the v2cor expert 0.949 (+0.012 vs attention, E3). The expert transfers; de-averaging does not; gold-58's lse advantage was noise (probes.md, submissions/oct11_oa_e1 / e2) |
| **Pipeline-1 OA setting (final 18:35; GPU, 20 epochs, seed 0 per fold)** | **v2cor expert** (coronal medial / lateral bags = slices with that compartment's tibial cartilage, centred on it): 5 folds alone text3 0.9650 vs 0.9614 (4 / 5 folds up); LB E3 0.949 vs E1 0.948. Nothing stacks on it (fold-1 screens, folds 0-2 where noted): clean labels (folds 0-2, C3 text3 -0.0020), init from main (folds 0-2 below v2cor), iso x3, OA-case upsampling x2, OA-only loss, enhanced main (crops as extra windows, free / OA-restricted: = an extra seed once the expert is in the stack). Closed earlier: residual / PoE, localiser attention prior. Best 2-member stack: a main + the v2cor expert (OA replace or blend w 0.5: equal). Next: v2cor expert on the R50 / NFNet encoders (queued 18:36) |
| Evaluation policy (12:00, user: text labels may be wrong) | text agrees with gold-58 only at AUC 0.85-0.92 (A68); models already beat text on gold. Every arm is reported on gold-58 (paired bootstrap vs b0), md3 (meets vs denied: cleanest text negatives) and text3 (sanity) via scripts/reeval.py. A variant stays alive if gold or md3 improves beyond the seed floor even when text drops |
| Candidates | C3 (blend3 + attention expert + de-coupling): text3 0.9647, gold3 0.913. C4 (+ lse expert as a 2nd member): text3 0.9649, gold3 0.918, gold LOA 0.869 (3 folds). Hybrid (attention + lse heads in one model, f1): gold3 +0.013 at one model's cost. Init from the main encoder (f1): best text (C3 0.9766), gold +0.004 |
| Medial vs Lateral OA (A70-A72) | mirroring is correct: bag-swap test shows the MOA and LOA heads are interchangeable readers (<= 0.003). The gap (matched-case parity ~-0.05) is the lateral data: isolated focal / posterior lesions, fewer cartilage slices, noisier labels (gold confirms text LOA meets 5 / 7 vs MOA 6 / 6). Coronal lateral bags were diluted more (44% vs 29% windows without target cartilage): v2cor makes them symmetric (85% / 85%) and halves the parity gap (-0.027 vs -0.046) |

## Layout

- main/: R2 pipeline copy (train.py with poe / oa_only / oa_grade options; labels.py oa_grade) and the OA expert (oa_data.py
  compartment bags from the localiser, oa_train.py GPU trainer with bag-restricted attention).
- scripts/: oa_eval.py (panel), baseline_panel.py, make_proxy.py, posthoc_decouple.py, posthoc_blend3.py, blend_expert.py,
  loc_data.py / loc_train.py / loc_infer.py / loc_viz.py (localiser), oa_crops.py / oa_cpu.py (CPU OA expert), analysis scripts
  (image_qc, side_swap, meta_signal, oa_attrs, oa_size, site_silence, label_suspects), montage.py, chain_cpu1.sh.
- refs/: OOF / gold of the 3 R2 references, NFNet e14, RadImageNet e20 (copies for self-contained comparisons).
- data/: loc_128 (localiser windows), loc_pred (localiser predictions for all slices), oa_crops_160 (CPU expert crops).
- out/: tables and images; logs/; runs/.

## Evidence strength of the negative results (09:35, after the user's question)

Compute used: GPU only since 08:08 (main result: GPU expert folds 0-2; one seed replicate; fold-1 screens). Before that, CPU
(efficiency cores): a MobileNetV3 localiser and MobileNetV3 compartment experts (~80 min per fold), plus post-hoc analyses of
existing out-of-fold predictions (no training). Fold-1 single-seed screens have noise ~0.002 text3 alone / ~0.0006 in the blend,
so "no gain" there means "nothing detectable", not "proven useless".

| claim | test | strength |
|---|---|---|
| residual training < independent expert | CPU expert, fold 1, stopped at epoch 6 of 12, both forms | weak: re-run on the GPU expert (queued) |
| mirror TTA hurts | GPU inference with existing fold models, fold 1 | moderate (large effect, 1 fold) |
| per-site / per-vendor recalibration loses | post-hoc, nested over 5 folds | solid |
| oedema auxiliary head | CPU on par; GPU fold 1 best on every column | open: folds 0 / 2 running (an earlier chat summary wrongly listed it as a failure) |
| per-crop normalisation, graded OA targets | GPU expert, fold 1, one seed | weak (within noise) |
| checkpoint averaging, per-finding expert weights | post-hoc on CPU-expert predictions, 3 folds | moderate |
| nnU-Net cartilage thickness / coverage | feature AUCs, 4,150 studies | solid |
| the 52 analysis areas | CPU: labels, reports, images, existing predictions | descriptive, per area |

## Candidate C3 (best; not submitted: needs the user's go)

- Members: the 0.948 package's three pipelines (folds 0-2) + the GPU compartment expert runs/oa_crop_b0/fold{0,1,2}/model_ep19.pt.
- OA logits: (1 - 0.5) * blend + 0.5 * expert (expert logits z-scored to the blend's OOF mean / sd per finding), then
  F - b_F * (linear fit of F on the other two OA logits), b = 0.2 / 0.4 / 0.2 (Medial / Lateral / PF). Other 9 findings unchanged.
- Test-time expert path: localiser (out/loc_mnv3_e2.pt) on every slice -> series boxes -> compartment bags -> crops -> expert,
  as scripts/oa_expert_infer.py (verified on the CPU expert: max diff 0.002 vs OOF); swap in main/oa_train.py OANet
  (effb0, 224 px, k_eval 32 per bag). Needs the package's side correction (its mirror detector) before localising.
- Reading on the LB: replace the 3 OA columns of the 0.948 package; LB change = sum of the 3 OA AUC changes / 12.

## Candidate C1': how to deploy and read it (not submitted: needs the user's go)

- Code: scripts/c1_postproc.py `apply(logits, coef, b)`; slopes in out/c1_coef.json (fitted on the blend OOF). In the package
  (submissions/oct10_3pipe_3f/ensemble.py) apply it to the mean pipeline logits, before the sigmoid; only the three OA columns
  change. No new models, no runtime cost.
- Reading on the LB:
  - full entry (the 0.948 package + C1'): LB change = (dMOA + dLOA + dPF) / 12; expected ~+0.001 (mostly Lateral OA), at the edge
    of 3-decimal visibility;
  - Lateral OA probe pair (keep only Lateral OA, others 0.5), package without vs with C1': each reads Lateral OA AUC +-0.006;
    expected +0.005..+0.012. Two submissions.
