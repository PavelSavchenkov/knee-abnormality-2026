# oct13_deep_ensembling_test: deeper ensembling of the 0.957 package without retraining

Goal (user 2026-10-13 ~13:45): improve the best package (submissions/oct12_global_best, LB 0.957) by ensembling its trained models on a
deeper level (experts x pipelines x folds), no retraining; find the options most likely to raise the Kaggle LB; compile a submission
(no official submit without the user's go). User 14:10 (via review session 8e): try the window-level fusions P4 / P6 on the LB, check
on gold.

## Summary (2026-10-13 ~17:00)

- **LB result 2026-10-14 00:44: submissions/oct13_f5_win scored public 0.961 (0.957 -> +0.004; submission 57154196, kernel v3,
  262 min).** Commit v1 was OOM-killed on Kaggle (cuDNN per-batch-size kernels + knee_infer's pinned cache; fixed in v3, see the
  package README). Folds 3-4 are likely ~0 of it (A2 revisited) and cost ~60 min of runtime: next candidate = the same fusions on folds 0-2.
- PL session cross-check (10-14 ~01:00, experiments/v2/oct12_pseudo/out/fusion_k6r.txt; inner world, 300 fold-3 studies + gold-58,
  9 models, geo|pm|g20|wmean on Effusion + Baker's): K6r - B0 without -> with fusion: trusted +1.4 -> +1.6, text +1.5 -> +1.8, gold_nofc
  +5.3 -> +4.4, Codex effusion +9.4 -> +3.4; fusion gain on B0 vs K6r: Codex effusion +7.4 vs +1.4, gold Effusion +8.7 vs +5.0,
  Baker's +9.1 vs +3.6. The PL / clean-label stage and the fusion fix largely the same image-read effusion errors (so does the
  img_eff blend here): a PL transfer into oct13_f5_win should add its report / gold_nofc part, not its Effusion part.

- Study-level options are exhausted: equal logit mean across the 3 pipelines is right; fitted weights, fusion functions, recipe weights
  and seed-like extra heads add nothing. Two free study-level gains: the fold 3-4 checkpoints and the crop experts' unused image-read
  Effusion head -> submissions/oct13_f5_eff.
- Window / crop level (no retraining, exact per-window / per-crop decompositions of the attention-MIL heads): a relevance-gated
  probability fusion (top-k windows by mean attention, mean window probability) helps FOCAL findings and every expert family whose
  members share crops: 3x3 Effusion / Baker's (Codex reads +4.5 / +9.1, 5-fold cross-fitted), OA experts (report +0.74 held-out, gold
  Lat OA +17), bone experts (gold F +12.5, C +10.8), synovitis crop experts (Codex +1.75). It hurts diffuse 3x3 columns (MCL, Lat OA /
  LM in the 3x3) and pushes report-silent negatives up when the gate is used on Synovitis in the 3x3.
- Coarse pre-registered specs chosen leave-one-fold-out on image references transferred better than the full grid (user's rule).
- submissions/oct13_f5_win = oct13_f5_eff + all the window / crop fusions; local gates pass (DICOM path = dump rebuild, corr >= 0.999);
  gold-58 0.9419 -> 0.9480 (local), OOF reports flat, Codex reads up. Estimated LB vs 0.957: oct13_f5_eff +0.0005..+0.002, oct13_f5_win
  +0.0015..+0.004. Kaggle checks need the user's go.

## What the package is (members x pipelines x folds)

| column(s) | members (folds in the package) | combination |
|---|---|---|
| ACL, MCL, Effusion, Baker's | 3x3: main effb0 sub_b0, NFNet-L0, R50 (f0-2) | logit mean over pipelines of the fold-mean logit |
| OA x3 | 3x3 base + OA crop experts effb0 (f0-4) + NFNet (f0-2) | 0.4 base + 0.6 z(expert mean) + de-coupling |
| MM, LM | 3x3 base + the OA experts' meniscus heads | 0.4 base + 0.6 z(...) |
| Fracture / Contusion | 3x3 base + bone experts b_v3, b_lr3 (effb0), b_nf (NFNet) (f0-2, k64) | F 0.1 base + 0.9 z(...), C experts only |
| Synovitis | image heads m_prob05 / n_prob05 + synovitis crop experts s_fs / s_fc125 on effb0 + NFNet (f0-2) | 1/3 image heads + 2/3 crops |

Unused but trained: fold 3-4 checkpoints of main / NFNet / R50 / b_v3 / b_lr3; the crop experts' Effusion / Baker's / img_eff heads
(computed in the same pass, discarded); the image-head models' 11 text heads; R50 synovitis crop experts; per-window outputs.

## Evaluation

- OOF folds 0-4 (one fold model per study per member): weights / combination functions; folds 3-4 = independent check of recipes chosen
  on folds 0-2. OOF cannot show fold averaging (ENSEMBLE_LEARNINGS: OOF gains overstate test gains).
- gold-58 (every fold model predicts it = test-like, small, veto only); inner world (review session: fold-3 studies held out by all
  inner models, 3 backbones x 3-4 inner folds) for fold-count / window-fusion questions.
- Codex image reads (Effusion / Synovitis) as an image-truth proxy; the LB is image-read truth (synovitis image heads: LB +0.048 AUC).
- lib.py rebuilds the package exactly from member logs (OOF trusted 0.9807 / lookat 0.9613 = describe.py; gold with OA effb0 5 folds).

## Results (x1e-3 unless AUC)

| # | option | evidence | verdict |
|---|---|---|---|
| A1 | crop experts' img_eff head -> Effusion, w 0.3 (a1_free_heads.txt, eff_recipe.txt) | folds 0-2 OOF: Codex Effusion +18.4, trusted Eff -0.3, lookat Eff -2.2; gold Effusion +7.5; LOFO w 0.4-0.5 | in candidate (free) |
| A1 | crop experts' Effusion text head | lookat +0.5, Codex +1.1, gold +3.7 at w 0.2 | weaker than img_eff |
| A1 | crop experts' Baker's head | negative at every w | no |
| A2 | fold 3-4 checkpoints (a2_folds_recipes.txt) | gold-58 +1.4 (MCL +4.5, Lat OA +5.8, F +4.2, C +4.0, Baker's -3.6); inner world 3 -> 4 folds +0.3 trusted / +0.4 lookat / +1.5 gold_nofc (c27) | in candidate (runtime +~25%). Revisited (user 10-14 00:30): the 3rd baseline's fold_count test already showed saturation (gold-58, 3 pipelines x k: k3 0.9293, k5 0.9292); half of the +1.4 is the bone experts (F / C), the 3x3 half is 2-3 pair flips per finding (noise); c27 inner world also mixed (Codex eff -1.1); Kaggle cost measured: est 208.5 -> 268.5 min (+60, Basin +1.5 s / study, bone +0.9): over the user's 25-min rule for no detectable gain |
| A2 | recipe w on folds 3-4 | OA 0.6 holds (Lat OA prefers 0.4-0.5 on both fold sets, ~+1-2 on one finding); men 0.6 holds; C 1.0 flat; F 0.9 fails on f3-4 (-2.1 vs w 0, peak 0.3) -> 5-fold optimum ~0.6-0.7, +0.6 F | no change (below resolution) |
| A3 | 3x3 pipeline weights / fusion (a3_pipeline_weights.txt) | cross-fitted global / group / per-finding weights: trusted -0.1..-0.2, lookat -0.1..-0.2; prob mean +0.1; z +-0; rank -0.3..-0.5 | equal logit mean stays |
| A3 | image-head models' unused 11 text heads (seed-like) | +0.07 trusted / +0.35 lookat OOF (1 -> 2 models per study, i.e. mostly averaging) | no (seed-like, never beat 'replace') |
| W | window-level cross-backbone fusion (review C46-C48, C47 confirm on 9 inner models) | inner world: reports flat (trusted -0.0..+0.3, lookat -0.7..-0.9), gold +5.3 (fracture +36, MCL +18), Codex Eff +7..+9 | see W2 |
| W2 | the same on the package's own members (dump_main.py / dump_basin.py: exact decomposition, gold-58, 9 and 15 models; all slices and the deployed T_eval 24) (eval_windows_gold.txt, a5_windows_bare.txt) | bare 3x3 gold macro: F (mean attention, prob-mean values, top-20 gate) -0.1..+1.6, P4 -0.7..+0.7, P6 -0.2..-1.9; Fracture replicates (+11..+28), MCL does not (-4.5..+2.3), LM -5..-17, Lat OA -2..-17. Inside the full package: gold macro -1.2..+0.9 for every variant (F is 0.9 bone experts; Effusion turns negative next to the img_eff blend) | no (no gain for this package; not worth a toolchain change / LB slot) |

### Gold flips of window fusion, read by hand (a6_window_gold_flips.txt; 15 deployed models, T_eval 24, mean attention / prob-mean / top-20)

- Helps focal findings: Fracture +11.1 (11 pairs fixed, 5 of them report-'meets' positives that the study mean ranked low: focal
  evidence diluted by averaging), Baker's +9.1, Effusion +5.0 (all 7 fixes = gold-negative report-'below' small effusions pushed down),
  Medial OA +4.7.
- Hurts diffuse / compartment findings: Lateral OA -7.7, Lateral Meniscus -5.0; MCL 0, ACL +1.2.
- Main failure mode: report-silent gold negatives pushed up (29 broken pairs come from 'not_mentioned' negatives, 23 of them in
  Synovitis, -11.9; the package takes Synovitis from image models, so this loss would not reach it).
- So a coarse per-finding rule is the only plausible form: fuse focal findings (Fracture, Baker's, Effusion), keep the study mean for
  the rest. Effusion / Baker's can be checked at scale on Codex reads (w_search.py); Fracture lives in the bone experts (crop-level
  fusion inside them: dump_bone.py).

### Window-fusion iterations (user 14:45 / 14:47: report metrics misleading / at ceiling; cross-fit; coarse choices; read flips; iterate >= 2 h)

| round | idea | evidence | reading |
|---|---|---|---|
| 1 | consensus operators against the silent-negative failure: window value = geometric mean / min over backbones, relevance = attention agreement (prod) (wfuse.py, w_gold.py -> w_gold_round1.txt; gold read only) | silent-negative breaks: gated (k 10-20) ~30 (23 of them Synovitis), k 40 16, no gate 7-9; gm / min / prod do not reduce them (min 41); package columns: 7 (gated) vs 2 (no gate) | the top-k gate, not the value operator, causes the failure; it is almost all Synovitis (not a 3x3 column in the package). Stable signs over all operators: Baker's +5..+14, Effusion >= 0, Fracture +6..+15 (gated pm), Lat OA < 0 |

| 2 | OOF fold 0 first look (w_search_f0.txt; 855 studies, triplets; image refs = Codex reads) | Codex Effusion +2..+4.6 (on top of the img_eff blend), Codex Baker's +3..+9; MCL lookat -12..-39, Effusion lookat -5..-10, ACL ~0; 'geo' = 'prod' relevance for one model per backbone (dropped) | per-finding rule again: fuse Effusion / Baker's, keep ACL / MCL deployed |
| 2b | Codex-read flips, fold 0, geo/pm/top-20 (a7_f0_geo_pm_g20.txt) | Effusion reads +6.0 (977 pairs fixed / 279 broken): image-visible effusions reported 'below' (335) or 'denied' (214) move up, read-negative 'below' move down (222); Baker's reads +5.3 (9 / 1), report-'unsettled' cysts the reads confirm move up | fusion moves toward image truth where reports are vague or wrong: the focal-evidence mechanism |
| 3 | gate size on Codex reads, folds 0-1 (1,726 studies; a7_f01_gates.txt; bare 3x3) | no gate: Effusion -0.5 / Baker's +4.8; top-20 (geo / pm): +5.1 / +10.7; top-40: +2.4 / +8.9 | the top-k gate makes the Effusion gain (fixes: read-positive 'below' 965 and 'denied' 750 pairs up, read-negative 'below' 861 down); k 20 > 40 > none |
| 4 | cross-fitted search, folds 0-2, inside the package (Effusion with the img_eff recipe on top; w_search_f012.txt) | Effusion: LOFO picks top-10/20 + top-3 every fold, held-out Codex +3.7; Baker's: LOFO held-out Codex +13.3 (folds +2.0 / +15.5 / +22.4) | gains hold out of fold on image refs |
| 4b | coarse family check, folds 0-2 in-package | top-20 gate + weighted mean, any relevance (mean / geo) x value (pm / um), a 1.0: Effusion Codex +2.5..+3.0 (all folds > 0), Baker's +12..+14.6 (all folds > 0); report: Eff lookat -4.5..-4.8, trusted -1.1..-1.4, Bak lookat -1.5..-2.3; a 0.5 roughly halves both | coarse pick: geo / pm / top-20 / wmean, a 1.0, Effusion + Baker's only |
| 4c | read flips folds 0-2 (a7_f012_winners.txt; bare 3x3) | geo/um/g20: Effusion reads +5.6 (fixed 8,289 / broken 2,555; 'denied' read-positives up 2,660), Baker's +16.0 (391 / 74; 'unsettled' read-positive cysts up 166) | same mechanism on 2,000+ read studies |
| 6 | full 5-fold cross-fitted search inside the package (w_search.txt, out/w_search_metrics.parquet; 4,349 OOF studies; folds 3-4 without the img_eff recipe) | LOFO coarse set: Effusion picks mean/pm/g20/maxmix on 5/5 folds, held-out Codex +4.7 / report-vague +4.9; Baker's picks geo/pm/g20/wmean 5/5, held-out +9.1. LOFO over the full grid (360 options): +4.3 / +7.5 (coarse transfers better, as the user expected). geo/pm/g20/wmean a 1.0 per fold: Effusion Codex +3.1 / +2.2 / +3.6 / +9.6 / +4.1, Baker's +5.3 / +13.3 / +19.9 / +2.9 / +4.2; Effusion lookat -1.1..-5.8 on all folds; ACL lookat +0.5 (mean); MCL -4.6 | decision: geo/pm/g20/wmean, a 1.0, Effusion + Baker's (ACL / MCL keep the study mean) |
| 5 | crop-level fusion inside the bone experts (dump_bone.py: per-crop s / u of b_v3, b_lr3 f0-4, b_nf f0-2 on identical crops, rebuild vs logged k64 <= 0.0012; w_bone.py gold; never fitted on gold) (w_bone_gold.txt) | 13 deployed models in the F / C recipe: top-20 gated: Fracture +11..+24, Contusion +8..+12 (83% of 108 specs > 0); coarse geo/pm/g20/wmean: F +12.5, C +10.8; no gate F +5.6 / C 0; lse + top-10 fails (F -25..-29) | same gated-fusion pattern on a focal finding; OOF floor pending |
| 5b | bone gold flips by hand (a9_bone_gold_flips.txt; geo/pm/g20) | Fracture fixed 19 / broken 10: report-vague positives up (unsettled 7, hedged 4), report-'meets' gold negatives down (4); broken = report-confirmed fractures down (6). Contusion fixed 10 / broken 2 (report-'meets' gold negatives down 6) | moves toward the hand-read truth where reports are vague / wrong |
| 5c | bone fusion recipe (make_fuse_recipe.py, fuse_recipe.txt): fused e aligned to the deployed e's OOF distribution (folds 0-2) | OOF report floor text Fracture -2.4 / +7.0 / -3.7, Contusion -0.3 / +4.4 / +0.8 (folds 0-2); gold (13 models) F 0.9375 -> 0.9500, C 0.9501 -> 0.9609; full candidate oct13_f5_win gold macro 0.9422 -> 0.9448 | in oct13_f5_win |
| 7 | window fusion of the synovitis image heads (m_prob05 all windows + n_prob05 T_eval 24, img_Synovitis; Synovitis G = 1/3 heads + 2/3 crops; w_syn.py, w_syn_f0.txt; fold 0, 855 studies) | best of 180 specs: Codex synovitis +0.2 / report-vague +0.5 (noise level); image-head part alone -0.6; the package already scores 0.987 on the held-out synovitis reads | no gain detected (one fold; remaining dumps skipped) |
| 8 | crop-level fusion inside the synovitis crop experts (dump_syn.py: s_fs / s_fs_nf / s_fc125 / s_fc125_nf f0-2, img_syn / img_eff; all four crop lists identical; w_synx.py, w_synx.txt; Synovitis G with the crop part fused) | Synovitis Codex reads: best coarse design / mean / pm / g20 / maxmix a 1.0 +1.75 (report-vague +2.2), folds +0.7 / +2.0 / +2.5, LOFO held-out +2.0 (same pick every fold); img_eff -> Effusion: LOFO held-out -0.2 | small real gain (~+0.0001-0.0002 LB, reads already 0.988); added to oct13_f5_win after all (zero runtime: the per-crop path replaces the forward; gold Synovitis 0.8853 -> 0.8865); img_eff fusion no |
| 9 | crop-level fusion inside the OA compartment experts (dump_oa.py: effb0 v2cor f0-4 + NFNet v2cor f0-2 on identical crops; w_oa.py, w_oa.txt; fused term replaces the experts' mean e in the OA + meniscus recipes) | OOF folds 0-2 report metrics (trusted / lookat / text over the 5 columns) +0.5..+0.9 for every gated spec, LOFO held-out +0.74 (picks geo/pm/g10); gold-58 in the package: Lateral OA +17.4 (0.849 -> 0.867), PF OA +7.7, MM / LM +1.2, Medial OA -4.7 | in oct13_f5_win (report-positive, unlike the 3x3 window fusion) |
| 9b | Lateral OA gold flips (a10_oa_gold_flips.txt) | 17 fixed / 5 broken: 6 report-'meets' positives up, 8 report-silent positives up, 2 report-'unsettled' negatives down; broken: 4 report-'normal' positives | mechanism looks real (unlike A8's contusion heads) |
| 10 | ACL into the 3x3 window fusion (free) | per fold trusted ACL +0.4 / -1.0 / +2.0 / -1.1 / +0.4 (geo/pm/g20); gold +1.2..+2.5 | no (mixed: no evidence) |
| 11 | recipe weights after fusion (a11_recipe_retune.txt): OA / meniscus w 0.5-0.8 with the fused OA term, Fracture w 0.8-1.0 with the fused bone term | LOFO held-out vs the packaged w: OA -0.02, F -0.03 (packaged 0.6 / 0.9 stay optimal); gold mixed | no change |
| 12 | weights next to the fused parts (a12_syn_eff_weights.txt; Codex reads, folds 0-2) | Synovitis image_heads_w: 1/3 stays (alternatives held-out -0.26); Effusion img_eff w: 0.4 > 0.3 on reads (+3.4 held-out, all folds) | no change: the img_eff head is trained on Codex effusion reads, so the read metric rises with its weight all the way to w 1 (A1: +31.6); w 0.3 was the report / gold balance (lookat -2.2 vs -3.7 at 0.4) |
| 13 | full-package OOF (a13_package_oof.txt; folds 0-2, every fused part from its dumps) | oct13_f5_win vs oct13_f5_eff: reports flat (trusted +0.3, lookat -0.5, text +0.8), Codex Effusion +3.0, Synovitis +1.75, Baker's +12.8 | the profile of the earlier image-truth wins (0.952 synovitis head) |
| 14 | per site / manufacturer (a14_sites.txt) | manufacturers: trusted -0.7..+0.7, lookat -2.7..-0.2, Codex Effusion +1.4..+4.5; no site with a systematic loss (largest site swings +-4 on 60-160 studies) | no robustness flag |
| 15 | package gold flips by hand (a15_pkg_gold_flips.txt; local DICOM runs) | fixed: report-silent positives up 18, report-confirmed positives up 17, report-'meets' gold negatives down 15 (F / C over-calls); broken: report-silent negatives up 14 (12 Synovitis, where gold under-rates synovitis), report-confirmed positives down 5 (F) | net: Lat OA +23, F +18, LM +11, C +11, PF +9, Baker's +9, Syn -2.4, Med OA -6.2 |
| 16 | R50 synovitis crop experts (s_fs_rad / s_fc125_rad, not in the package) as a 3rd backbone of the per-design crop fusion (a16_syn_r50.txt) | Codex synovitis -0.23 / report-vague -0.33 (2 of 3 folds < 0); gold +4.8 | no (no gain at scale; +6 models of runtime) |
| 17 | per-finding OA recipe w with the fused OA term (a18_oa_perfinding_w.txt; user 17:05 rule: consistent gains go into the probe unless disproportionately tiny or > ~25 min runtime) | Lateral OA w 0.5: +0.5 / 0.0 / +0.4 per fold, LOFO held-out +0.23 on one finding (~+0.00002 LB), gold 0; Medial / PF OA: 0.6 stays (held-out -0.09 / -0.10) | no (disproportionately tiny) |
| 18 | other coarse bone fusion specs vs the packaged geo/pm/g20/wmean (a19_bone_specs.txt; OOF text F / C, LOFO) | switching: held-out -0.85; maxmix (best on gold, F +8..+10) loses on the report floor in every fold (-2.4..-3.5); g40 mixed | no change |
| 19 | main study logits from every window (free in oct13_f5_win's window pass) instead of the 96-window grid, all columns (a20_main_allwin.txt; OOF folds 0-4) | per-fold deltas within +-0.2 (mean trusted +0.05, lookat -0.04, MCL lookat -0.58), Codex +-0.1, gold equal | no (no gain detected) |
| 20 | Fracture recipe base (0.1) from the 3x3 window fusion (a21_frac_base.txt; free) | text F per fold +0.6 / -0.7 / 0.0; gold +1.4 | no (not consistent) |
| 21 | Basin on EVERY slice (a17 fusion only, a22 whole package: Basin study logits change for every column; ~+20-25 min Kaggle) | fusion only, folds 0-1: Codex Effusion +0.03 / +0.36, Baker's +0.66 / +3.6; whole package folds 0-1: trusted -0.10, lookat -0.38, text -0.10, Codex Effusion +0.2, Baker's +2.1, gold macro -0.7 (Effusion -3.7) | folds 0-2 whole package: reports ~0 (trusted -0.06, lookat -0.24), Codex Effusion +0.27, Baker's +1.84 (3/3 > 0), gold -0.7 -> ~+0.0001 LB for ~+20-25 min and a change to every Basin column: no (disproportionately tiny, user rule 17:05) |
| 22 | trust check (a23_trust.py, out/a23_trust.txt; user evening): gold split of oct13_f5_eff + paired bootstrap (2,000 resamples of the 58 gold studies, local DICOM runs) | oct13_f5_eff offline: folds 3-4 alone +1.4 macro, img_eff alone +0.6 (Effusion +7.5), both +1.7 (Effusion +3.7 next to 5 folds); DICOM runs 0.957 -> f5_eff +0.6 [90%: -2.1, +3.4], P(>0) 0.63. f5_eff -> f5_win +6.2 [+1.3, +12.0], P(>0) 0.98: Lat OA +23, F +18, LM +11, C +11, PF +9, Baker's +9 (each P(>0) 0.85-0.90), Med OA -6.2 (0.04) | f5_eff's case rests on Codex reads + inner-world folds (gold cannot see it); f5_win's gold gain is beyond resampling noise, but the bone part was included mainly on gold (selection not in the interval) |
| A8 | bone experts' contusion-location heads as free signal (a8_bone_loc_heads.txt, a8b_loa_flips.txt): Medial / Lateral / PF OA, ACL | Med / PF OA, ACL lose on report and gold at every w; Lateral OA: text -1.5..-18.6 but gold +15..+29, carried by 2 report-silent gold positives that the Codex lateral-OA read also calls negative (0.04 / 0.02), while traumatic lateral contusions on report-normal negatives move up | no: contusion is trauma-vs-degeneration confounded |

RAM incident 2026-10-13 15:27 (VS Code scope oom-killed, 61.8 GB): w_search.load() sliced `z["s"]` of an NpzFile per study, which
re-reads the whole array each time and keeps it alive through the slice (~10 GB per backbone per fold process), run as Pool(5)
inside VS Code's scope. Fixed (arrays read once; all scripts run in `systemd-run --user --scope -p MemoryMax=...`).

## Candidate submissions

| package | content | local evidence | est. LB vs 0.957 | status |
|---|---|---|---|---|
| submissions/oct13_f5_eff | 0.957 package + fold 3-4 checkpoints (main / NFNet / R50 / b_v3 / b_lr3) + Effusion img_eff recipe (w 0.3) | OOF flat; Codex Effusion +18.4; gold-58 0.9406 -> 0.9422 (local DICOM run 0.9419) | +0.0005..+0.002 (Effusion part 0..+0.0013 depending on whether public effusion labels follow image reads) | local gates pass; Kaggle check needs the user's go |
| submissions/oct13_f5_win | oct13_f5_eff + 3x3 window fusion for Effusion / Baker's (win_stage.py + basin_predict export) + crop-level fusion of the OA experts (OA / menisci, g10), of the synovitis crop experts (per design, maxmix) and of the bone experts (F / C) | Codex Effusion +4.5, Baker's +9.1 (5-fold cross-fitted); OA report +0.74 held-out; gold F +12.5, C +10.8, Lat OA +17, PF +8; gold macro 0.9422 -> 0.9468 (offline) | +0.001..+0.003 on top of oct13_f5_eff | local gates PASS for the first build (16:11, gold 0.9449) and the OA build (16:26, gold 0.9482 local DICOM); final build (+ synovitis) run_local 16:26 |
