# oct14_text_labels: more usable information from report text, metadata and sites

Started 2026-10-14 ~23:10 (user): lay the ground work for later GPU training that extracts as much useful information as possible from
the report text + metadata + sites to improve generalisation; >= 5 diverse directions, each analysed per class of the 0.961 package
(submissions/oct13_f5_win). Details came from the Pseudo Label Research session (knee-abnormality-2026-29; ledger 23:15). Later user
asks, the same night: lower silent defaults? how to use site phrasing and mild-mention clues? does v4 combine everything (the
landscape of uses)? how much can METADATA-ONLY post-processing add to 0.961? any small usable signal in reports / metadata? was a
site flag ever learned by a model?

GPU: none used beyond ~1.5-6 GB smoke tests. Every arm waits in queue.txt behind gpu_go.txt = 0 until the user or the GPU owner
grants the GPU. Ledger: ledger.md. Gold-58 is a veto only here: never fitted, never a promotion criterion.

## REREAD 10-15 ~22:10 (user: re-think older learnings; the report references are v1-built and inaccurate)

Protocol: image truth = gold-58 (veto only, never fitted) + Codex (held-out; not independent for Codex-calibrated sets). Report reads
(trusted / lookat / text / report-explicit cells) = agreement with v1. One more caveat: for the non-Codex classes my 'image reference'
was the v1-TRAINED generalists' OOF > 0.5, so it is v1-shaped too (circular for silent / vague cells, where those models were trained on
v1's fills).

Gold positives per v1 status (all 58 gold studies; scratchpad gold_status.py):

| finding | meets | unsettled | below | not_mentioned | denied | structure_normal |
|---|---|---|---|---|---|---|
| ACL | 20/20 | 1/2 | 1/9 | 0/4 | 0/2 | 0/19 |
| MCL | 2/2 | 7/9 | 0/7 | 0/7 | 0/1 | 0/31 |
| MM | 22/25 | 1/2 | 1/7 | 1/3 | 0/11 | 0/7 |
| LM | 16/18 | 1/2 | 1/7 | 1/8 | 0/10 | 2/9 |
| MOA | 6/6 | 6/10 | 0/2 | 0/16 | 0/5 | 1/17 |
| LOA | 5/7 | 3/9 | 1/2 | 1/20 | 0/4 | 1/16 |
| PF OA | 9/9 | 7/8 | 1/8 | 3/13 | 0/5 | 1/14 |
| Effusion | 9/10 | 16/19 | 10/21 | 0/2 | 0/5 | |
| Synovitis | 13/17 | | | 14/40 | | |
| Baker's | 4/4 | 5/7 | 2/5 | 0/30 | 0/10 | |
| Contusion | 12/17 | 2/7 | | 1/10 | 1/4 | 0/5 |
| Fracture | 10/14 | 2/6 | | 4/23 | 0/5 | 0/2 |

| claim (where) | original basis | re-read on image truth | verdict |
|---|---|---|---|
| silence ~ negative (A2) | OOF > 0.5 rate of v1-trained models (circular) | gold silent positives: ACL 0/4, MCL 0/7, MOA 0/16, LOA 1/20, Baker's 0/30, Effusion 0/2; but Synovitis 14/40, Fracture 4/23, PF OA 3/13 | holds for ligaments / MOA / LOA / Baker's; REVERSED for Synovitis, Fracture, PF OA (silence is not absence) |
| 'unsettled' ~ 0.6 is fine; calibration may lower it (consnp 0.53-0.55, rules 0.30 for ungraded PF) | image ref (v1-shaped) | gold 'unsettled' is mostly positive: MCL 7/9, PF OA 7/8, Effusion 16/19, Baker's 5/7, MOA 6/10 | lowering 'unsettled' targets goes AGAINST gold (explains consnp / rules PF loss). Veto-level observation only: do not fit labels on these counts |
| 'below' (mild mention) ~ negative | v1 0.2 | gold: Effusion 10/21 positive (others 0-2 of 5-9) | Effusion 'below' is ~half positive on gold; other findings hold |
| calibrated sets beat v1 at label level (D1) | image ref (fitted, v1-shaped) + gold label level | gold label-level gains (consnp 0.877 -> 0.902) are the independent part | holds at label level; GPU: member-level gains, ~0 in the package |
| metadata post-processing not robust (A8 / A13c) | Codex-fitted offsets 'hurt report-explicit cells' | report-explicit = v1 agreement. Image truth: Baker's held-out Codex +7.0..+10.0 (3 / 3 folds), gold +12.7; Effusion Codex +4.4..+4.7 (3 / 3) but gold -3.7 | REVERSED for Baker's (re-opened: DICOM-site offsets, no training); Effusion undecided |
| per-group screen table (a17) | text + lookat with silent negatives | superseded by the nosil table; nosil is still v1 agreement | superseded |
| vag15 leads report reads (LABEL_SETS) | 1 seed x 2 folds | seed 2 on f3: tr9ns +1.9 -> -0.7, menisci +2.6 -> -7.0; Codex effect repeats across seeds | the lead was largely a seed draw; label effects on image reads are stable, report-read gaps sit in seed noise |
| A14 report-at-inference ceiling | gold | gold-based | holds (PF OA / MCL carry untapped report information) |
| A4 package error drivers | report statuses as 'truth' for FP / FN | partly report errors (ERRORS.md found report misses) | weakened: re-derive on gold / Codex before using |

Re-prioritised:
1. Pipeline stability first (queued): tl_v1 reruns + batch 32 / EMA. Every label comparison is inside seed noise on reports.
2. Judge label sets on image truth: gold per group + Codex where independent; report reads = v1 agreement only.
3. Re-opened, no training: Codex-fitted DICOM-site offsets for Baker's on 0.961 (held-out Codex 3 / 3, gold +). LB only on the user's
   ask (research mode).
4. Label design hints (veto-level, never fitted on gold): do not lower 'unsettled' targets (MCL, PF OA, Effusion, Baker's, MOA). Keep
   silent targets for Synovitis / Fracture / PF OA well above zero. Silence ~ absent for ligaments / MOA / LOA / Baker's. These match
   the image-calibrated sets (v4b's silent Syn / F levels) and contradict rules / consnp on PF.
5. Expert hand-off: Synovitis / Fracture / Baker's text heads get the gold-consistent silent handling (calibrated, not uniform).

## Short answer

1. **Most of the usable report information is not in what the models train on today.** Three mechanisms carry image-side information,
   and none is used in the 0.961 members' targets:
   - (a) For silent findings, the other findings' statements predict the image (context). Silent cells are ~negative for most
     findings, yet the trainers fill them with 0.15.
   - (b) Sites differ in what a statement means: Effusion 'unsettled' is image-positive in 30-88% of cells depending on the site.
     Sites also differ in how often they say nothing.
   - (c) Fact clues inside mild mentions (grade, edema, partial tear, location).
2. **Calibrated labels (D1) put (a)-(c) into targets, without v3 and without gold.** Lead recipe = v5 = v4 + two fixes (silent cells
   kept just above explicit negatives: v4b; bounded re-ranking inside explicit cells: v5). v4 numbers, label level, held-out folds:
   - Codex Synovitis 0.626 -> 0.763, Effusion 0.828 -> 0.842, Baker's 0.759 -> 0.807.
   - Inside mild cells, agreement with the image models rises from ~0.5-0.6 to 0.71-0.79 (ACL, OA x3).
   - Gold veto passes on 10 / 12 findings (LOA -9e-3, LM -4e-3).
   - Built per fold, ready to train.
   - Correction (A10): do NOT lower silent cells below the explicit-negative targets (uniform 0.03 is worse than v1 on the image
     reference). v4b keeps them just above; v4 and v4b both go to the stage arms.
   - v5 adds re-ranking inside report-explicit cells (A11): image-reference all-cell AUC vs v1 ACL +3, MCL +2, MM +15, LM +3, MOA +9,
     LOA +22, PF OA +21, Effusion +17 (Codex), Synovitis +137 (Codex), Baker's +48 (Codex, vague), Contusion +27, Fracture +8.
3. **The site bias is measurable, and it is the bias that image reads removed for Synovitis.** Cross-fitted site offsets fitted on
   Codex reads raise:
   - the text-trained generalists' Codex AUC by +16 (Effusion), +18 (Synovitis), +18 (Baker's) x1e-3, 5 / 5 folds;
   - the exact 0.961 package: still +4.7 Effusion and +9.7 Baker's; Synovitis 0, because its column trains on image reads.
4. **Metadata-only post-processing of 0.961 has no robust gain** (A8, A12, A13 and the A13c cross-reference check).
   - Expected LB ~0 +- 0.0005, sign unknown. This revises my first '<= +0.001' estimate.
   - DICOM-site offsets fitted on Codex reads help Codex (Effusion +4.8, Baker's +9.7 x1e-3) but hurt the report-explicit cells
     (-4.8 / -3.4, 3 / 3 folds). Fitted on explicit cells, they hurt Codex (-18.6 / -22.9).
   - Per protocol cell, gradient-boosted residuals and coarse single fields (field strength, vendor, fat-sat series, 3D, slice
     thickness, pixel spacing, FOV, coil, position) are not robust.
   - Patient sex looked robust until the rare 'other / unknown' levels were removed.
   - Not recommended. data/postproc/ is kept for reference only.
5. **Report words and style beyond the structured facts** add almost nothing.
   - 38 of 48,489 word tests pass FDR 5%, nearly all site-template phrases.
   - Dense text embeddings add nothing over the v1 / v2 features in 17 / 19 strata.
6. **Fact-level auxiliary heads (D2) target the package's actual errors.** False positives come from co-occurring findings (ACL /
   MCL / Fracture <- lateral bone bruises; OA <- meniscal tears / extrusion); false negatives are small subtypes (peripheral / small /
   root meniscal tears, high-grade patellar chondromalacia). 305 report facts are ready as targets; the trainer's 22 v1-attribute
   heads were never run.
7. **Site flags in models were never tried.** A test-time site input would re-learn report habits, including the silent-fill
   artefact. The usable forms are fixed labels (D1), a training-only site term (D3, built) and post-hoc image-read offsets (D6).

## Evidence (CPU, all scripts in scripts/, outputs in out/)

| id | question | result |
|---|---|---|
| A1 / A1b | does the package carry site offsets that images contradict? | Codex-fitted cross-fitted site offsets (language x scanner): package 0.957 recipe Effusion +13.3, Baker's +17.3 (3 / 3 folds); with 0.961's img_eff Effusion +5.9; text-trained generalists Effusion +16.1, Synovitis +17.8, Baker's +18.5 (5 / 5); Synovitis column (image reads) +0.2. Caveat: Codex's own positive rate on report-denied effusion varies by site (0.00-0.34) |
| A2 | what does silence mean? | silent cells' image-positive rate (OOF > 0.5) ~ denied cells for ACL, MCL, LM, MOA, LOA, Baker's, Contusion, Fracture (0.5-3.6%); gold silent positives: Fracture 4/23, PF OA 3/13, Synovitis 14/40. Silent share per site 0 -> 0.6-0.9 for every finding; with fill 0.15 the site-mean target tracks the site's silent share (MCL r 0.58, Baker's 0.38, Fracture 0.32). Context score inside silent cells: AUC 0.80-0.89 (OA, F, C, Syn), top-10% 7-24% positive, bottom-50% 0.2-1.3% |
| A7 | can the site be known at test time (no report)? | yes: DICOM header tokens -> site (43 groups) CV accuracy 0.972, language 0.976; 88% of studies at p >= 0.9 with accuracy 0.998 |
| build_targets | label recipes, held-out label-level read | see D1 below; out/build_targets.txt |
| A11 | gaps of v4b | inside explicit 'meets' a clue model ranks the image reference (LOA 0.88, MOA 0.80, MM 0.78, PF 0.77, Contusion 0.73, Effusion 0.72, LM 0.70, ACL 0.69) and inside explicit negatives for PF OA 0.86, Effusion 0.71, MM 0.65 -> v5. Protocol metadata adds nothing to the silent, explicit or vague clue models (deltas ~0) |
| A12 / A13 / A13c | learned / coarse metadata post-processing (user) | per-cell offsets, gbm residuals, one coarse field at a time with nested selection; cross-reference check: Codex-fitted site offsets hurt report-explicit cells and vice versa; sex gains came from rare levels -> no robust metadata correction |
| A14 | thought experiment: report available at inference (ceiling of label work, not deployable) | package + v5 fixed blend on gold-58: macro +4.5e-3 at w 0.2 (90% [-2.2, +11.9]); MCL +27, PF OA +23 (+52 at w 0.5), Baker's / Synovitis +7, LOA +6, MM +5; Contusion -20, Fracture -6, MOA -2. Reports alone beat the package only on MCL, PF OA, ACL. Codex stacker: Effusion +1.1, Synovitis -0.6, Baker's +9.0. => untapped report information sits in PF OA and MCL; for Effusion / Synovitis / Contusion / Fracture / MOA text can only de-noise |
| A10 | why does v4 lose gold on Lateral OA / LM? (review 29) | one gold-positive silent LOA study dropped below 28 explicit negatives: v4's silent level 0.03 sits BELOW the 'structure_normal' targets (~0.045), while the image reference ranks silent cells above explicit negatives in 11 / 12 findings. Uniform low silent fill (nmc) is worse than v1 on the image reference (all cells: MCL 0.925 -> 0.917, LOA 0.901 -> 0.887, Contusion 0.901 -> 0.889, Fracture 0.836 -> 0.810). Fix v4b (silent >= explicit-negative mean, level = that + 0.02): vs v4 on the image reference ACL +2, MCL +9, LOA +5, Contusion +5, Fracture -5, rest equal; gold veto mixed (LM back to v1, LOA -6, MOA -17 vs v1). Both v4 and v4b queued as stage arms |
| A8 / A8b | metadata-only post-processing of 0.961 | exact 0.961 OOF (2,584 studies): DICOM-site offsets on Codex: Effusion +4.4..+4.7 (3 / 3 folds, shrink 10-50; gold -3.7), Baker's +7.0..+10.0 (3 / 3; gold +12.7); offsets of disjoint fold sets correlate 0.52-0.55; report-fitted forms: gold Fracture -103, LOA -56, Effusion -60, Synovitis -35 |
| A4 | which facts drive the package's errors? | FP: ACL <- lateral tibial / femoral marrow edema (x4-8), hemarthrosis; MCL <- lateral condyle contusion (x10-15), complete ACL tear (x6); Fracture <- lateral femoral condyle contusion (x29); MOA <- medial meniscus extrusion (x13) / tear; LOA <- LM tear (x6) / degeneration; Effusion (Codex) <- report 'moderate' (x16). FN: MM peripheral (x11) / small tears; LM small / root; PF OA grade-3 / high-grade patellar chondromalacia |
| A5 | does the report say where? | stated location among 'meets': menisci horn 81-82%, Contusion / Fracture bone 97-98%, PF OA 89%, LOA 70%, MOA 64%, MCL 76%, ACL 42%. Key-image refs ('(5/22)'): 460 mappable refs in 148 studies (96% resolve to a real series / slice), mostly German sites; menisci ~100 studies, OA 66, contusion 50 |
| A6 | do dense report embeddings add information? | TF-IDF word + char -> SVD 256 vs the 683 structured features: no gain in 17 / 19 strata; no transformer encoder in the rsna env |
| A9 | any small usable signal in report words / style / metadata? | words: 38 / 48,489 tests at q < 0.05, almost all site templates (es / tr) with tiny effects; usable: LM vague 'signal change' / 'degeneration' wording -> image-negative (-0.10). Style (hedging, measurements, prior exam, clinical question): nothing. Metadata vs package-minus-Codex: 3D sequences -> package over-calls effusion (-0.56 logit, q 1e-5); axial PD-FS present -> under-calls (+0.28); Baker's Siemens -0.81 (site-specific protocols; site offsets cover them) |

Earlier evidence reused (other experiments): uniform silent fill 0.05 (lab_nm05, main, folds 0-1): lookat +5.2 / +1.9, gold ~+6.5 /
+2.6 x1e-3, never deployed. PL's clean-label stage (explicit cells, 4 epochs) gave the inner world's largest LB step (B0 0.943 ->
B0_xF 0.945). Image-read targets for Synovitis gave LB 0.940 -> 0.944 from one column. Single-fold site rules: findings-only
sites fill (lab_fo) lookat +3.8 / gold +1.8; site silence fill -1.7 / 0 / -1.6. Post-hoc OA site offsets fitted on text: lost.

## Work plan: one direction at a time (user 10-15)

Split by intervention point. Each direction changes its own set of 0.961 columns, is read by swapping its member(s) in the
package (scripts/eval_arm.py vs the deployed / seed row), and can be LB-probed by replacing only those columns (user's go per
submission). Order = proven first, then expected value per GPU hour. Gains below are my forecasts over 0.961; PL numbers are
inner-world LB measurements (oct12_pseudo: B0 0.943 -> B0_xF 0.945 -> K6r 0.946).

| # | direction | what | columns | evidence so far | forecast | GPU | ready |
|---|---|---|---|---|---|---|---|
| A | clean-label / calibrated-label STAGE on the deployed generalists | st_x (main xF) + st_x_r50 (R50 xF; NFNet gets no stage), then st_x_v5 and the label variants st_* against st_x | all 12 via the 3x3 (fully ACL, MCL, Effusion, Baker's) | xF = +0.002 LB in the inner world (the most reliable PL result; mask is what works); PL forecast inside 0.961 ~+0.001 (the effusion / synovitis part overlaps img_eff / fusion / G) | +0.001..+0.002 incl. variants | ~3 h | yes (queue block A) |
| B | PL students | NFNet students (ct blend, no teacher synovitis, e16) + main students (ct) + 1-4 epoch teacher-confidence stage; teacher = deployed 3x3 OOF (T_3p recipe) | all 12 via main / NFNet | +0.001 LB over xF in the inner world (not separable); teacher ceiling: a package holding the teacher's pipelines gains ~one ensemble-size step; R50 students hurt | +0.0003..+0.001 | ~4 h | yes (block B, commented) |
| A' | from-scratch training on A's winning labels (+ site term / fact heads) | tl_* | as A | only if a variant beats st_x | +0..+0.001 over A | ~4 h | yes (commented) |
| C | Fracture / Contusion | bone experts on A's labels (replacing v3), impaction / contusion-with-fracture conventions, F-C de-coupling | F, C | ERRORS.md P1 / P2 | +0.0003..+0.001 | ~5 h | labels yes; bone stage / conventions not built |
| D | OA / PF | OA experts on A's labels; PF bag covering the trochlea; LOA wording | MOA, LOA, PF, MM, LM | ERRORS.md P4 (trochlea-only 0.54 vs 0.63-0.67), LOA conventions | +0.0005..+0.0015 | ~5 h | labels yes; PF bag not built |
| E | MCL expert (new model on nnU-Net MCL boxes) | MCL crops so the score stops following the trauma pattern | MCL | isolated MCL 0.24 vs 0.35 with ACL / contusion | +0.0003..+0.001 | ~4 h + build | no |
| F | pipeline robustness | side handling for flipped / other-knee series; studies without sagittal FS | LOA, PF, LM, ACL (~6% of studies) | AUC 0.86-0.91 vs 0.96-0.98 on flagged studies | +0.0002..+0.0006 | small | no |

Also from oct12_pseudo, kept out of this list:
- **Data beats steps** (60% -> 80% data +0.005 LB): the All Data session's lever. Every direction's final artifact must be redone
  on the all-data models if those get deployed. Stages are cheap fine-tunes of each checkpoint; PL students need all-data student
  runs.
- **Still-rated ideas from the review session:** S6-style 4th generalist (forecast +0.001..+0.004); lab_mix4 as a member.
- **Closed there, not to repeat:** uncertain-cell down-weighting from scratch, deeper heads, attention / class / feature KD, window
  distillation, R50 students, round-2 teachers, NFNet stages.

Rules carried over: median-of-best epoch on each run's own validation; same-config seed rows; promote on the package swap
(Codex Effusion read on Codex AND explicit cells) with gold-12 as a veto; discount best-of-N; gate any LB probe on all 12 findings.

## Label variants (user 10-15: diverse reasonable ways, each built on a different assumption)

scripts/build_targets.py (v4 / v4b / v5) and scripts/build_variants.py; data/targets/{name}_f{0-4,all}.csv, fitted per fold (outside
the fold, never gold); out/build_variants.txt. All keep the trainers' file format (StudyInstanceUID + 12 findings in [0, 1]).

| variant | assumption ('a reasonable label is ...') | how |
|---|---|---|
| v5 | what the image references say, using every report / site clue | silent context + site x status rates + bounded clue re-ranking of vague and explicit cells, fitted to Codex / generalists' OOF ranks |
| v5ns | the same, but site effects are reader / reference quirks | v5 without site features |
| own | direct evidence only; co-occurrence is a shortcut | only facts about the finding's own structure (no co-occurrence, context, site); silent cells constant |
| txt | the reports themselves, no image model (references are circular) | models fitted on the report's own explicit cells, applied to silent / vague cells |
| rules | the official definitions, applied by hand (no fitting) | v2 facts (grade, size, depth, extent, acuity, cause, tear type, assertion) -> hand-set values per class |
| site | v1 statements, corrected only for per-site leniency | per (finding, status, site) odds shift of the image-positive rate, shrunk |
| consnp | the average of independent readers (v1, v5, txt, rules, site) | logit mean |
| cons | consnp + two public text-label tables | logit mean (gold read not clean: the public tables may be tuned to gold) |

Label level, held-out folds (x1e-3 vs v1 where marked; image-reference columns favour v5 / v5ns / own / site, which are fitted to it):

| | v1 | v5 | v5ns | own | txt | rules | site | consnp |
|---|---|---|---|---|---|---|---|---|
| Codex Effusion | 0.828 | 0.845 | 0.841 | 0.826 | 0.835 | 0.822 | 0.846 | 0.851 |
| Codex Synovitis | 0.626 | 0.763 | 0.764 | 0.631 | 0.704 | 0.753 | 0.690 | 0.781 |
| Codex Baker's (vague) | 0.759 | 0.807 | 0.770 | 0.737 | 0.712 | 0.733 | 0.797 | 0.814 |
| image reference, mean of 12 | 0.865 | 0.891 | 0.888 | 0.870 | 0.872 | 0.879 | 0.878 | 0.898 |
| gold-58 mean (veto) | 0.877 | 0.895 | 0.894 | 0.884 | 0.884 | 0.890 | 0.892 | 0.902 |

Provenance (user 10-15: every read of an arm trained on these sets must state it). Validation reads: R = report-derived (trusted /
lookat / text / tr|), C = Codex Effusion / Synovitis / Baker's, G = gold-58.

| set | fitted toward | biased reads for an arm trained on it |
|---|---|---|
| v1 | (source of R) | R |
| v5, v5ns, site, own, v4* | Codex (Eff / Syn / Baker's), generalists' OOF ranks (others; weak fold dependency, user-accepted) | C for Eff / Syn / Baker's; R mildly (v5 explicit re-ranking); G slightly (v4b fix gold-motivated) |
| txt | the reports' own explicit cells | R mildly |
| rules | nothing (hand rules from the official definitions) | none |
| consnp | v5 + site + txt + rules + v1 | C for Eff / Syn / Baker's; R mildly; G slightly |
| consv3, v3 | + v3 (prompt saw gold-58 summary statistics) | G not independent; C as consnp |
| Codex-target arms | Codex reads | C fully |

Class-specific strengths (label level): rules best on Fracture (image ref 0.876) and PF OA (gold); site best on MCL (0.938); txt best
inside silent Medial OA (0.812); own on Fracture / LM; v5 / consnp on Synovitis, Effusion, Baker's, OA. Changes vs v1 correlate
0.19-0.52 between variants (v5 / v5ns 0.92): genuinely different label sets, also usable as diverse ensemble members.
Queue: every variant as a Tier A stage arm (st_*), v5 and consnp from scratch in Tier B.

## Directions and ways to use them

Every direction can be used in several modes. Mode letters used in the class table:
- S: soft targets from scratch.
- F: short fine-tune (stage) of the deployed checkpoints.
- X: expert retrain (OA / bone / synovitis crop experts).
- H: auxiliary heads.
- T: training-only term.
- P: post-hoc on 0.961 logits.
- M: extra ensemble member.

| id | direction | information used | modes | built | status |
|---|---|---|---|---|---|
| D1 | calibrated labels v5 (lead; v4 / v4b kept as arms) | (d) v5: explicit cells re-ranked inside the stratum within bands, for ACL / MM / LM / MOA / LOA / PF / Effusion / Contusion (meets) and MM / PF / Effusion (negatives); (a) silent cells (v4b: level >= explicit-negative mean + 0.02) -> cross-fitted context model (other findings' statuses + v2 facts / attributes + v2 context + site) shifted to a per-finding level (0.03 negative-like findings; 0.10 MM, PF OA, Fracture; Codex rate for Eff / Syn); (b) vague cells of Effusion / Baker's -> site x status Codex rates shrunk to language and status; (c) vague cells -> bounded clue re-ranking inside each status (alpha 0.5, status bands); explicit cells = v1 | S, F, X, M | data/targets/{nmc, sil, vsite, vclue, v4, v4ns, v4b, v5}_f{0-4,all}.csv; main + Basin trainers take tgt_csv | queued (st_v4, st_v4b, st_v5, st_sil, tl_v5, tl_nmc, tl_sil) |
| D2 | report-fact auxiliary heads | 305 v2 facts / attributes (>= 50 studies): sub-locations, grades, sizes, tear types, co-findings, mimics (degeneration, low-grade sprain, chondromalacia), context | H (main / experts) | data/aux_facts.csv (+ _cols.csv); main trainer aux_csv / aux_weight / aux_cols | queued (tl_aux05) |
| D3 | training-only site term | site (language x scanner) per study; offsets per (site, finding) on all cells, or per (site, finding, silent / vague group) on non-explicit cells only ('vague' = the user's concern: misleading probabilities of silent / uncertain cells absorbed by the site term, not by the image pathway) | T | main trainer site_csv / site_mode / site_l2 | queued (tl_sitev) |
| D4 | text localisation | stated sub-site / horn / bone (64-98% of positives) -> crop / bag targets in the experts; key-image refs -> slice-level attention targets (148 studies) | H, X | data/key_image_refs.csv; location facts are in aux_facts.csv (study level) | slice-level loss not built (coverage 3%) |
| D5 | dense report embeddings | multilingual report embedding as an auxiliary regression target | H | no (no encoder in env; A6 proxy adds nothing) | low |
| D6 | metadata post-processing / conditioning | DICOM-only site id; protocol cells; coarse fields; sex | P | data/postproc/ (reference only) | NOT recommended: no correction survives the cross-reference check (A13c); conditioning inputs not built |
| D7 | label-recipe heads / members | the same recipes as extra 12-output heads (heads4 recovered ~80% of a 4-model blend) or separate members | M, H | main trainer head_recipes (existing) | after D1 reads |

Ways to use the same information (the user's 'landscape'):
- different target values (D1 S);
- confidence weights: down-weighting uncertain cells from scratch failed in PL; silent masking failed;
- filtering + fine-tune on reliable cells: PL xF worked; D1 F = fine-tune on calibrated targets;
- auxiliary heads (D2 / D4);
- training-only nuisance terms (D3);
- test-time correction (D6);
- ensemble diversity (D7).

## Per class (0.961 package)

LB AUC from the 10-09 per-finding probes on the 0.944 package (headroom; experts improved OA / F / C / menisci since): LOA 0.884,
Fracture 0.908, PF OA 0.920, LM 0.944, Baker's 0.968, the other 6 ~0.964 implied. D1 numbers are label-level (held-out folds;
gold = veto only).

| class | 0.961 owner | what hurts now (A2 / A4) | D1 v4 | D2 heads | D3 / D6 site | D4 location | priority |
|---|---|---|---|---|---|---|---|
| ACL | 3x3 generalists | FP from lateral bone-bruise patterns, hemarthrosis; ~0.99 already | mild cells: image agreement 0.59 -> 0.79; silent 0.03; gold +6 | bruise-location and hemarthrosis heads separate the mimic | little (explicit statements dominate) | location 42% | low (headroom) |
| MCL | 3x3 generalists | silent fill drives site targets (r 0.58); FP from ACL tear / lateral contusion | silent 0.03, gold +4 | contusion / ACL-tear heads | D3 relevant (silence habit) | 76% | medium |
| Medial Meniscus | 0.4 3x3 + 0.6 OA experts (crop fusion) | FN small / peripheral tears; FP with OA / effusion; site leniency of 'below' 0.10-0.33 | below 0.52 -> 0.64; silent context 0.66; gold +12 | tear-type (peripheral, small, root) heads | small | horn 82%, key images ~100 studies | medium (needs the OA expert retrain for 0.6 of the column) |
| Lateral Meniscus | as MM | FN small / root tears; FP when MM torn | below 0.52 -> 0.66; gold -4 (veto borderline) | tear-type heads; 'signal change' mimic | small | horn 81% | medium-low |
| Medial OA | 0.4 3x3 + 0.6 OA experts | FP from medial meniscal tear / extrusion (x13) | mild 0.52-0.59 -> 0.74-0.76; silent 0.67; gold +11 | extrusion / meniscal-tear heads (de-coupling by supervision) | small | sub-site 64% | medium (OA expert retrain) |
| Lateral OA | 0.4 3x3 + 0.6 OA experts | weakest LB class; FP from LM tear / degeneration | mild 0.51-0.61 -> 0.77-0.79; silent context 0.82; gold -9 (silent part -16) | LM tear / degeneration heads | small | sub-site 70% | high headroom, uncertain sign: test sil vs vclue separately |
| PF OA | 0.4 3x3 + 0.6 OA experts | FN high-grade chondromalacia; silent positives on gold 3/13 | mild 0.55-0.59 -> 0.71-0.73; silent 0.77; gold +22 | chondromalacia grade / extent heads | small | patella / trochlea 89% | high |
| Effusion | 3x3 window fusion + img_eff | report 'moderate' FP vs Codex; site structure: Codex and explicit report cells disagree (A13c) | v5: Codex 0.828 -> 0.845 (site x status rates + clues + explicit re-ranking); gold +19 | small | D6 not robust; D3 for the generalists | - | medium; read arms on Codex AND explicit cells |
| Synovitis | G = image heads + crop experts (Codex reads) | column already image-trained; generalists' synovitis site-biased (+17.8) | silent context: Codex 0.626 -> 0.763 (generalists only) | - | none for the column | - | low for 0.961 (could replace the crop experts' 0.5 v1 text part) |
| Baker's | 3x3 window fusion | silent fill site habit (r 0.38); vague cells site-dependent | silent level + site rates (vague): Codex 0.759 -> 0.807; gold +43 | - | D6 not robust (Codex-fitted offsets hurt explicit cells -3.4) | - | medium |
| Contusion | 3x3 + bone experts (b_v3 on v3 F / C) | FP from complete ACL tear / trauma pattern | silent context 0.90; gold +6 | ACL-tear, location heads | small | bone 97% | medium (bone experts already use v3 context) |
| Fracture | 3x3 + bone experts | weak LB class; FP from lateral condyle contusion (x29); silent positives on gold 4/23 | silent context 0.74; gold +49 (label level) | contusion-location heads separate the mimic | small | bone 98% | high (bone expert on v4 F instead of v3) |

Reading: D1 helps most where silence and vague mentions dominate and context is informative: Fracture, PF OA, Synovitis
(generalists), MOA, Baker's, Effusion. D2 targets the co-finding confusions (ACL / MCL / Fracture, MOA / LOA). D3 / D6 matter for
the findings whose site habits differ (Effusion, Baker's, MCL). For the expert-owned columns (OA x3, menisci, F / C), D1 reaches 0.961
only through expert retrains on v4 targets (mode X).

## Ground work built

- scripts/base.py (shared tables), features.py (683 text / site features, data/text_features.parquet), pkg_lib.py (package rebuild,
  copied from oct13_deep_ensembling_test with data/config.json, members.parquet, eff_recipe.json, oof_f5_win.parquet,
  expected_gold_f5_win.csv).
- D1: scripts/build_targets.py -> data/targets/*.csv (per fold, fitted without the held-out fold; 'fall' = all non-gold studies).
- D2: data/aux_facts.csv (305 columns), data/aux_facts_cols.csv (feature names, counts).
- D3: data/site_train.csv (report site per study); data/dicom_site.csv (DICOM-only site, CV prediction).
- D4: data/key_image_refs.csv.
- D6: data/postproc/site_clf.joblib (TF-IDF of DICOM header tokens + logistic; doc builder = a7_dicom_site.study_doc), site_offsets_961.json
  (Effusion / Baker's, shrink 20, applied when p >= 0.9).
- main/ = copy of the deployed sub_b0 trainer (experiments/v0/oct7_baseline) with new options, all off by default:
  - tgt_csv / tgt_findings: D1 targets, '{fold}' per fold.
  - aux_csv / aux_weight / aux_cols: D2 heads after the recipe heads; masked BCE.
  - site_csv / site_mode all|vague / site_l2 / site_min: D3 offsets in the training loss only; learned offsets saved to
    site_offsets.csv.
  - init / ft_mask x / ft_keep: the PL clean-label stage.
  - predict.py reloads models with fact heads.
- basin/ = copy of the 3rd-baseline Basin trainer (runs the deployed NFNet P_final and R50 F20 configs, incl. --mono) with
  --tgt_csv / --tgt_findings; cache -> v0/oct8_baseline_v2/cache; weights/radimagenet.
- oa/ bone/ syn/ = oct14_all_data's self-contained copies of the deployed expert trainers (+ data/loc_pred*, loc/pred, runs/_bags,
  data/img_syn_prob.csv, data/synovitis_labels_codex.csv); oa / syn labels.py got tgt_csv (bone already had it). Deployed expert
  configs and their own OOF / gold outputs: refs/exp_*.
- scripts/launch.py: an arm from a deployed source config + overrides; kinds main (refs/sub_b0/fold0), basin (refs/basin_nfnet_P_final,
  refs/basin_r50_F20), oa (refs/exp_oa_crop_v2cor[_nfnet]), bone (refs/exp_b_v3 / b_lr3 / b_nf), syn (refs/exp_s_fs / s_fc125).
- run_queue.sh (copied runner: gpu_go.txt, slots_max.txt, logs/events.txt, done.txt), queue.txt (pre-registered).
- weights/sub_b0/fold{0-4}.pt (deployed main checkpoints, for stage arms).
- scripts/eval_arm.py: arm vs the same-recipe references; package swap of any member (--member main / nfnet / r50 / oa_b0 / oa_nf /
  b_v3 / b_lr3 / b_nf / s_fs / s_fc125) vs the deployed row (the member's own training-time eval, same k) and, for main, the seed row
  (sub_b0_s1); Codex Effusion / Synovitis / Baker's, report metrics (info), gold-12 veto. Identity arms read 0 for main, NFNet,
  OA, bone and synovitis experts.
- Smoke tests passed (GPU <= ~1.5 GB, Basin 6 GB):
  - v4 targets + 305 heads + site 'vague' (train, eval, save, predict.py reload);
  - stage from sub_b0 with ft_mask x;
  - site 'all' with partial tgt_findings;
  - Basin NFNet with v4 targets through launch.py;
  - OA expert (v4), bone expert (v4 replacing v3), synovitis crop expert (v4 text part) through launch.py.

## Launch plan (when the GPU is granted)

1. Start the runner detached and open the gate (one arm at a time, ~10 GB for main arms):
   `nohup setsid experiments/v2/oct14_text_labels/run_queue.sh > /dev/null 2>&1 &`, then `echo 1 > gpu_go.txt` (from the repo root).
2. Tier A (cheap, ~60 min): st_x (reference: PL's xF on the deployed main), st_v4, st_v4b, st_v5 (lead), st_sil on folds 0-2.
   - Read: `scripts/eval_arm.py st_x st_v4 st_sil --epoch E`, using the median-of-best epoch of 0-3 (protocol), then the last.
   - Promote to Tier C only if the package swap beats the seed row on Codex and gold-12 is not vetoed, on >= 2 / 3 folds.
3. Tier B (~4 h): tl_v4, tl_sitev, tl_aux05, tl_nmc, tl_sil on folds 0-2.
   - Same read; single-model rows vs the 3 references.
   - Report metrics are v1-derived and penalise every label change by construction, so they are info only.
4. Tier C: stacks of the winners, a seed replicate of the lead (seed 1), the lead recipe on NFNet / R50 (cross-backbone, user rule),
   then expert retrains on v4 (queue.txt Tier C: oa_v4, b_v4, s_fs_v4; trainers, bags and launch kinds are in place).
5. LB: one replace-columns probe per promoted change, only with the user's go. No metadata-only probe (D6 not robust).

GPU cost: Tier A ~40 min; Tier B ~4 h (13 min per fold run); Tier C NFNet ~25 min, R50 ~30 min per fold.

## Caveats

- Image-side references:
  - Codex (Effusion / Synovitis; Baker's only on report-vague studies) is one reader with its own site bias.
  - The generalists' OOF ranks inside a status are image-driven, but they can echo co-occurrence the models learned. That is why
    their levels are never used, only ranks inside a status.
- Label-level gains do not guarantee model gains (kNN / severity labels looked neutral as targets and moved models differently).
  The GPU reads decide.
- Gold-58 numbers carry ~3e-3 sd per run and best-of-N selection bias; v3's prompt saw gold summaries (v4 does not use v3).
- Not used: v3 (needs the user's go), other participants' weights, gold for fitting.
- Test-time site offsets (D6 P1) are risky for the ~5% of test studies from new sites (memory test-site-overlap): the classifier gives
  no offset below p 0.9, but a new site sharing a scanner / protocol with a train site can be mapped confidently to it (offsets are
  small: sd 0.24 logit). Review 29 prefers site only as a training-time term or label calibration.
- Review 29 (10-15): weight the reads on the columns the arms can change in 0.961 (OA x3 / menisci, Effusion, Baker's, F / C);
  Synovitis label gains do not reach the package (its column = image reads).
