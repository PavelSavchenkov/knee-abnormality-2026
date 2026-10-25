# oct12_pseudo: pseudo-labels (PL) for the main pipeline (all findings)

**FINAL SEAL 2026-10-15** (user: 'seal current learnings, next plans and all documentation, then quit'). History: sealed
10-14 05:33 -> reopened 07:37 for the user's feature-KD addendum -> stopped by the user 10:05 -> GPU owned by the All Data session
since 10-14 20:34. Nothing runs: runner stopped, gpu_go.txt 0, every queue line commented out. Earlier notes kept below.

### Inner-world LB results (outer fold 3, inner folds 0-2, 3x3 recipe, no experts; submissions/ folders hold the packages)

| package | what changes | public LB |
|---|---|---|
| B0 (oct13_pl_b0) | 9 honest inner models (60% data) | 0.943 |
| B0_xF (oct13_pl_b0xf) | + clean-label stage on main + R50 | 0.945 |
| K6r (oct13_pl_k6) | + PL students (main, NFNet) + teacher-confidence stage on main | 0.946 (best unrounded) |
| oct14_pl_hyb (review session) | K6r's Fracture / Contusion / Synovitis / Effusion + K12fkd's other 8 | 0.946 (between) |
| K12fkd (oct14_pl_k12fkd) | K6r with feature-KD main students | 0.946 (lowest unrounded) |

### Learnings (evidence in ledger.md; strength = folds x seeds as stated there)

1. Clean-label fine-tune stage (4 epochs on explicit v1 cells; synovitis / fracture / contusion keep all cells) = the most reliable
   gain (LB +0.002). The mask is what works (the all-cells control is negative); NFNet must not get the stage.
2. PL students (honest 3-backbone teacher T3pH, ct blend; NFNet without teacher synovitis) beat B0 locally on 2 seed sets; on the LB
   they add ~+0.001 over B0_xF (K6r), not separable. R50 students, round-2 teachers and noisy students do not help.
3. The teacher-confidence (c) stage helps only on the PL main students; its confidence gate matters (all uncertain cells at the
   teacher value hurt).
4. Local metrics mislead in two ways: report agreement on uncertain cells is circular (teacher alpha down -> trusted up, gold down),
   and gold_nofc_xe gains of ~+3 do not predict the LB (K6r +3.2 -> +0.001; K12fkd +3.3 -> slightly negative). Gate LB probes on all 12
   findings, discount best-of-N selection (~2-3e-3 gold at N ~12), and size per-finding noise from >= 3 same-recipe runs (Baker's
   trusted seed sd ~6.5e-3 on 41 fold-3 positives).
5. Closed negative (screens as stated in ledger.md): uncertain-cell down-weighting (loss weight, teacher alpha, ramp), deeper heads
   (MLP class head, 2-layer attention), attention / class-contribution KD, window max-negative, the window-distillation family, stage
   variants (xu, cu, c + window KD, silent weight, agreement filters), R50 stage at 3x lr, longer stages. Feature KD: the local gain
   did not reach the LB (K6r > hyb > K12fkd); MSE form, FitNets projections, intermediate maps (no stack with fkd), NFNet feature KD
   (tie) and cross-backbone study-level KD (learned projection and contrastive) are all negative or ties: learned-projection KD costs
   the student.
6. Data beats steps: 80% vs 60% data at equal steps +4.4 trusted; more steps ~0.
7. Window fusion (0.961 package) stacks with the PL gain on reports / gold but overlaps it on effusion (fusion + img_eff fix the same
   image-read errors).
8. Feature spaces: the inner fold models' spaces are unrelated rotations (per-fold projections needed); Basin pooled embeddings are
   ~one shared direction until centred.

### Next plans (all need the user's go; this session runs only inner-world experiments)

- Transfer to the 0.961 package (other sessions): the clean-label stage on the deployed 3x3 (main + R50), then per expert (OA first);
  expect only the non-effusion part; test any _xe-promoted change by transfer and gate on all 12 findings.
- PL students in the deployed world only if the user wants the ~+0.001 that the LB cannot separate.
- The data lever (more studies) belongs to the All Data session (full-data retrains, extra seeds); OAI as an unlabeled PL pool is a
  user decision. Outer fold 4 confirmation was never run (user request only).
- Not run because of the 10:05 stop (low value after the LB result): F1 MSE dose 0.01, F2 block-4 map, F3 NFNet feature KD dose 0.3.
- Do not repeat the closed lines in Learnings 5.
- The PL review session's own learnings, method lessons and still-rated open ideas (all-data training, clean-label stage on
  deployed members, calibrated text labels v4, S6-style generalist, lab_mix4 blend): review_learnings.md (saved verbatim 10-15).

### Where things are

review_learnings.md (the PL review session's learnings, verbatim), README sections below (state at each stage), ledger.md (every result in time order), user_directives.md (every user instruction),
probes.md, queue.txt (all arms incl. dropped ones), scripts/ (reads: eval_inner, stage_variants, student_combos, ft_curve(_inner),
moved, pkg_moved, fusion_k6r; builders: build_honest(_xf), xkd_main_dump, win_dump_*), main/ and basin/ trainers (options documented
in their docstrings), out/ (reads), submissions/oct13_pl_b0, oct13_pl_b0xf, oct13_pl_k6, oct14_pl_k12fkd. Review session 29's reads:
work/pl_research/ (c28 / c54 / c56 ...).

Earlier status notes (history):
**SEALED 2026-10-14 05:33 BST** (user plan relayed by review session 29: four directions exhausted, probes built, GPU handed to the
Backbone Sweep session). Final state: inner-world LB ladder B0 0.943 -> B0_xF 0.945 -> K6r 0.946; new probe submissions/oct14_pl_k12fkd
(K6r + feature-KD main students) built + gated, not uploaded (user's per-submission go; the review session submits). Runner stopped,
queue empty. Section 'User directive 10-14 01:35' (end of file) = the last block of work; ledger.md = every result in time order.

**REOPENED 2026-10-14 07:37 BST as a user-directed addendum** (user ~07:25 / 07:33 via review session 29; GPU back from Backbone
Sweep until these are exhausted): three feature-distillation DIRECTIONS (each with follow-ups while positive): F1 MSE loss for
feature KD, F2 intermediate encoder maps (effb0 blocks), F3 feature KD on the NFNet students. Section 'Addendum 10-14 07:25' at the end.

Started 2026-10-12 19:00 BST (user goal, 24 h; user_directives.md). Results in time order: ledger.md; ideas and ranking: ideas.md;
Kaggle probe design: probes.md; public evidence on PL: public_pl.md. This session is the GPU master (runner scripts/run_queue.sh).
Plan source: work/pl_research/README.md (sections 0, 2-4, 9) + the OA Pseudo session (experiments/v2/oct12_oa_pseudo).

## Question (updated 20:46 by the user, relayed by the review session)

Which pseudo-labelling (PL) techniques improve the 3x3 package (3 pipelines main effb0 / NFNet-L0 / R50 x folds 0-2, LB 0.948) when
NO experts are used anywhere (teacher from the 3 pipelines only)? The goal is to find what works in PL, not a one-shot best setting;
experts and PL get combined later, once each is tuned. Before 20:46 the session ran the main effb0 student with an expert teacher
(A1, T_x): kept as evidence (ledger 19:23-20:46).

## Setting and validation (20:46)

- Teacher T_3p = z-mean of the 3 pipelines' predictions (main seeds / NFNet e14 / R50 e20), quantile-mapped onto sub_b0's OOF marginal;
  cross teachers without one pipeline (T3p_nom / _nonf / _nor50).
- Honest nested validation (user): outer fold 3 (fold 4 later, to confirm winners). For each training fold g != 3, one model per
  pipeline trained on the 3 folds outside {g, 3} predicts fold g (runs in_m_h3 / in_nf_h3 / in_r50_h3); teachers T3pH*_h3 from those
  (scripts/build_honest.py). Students train on the 4 folds outside 3 with these honest targets and are scored on fold 3.
- Baseline = the 3x3-shaped package on fold 3 (plain logit mean of the 3 pipelines' fold-3 models); views: student alone vs its
  family's references (main: 3 same-recipe refs; NFNet / R50: the package member, seed noise borrowed: <= 1.5e-3), package replace /
  add vs count-matched seed swaps (scripts/eval_arm.py, eval_pkg.py).
- Reading rule (user): one fold and one seed per arm (a seed swap moves trusted ~1.4e-3): < ~2e-3 = unresolved; confirm candidates on
  fold 4 or with a 2nd seed before any LB probe. Metrics: trusted / lookat_m / text (explicit report cells), gold-58 (small), gold_unc /
  cx_eff_unc (image truth on uncertain cells); sg / thr / un are status-ordering proxies only (19:25). Every arm's targets dumped.
- Stage C (leak size): T1L (sub_b0 OOF, leaky) vs T1H (inner main, honest) twins on fold 3; strength gap printed by build_nested.py.
- LB (user's go): P probe = 3x3 with students on folds 0-2 vs 0.948; M probe = main student f0-4 vs M0 0.938 (probes.md).

## Layout

- main/: main effb0 student trainer (copy of OA Pseudo's main copy of experiments/v0/oct7_baseline; options in the train.py docstring:
  pl / pl_alpha blend, ext_labels teacher head, hold, filt (CR), conf_* (confident-only), conf_drop, init / ft_mask (fine-tunes)) and
  predict.py (--fit = train-fit probe).
- basin/: NFNet-L0 / R50 student trainer (copy of experiments/v0/oct10_baseline_3rd/src + --pl_head / --pl_w / --hold); cache / weights
  are symlinks to oct8_baseline_v2 (data).
- scripts/: pl_lib.py (loaders, teachers, package recipe, panel); teacher.py (leaky 5-fold teachers T_m / T_p / T_x / T_3p + cross);
  build_honest.py (honest nested teachers from inner runs; pipeline subsets); calibrate.py (per-finding isotonic for confident-only);
  alpha.py / v3_targets.py (blend tables, v3 targets); eval_arm.py (one arm vs refs, package replace, add, curve), eval_pkg.py (3-member
  package, replace / add, diversity), eval_pair.py (2-pipeline ensemble read); moved.py (AUC contributions + quotes); target_check.py,
  filter_preview.py (CPU checks); export_probe.py (probe exports); run_queue.sh (runner), wait_build*.sh (teacher watchers).
- refs/: OOF + gold of references and teacher members (copies; model.pt only for sub_b0 f3 / f4: fine-tune init and fit probe).
- data/: teacher_<T>.csv (+ _gold), alpha_<name>.csv. out/: evaluations. runs/: runs. logs/: runner events + arm logs. export/: probe
  exports. review.md: the review session's findings (2026-8e).

## State at the end of the 24 h goal (10-13 ~18:15; details and every number in ledger.md, probes.md, user_directives.md)

- LB (inner world, outer fold 3, inner folds 0-2): B0 = 9 honest inner models 0.943 (submissions/oct13_pl_b0); B0_xF = B0 with the
  clean-label stage on main + R50 0.945 (oct13_pl_b0xf; +0.002, the pre-registered bar). B0 (60% data) vs the deployed 3x3 (80%,
  same recipes) 0.943 vs 0.948: the data effect.
- Clean-label stage (xF): 4-epoch fine-tune from each checkpoint on explicit v1 cells (meets / denied / structure_normal); synovitis,
  fracture, contusion keep all cells; main synovitis's non-explicit cells from the honest teacher T3pH. Members chosen on each
  fine-tune's own inner validation (scripts/ft_curve_inner.py): main + R50 yes, NFNet no (its stage hurts NFNet). The all-cells
  control is negative: the mask is what works. Explicit-only stages widen the logit spread ~1.3-1.45x (calibration table).
- PL (user's definition: teacher = 3-backbone ensemble, student = one backbone, test = student package vs B0): main + NFNet students
  beat B0 on 3 folds (NFNet students carry it); R50 ct students lose synovitis -> 'ctns' (no teacher synovitis for R50 / NFNet, the
  training-fold rule); R50 ctns students running. PL stacks with the stage: K1 / K2 / K6 (review session c54, pre-registered rule).
- Teacher-confidence filter (c mask): uncalibrated T3pH only (the calibrated teacher is fitted to reports = 'silence is negative');
  effusion up, report agreement down; on R50 negative; best use so far: on the PL main students (K6).
- Pending (queue.txt): R50 ctns students, agreement / both-positive filters, longer / other stage variants, window distillation arms
  (cross-backbone teacher: Basin per-image export + lookup_mode v6 / pm, fk_topn, top3_kd; scripts in this README's ledger).
- Rules: stay in the inner world until the user says otherwise; read gold / Codex / calibration first plus manual cases (text can
  mislead); inner-world methods transfer to experts with a per-expert estimate (ledger 16:13); heavy jobs only in systemd-run scopes.
- Read tools: scripts/stage_variants.py (variant in B0_xF's slot, transfer-weighted columns, gold synovitis), ft_curve.py /
  ft_curve_inner.py (curves, calibration, moved cells), student_combos.py (T / S / TS packages), final_l6.py, eval_inner.py.

## Final additions (10-13 evening; details in ledger.md)

- PL test (user's definition), inner folds 0-1: all 3 backbones -> students = no credible gain vs B0 (R50 students are worse
  single models); main + NFNet students (R50 = B0's) = credible gain (trusted +1.3 [0.5, 2.1], lookat +2.4, gold_nofc +1.9).
- Seed replication (review 169, 3 inner folds, vs B0, x1e-3): main + NFNet students, R50 = B0's. Original seeds: trusted +0.7
  [0.0, 1.3], lookat +0.9, gold_nofc +2.9, gold_unc +2.1, Codex effusion -0.5. Re-seeded students (is_T3pH_ct_s1 + is_nf_T3pH_ctns_s2):
  trusted +0.8 [0.0, 1.6], lookat +1.2 [0.1, 2.3], gold_nofc +2.4, gold_unc +4.7, Codex effusion -2.5. Seed row (B0 main seed 1):
  trusted -0.1, gold_nofc +1.1. So the pure PL gain over B0 replicates across two seed sets; Codex effusion is slightly down in both.
- Best inner-world package (promoted under the pre-registered rule, median-of-best epochs): K6r = main PL students + teacher-
  confidence stage (L6Fc_ism@0) + NFNet PL students (is_nf_T3pH_ctns@16) + R50 xF (L6F_r50@4). vs B0_xF: trusted +0.04, gold_nofc
  +3.2 (3 / 3 folds), gold align +0.28 [0.17, 0.39]. LB forecast vs 0.945: ~+0.001-0.0015; package folder submissions/oct13_pl_k6
  (user go 10-13 23:50; submitted 00:04 BST 10-14, ref 57158313): LB 0.946 = +0.001 vs B0_xF 0.945 (consistent, not separable
  from noise), +0.003 vs B0 0.943. Inner-world ladder: B0 0.943 -> B0_xF 0.945 -> K6r 0.946.
- K6r seed robustness (reviews 166 / 171; every PL member retrained at another seed, x1e-3): vs B0 the six variants give trusted
  ~+0.7..+1.6, gold_nofc ~+4.3..+5.2 (seed row -0.1 / +1.1); vs B0_xF gold_nofc +3.0..+3.9 and gold align +0.27 are stable, trusted
  (-0.3..+0.6) and Codex effusion (-4.2..+0.8) move with the seed. LB forecast unchanged: ~+0.001 vs 0.945, ~+0.003 vs B0.
- Fusion x K6r (10-14 01:30, out/fusion_k6r.txt; d0's deployed window fusion of Effusion + Baker's, 300 fold-3 studies + gold-58):
  K6r - B0 without -> with fusion: trusted +1.4 -> +1.6, gold_nofc +5.3 -> +4.4, Codex effusion +9.4 -> +3.4. The package gain stacks;
  the effusion image-read gain overlaps with fusion (and with the 0.961 package's img_eff head, d0). A transfer into the 0.961 package
  should expect the reports / gold_nofc part on the non-effusion columns only.
- Closed (negative or no gain, with the evidence in the ledger): agreement / both-positive filters, 'below' cells, silent weight,
  R50 teacher-confidence, every NFNet stage (incl. on students), R50 students, round-2 teacher (both backbones), the window family
  (W1 / D1 / D1p / T3a / D1pm: Baker's loses even with its own window targets masked, so the loss comes through shared features), longer stages (students peak at e0; median-of-best).
- Data x steps (user): the gain is the data, not the steps (80% vs 60% at equal steps +4.4 trusted; more steps ~0 trusted).

## User directive 10-14 01:35: four directions (results 05:35; details in ledger.md, reads by review session 29)

Inner world only (user, standing); new training-side approaches for the 0.961 package, not ports of its methods (user). Screens:
fold-3 single models vs the ct seed pair (c28, per-finding guard from 3 same-recipe refs; promotion also needs gold_nofc_xe >= -1 and
align >= seed row - 0.05 since 03:15) and inner K-slot reads vs K6r (c54); every arm also read by hand (scripts/moved.py,
scripts/pkg_moved.py). x1e-3; '_xe' = without Effusion (overlaps 0.961's fusion + img_eff head).

| direction | arm | result |
|---|---|---|
| 1 from scratch, down-weight uncertain cells / their teacher | unc_weight 0.3 | FAIL (trusted -1.65, gold_nofc -2.3) |
| | teacher alpha 0.35 on uncertain cells (ctu35) | fold 3 trusted +1.35 but gold_nofc_xe -4.05 (follows reports' 'below' wording); in K6r's slot a tie -> STOP |
| | unc_ramp 0.34 (integrated stage) | FAIL (trusted +0.15, gold_nofc_xe -2.65) |
| 2 deeper heads | per-window MLP class head | FAIL (gold_nofc -4.8) |
| | 2-layer attention scorer | FAIL (trusted -1.95, Baker's -17.9) |
| | NFNet + transformer layer | not re-run (2nd baseline screened it: not adopted) |
| 3 KD / max-negative | attention KD (3-backbone pm) | FAIL (gold_nofc_xe -3.65) |
| | class-contribution KD | FAIL (gold_nofc_xe -4.65) |
| | window max-negative 0.5 / 0.05 | collapse (underfit) / FAIL (gold_nofc_xe -14.45): fights attention-MIL |
| | **feature KD 0.1** (main teacher window features) | **PASS** fold 3 (trusted +0.85, gold_nofc_xe +3.65); K12fkd 3 folds vs K6r trusted +0.5, gold_nofc_xe +4.0, align +0.099 (K6r seed band <= +0.7 / +0.071) -> probe |
| | feature KD 0.3 | trades gold for reports (gold 0.9193 vs 0.9226 at 0.1): 0.1 stays |
| 4 stage variants on the PL main students | explicit + unsettled (xu) | STOP (align -0.09) |
| | c stage + window KD | STOP (tie) |
| | c + all uncertain cells at the teacher value (cu) | STOP (gold_nofc -2.2): the c mask's confidence gate matters |

- Probe built + gated (not uploaded; needs the user's per-submission go): submissions/oct14_pl_k12fkd = K6r with main -> PL students
  + feature KD + c stage (e0). Gate: 275 studies, 0 failed, corr 0.99997, gold-58 0.9325 = local. Forecast vs K6r 0.946: ~+0.0007..+0.001,
  P(>= 0.947) ~0.4-0.45.
- Method findings: Baker's trusted (41 fold-3 positives) has seed sd ~6.5e-3 (3 same-recipe refs), 4x the 2-run seed row: most
  'Baker's losses' vs the ct pair were noise. Report-only gains on uncertain cells are circular (ctu35): the amended promotion rule needs gold.
- Manual direction of feature KD (K12fkd vs K6r): sharper on explicit structural findings (MCL complete rupture, LM tears up; silent
  fracture negatives down), softer on mild 'below' effusions.
- Open follow-up if the probe gains: feature KD for the NFNet students (needs Basin feature dumps) and for the experts that share the effb0 encoder.

## Addendum 10-14 07:25: feature-distillation directions (user; results appended as they land, details in ledger.md)

Reference to beat: K12fkd (main = PL students + feature KD 0.1 + c stage; probe oct14_pl_k12fkd). Screens as registered by review
session 29: F1 / F2 fold-3 single models vs the ct pair (amended rule) AND >= mbl_T3pH_ct_fkd on gold_nofc_xe and align, then inner
K-slot vs K12fkd; F3 straight to inner f0 (K13nf = K12fkd with NFNet -> is_nf_T3pH_ctns_fkd), same f0 / 3-fold rules vs K12fkd.
- F1 MSE: feat_kd_loss=mse (normalised MSE per window; weight 0.035 matches fkd 0.1 at step 0).
- F2 intermediate maps: fmap_kd on effb0 blocks (2 = stride 8, 4 = stride 16); online frozen honest in_m_h3 fold teachers on the
  student's augmented windows (fkd used clean-pixel dumps); weight 0.2 matches fkd 0.1 at step 0.
- F3 NFNet: basin --feat_kd with online frozen in_nf_h3 (e14) fold teachers on the same augmented batch (post-norm image tokens).

### Addendum closed by the user (10-14 10:05 'Stop now', after K12fkd < K6r on the unrounded hidden LB score)

K12fkd (feature KD 0.1) scored 0.946 like K6r but BELOW it unrounded: feature KD does not transfer to the test, and local package
deltas of ~+3 gold_nofc are not LB-predictive (K6r - B0_xF +3.2 -> +0.001; K12fkd - K6r +3.3 -> slightly negative). Final status
(fold-3 single models vs the ct seed pair unless noted; 1 seed each; reads by review session 29, manual reads in ledger.md):

| direction | arms | status |
|---|---|---|
| F1 MSE feature KD | fkdm (normalised MSE 0.035) | FAIL (gold 0.9154 vs fkd 0.9226; matching magnitude hurts); dose follow-up not run |
| F2 intermediate maps | fm24 (blocks 2+4, cos 0.2) / fm24f (+ feat_kd) | fm24 near-miss (trusted +0.4, gold_nofc_xe +0.3, below fkd on gold); fm24f FAIL (does not stack); fm4 not run |
| F3 NFNet feature KD (online in_nf_h3 teachers) | is_nf_T3pH_ctns_fkd f0 -> K13nf | ties K12fkd at f0 (trusted -0.2, gold_nofc_xe +0.5, align +0.01); f1-2 dropped (ROI), dose 0.3 not run |
| D4 FitNets projection | fkdp (per-fold 1280 x 1280, identity init) | FAIL (below fkd on every column: fkd acts as a regulariser, not feature copying) |
| F4 cross-backbone study-level KD | xkd_2 (NFNet + R50 v, centred, per-fold P), xkd_m (main control), crd (contrastive InfoNCE) | xkd_2 FAIL (trusted -2.3, gold_nofc_xe -5.5), xkd_m FAIL (-1.4 / -3.9: the projected-KD mechanism costs the student); crd FAIL (review 214: vs ct pair trusted -2.5, lookat -2.8, gold_nofc_xe -1.1, align 0.049, Codex effusion -18.5; package trusted -0.4, gold_nofc_xe -0.6; train-bank top-1 only 0.35) -> F4 closed negative across both forms |

Code added (main/train.py: feat_kd_loss, feat_kd_proj, fmap_*, xkd*, crd*; main/model.py keep_z; basin/train.py feat_kd / fkd_*;
basin/model.py last_h; scripts/xkd_main_dump.py, win_dump_*.py, fusion_k6r.py, moved.py --refs, pkg_moved.py). Runner stopped after crd;
GPU back to Backbone Sweep.

LB reading of feature KD (user, full hidden score; review 215): K6r > oct14_pl_hyb (K6r's Fracture / Contusion / Synovitis /
Effusion + K12fkd's other 8) > K12fkd, all displaying 0.946 (gaps < 0.001). hyb > K12fkd isolates the 4 swapped findings (K12fkd lost
there); K6r > hyb isolates the other 8 (K12fkd lost there too, where gold-58 said +4 on average). Feature KD is slightly negative on
the test in both groups; its gold gain was selection / sample noise. D4 (feature KD) is closed for the LB as well.
