# oct16_lateral_oa: Lateral OA for the 0.961 fold package (v3 lineage)

Started 2026-10-16 00:44 (user /goal, 24 h; user_directives.md). Target: the Lateral OA (LOA) column of the 0.961 FOLD package
(submissions/oct13_f5_win, folds 0-4 averaged), with ideas that transfer into that package. GPU only after Master's 'GPU GO'.
Files: ledger.md (time-ordered results / decisions / reviews), ideas.md (ranked), VALIDATION.md (LOA read practices), REVIEW.md
(10-min checklist), PLAN.md (pre-registered GPU plan), scripts/ (reads), main/ (OA expert trainer copy), refs/ (copies), out/.

## FINAL STATE (SEALED 10-16 ~23:00; the user concluded the session at ~22:15 and closed it ~23:00; learnings -> Experts V3 via HANDOFF_experts.md)

Start here: this section, then HANDOFF_experts.md (ranked learnings, noise table, tools, open ideas), then ledger.md for any single read.

- The one transferable LOA gain: inference-only k_eval 64 for the LOA column (output-level splice), LB-confirmed above 0.961
  (57273283); part of the fold-package recipe (user). The same change on the s5 all-data package: built and locally gated
  (submissions/oct16_probe_loa_ke64_s5); Master pushes / submits.
- Closed with verdicts (ledger.md has every read): post-hoc LOA knobs (trade-offs / no gain); P1 single members (bml, k16, gradeH, silL:
  no report gain; k16 mixed at member level); per-class experts (LOA-only no gain, 2 seeds; PF-only negative, 1 seed); silL on f0-2
  (no gain detected, 1 seed); TF reader not run (ROI).
- Method: member reads with each model's own pooling (s22 --pool own); count-matched slot reads (s08 / s19); single-class probes splice
  at the output level; kernel working copies copy the .py files.
- Open ideas (ranked in HANDOFF_experts.md): G20 per-bag window budget inside the shared expert (code ready), C15 refit the fusion gate
  for 64-crop bags, G19 PF own crops inside the shared expert, k16 seeds, G16 30 epochs, NFNet v2cor f3-4.
- GPU released 22:37 (verified); nothing of this session runs; heartbeat stopped at the seal.

## Runs index (all GPU work of this session; effb0 v2cor OA expert recipe, 20 epochs, epoch 19 read; runs/ + refs/win/ dumps)

| run | folds | seed | what | verdict (read) |
|---|---|---|---|---|
| v2cor_s1 | 0-4 | 1 | seed control of the deployed recipe | control |
| v2cor_s2 | 3, 4 | 2 | 2nd seed control | control; noise rows |
| v2cor_bml | 3, 4 | 11 | + oedema (bml) aux outputs | no gain detected (P1 slot) |
| v2cor_k16 | 3, 4 | 19 | 16 train windows / bag, k_eval 64 | mixed: member +4..+6 (own pool), -5.2 in the deployed fusion |
| v2cor_gradeH (+ T head dumps gradeHT) | 3, 4 | 13 | extra definition head on host-rule graded LOA (pl_head) | no harm, LB-sized (P1) |
| v2cor_silL | 0-4 | 18 | LOA targets v4b (silent cells calibrated) | P1 no harm; f0-2 vs s1: no gain detected (S -1.1); LB probe by Master (silL member probe) |
| v2cor_loa1, v2cor_loa2 | 3, 4 | 31, 32 | per-class LOA-only expert (keep_bags lateral, k 24 / k_eval 96) | no gain detected (2 seeds, own pool) |
| v2cor_pfx | 3, 4 | 33 | per-class PF-only expert (keep_bags pf) | negative (1 seed, PF trusted) |
| dumps: oa_crop_v2cor(_nfnet)_ke64, *_tr0.0, v2cor_s1_ke64, v2cor_s1 / s2_ke96 | | | inference-only eval-window / trim variants | k_eval 64 -> LB win (57273283); trim 0 worse |

Kaggle: 57273283 oct16_probe_loa_splice (0.961 + LOA at k_eval 64): public 0.961, above 0.961 by hidden precision. Submitted by Master
from this session's builds: submissions/oct16_probe_loa_ke64_s5 (the same change on the all-data s5 base) and
submissions/oct16_probe_loa_silL (0.961 + silL as an LOA-only member); their scores are in Master's log / those READMEs once scored.

## Headline results (10-16 evening)

| question | test | result |
|---|---|---|
| Does Lateral OA at k_eval 64 (inference only; the 11 other columns = 0.961 exactly) beat 0.961 on the LB? | splice probe 57273283 (submissions/oct16_probe_loa_splice; user's go) | public 0.961, ranked ABOVE oct13_f5_win by hidden precision (API sort by public score = the Kaggle UI order, user screenshot): 0 < dAUC_LOA < +0.012. Gold-58 called the direction (+11.6 local / +15.5 offline x1e-3); report reads did not (S -0.6). Transferable to any package with these experts. |
| Per-class experts vs the shared expert (user 10-16 ~19:0x) | keep_bags (own bag, own loss only), effb0 v2cor, f3 + f4, member level with each model's own pooling, vs shared seeds s1 / s2 | LOA-only: NO GAIN DETECTED (2 seeds; S -3.6 vs controls at 32 / -3.9 with matched windows; bar +13). PF-only: NEGATIVE (1 seed; PF trusted -4.8 / -15.5). The shared expert wins or ties. |
| Method lesson | member-level reads | Reading single models through the deployed fusion spec (top-10 gate fitted on <= 32-crop bags) penalised 96-crop experts (LOA-only fused S -31 vs own pooling -9): read members with their own pooling (validation.md: refit package constants for new models). |
| silL-only (option (b) cut-down; ran because the splice probe won) | silL vs s1 as an extra fusion group, f0-2, 1 seed each (s19, pre-registered bar) | NO GAIN DETECTED: S +2.7 / -1.0 / -5.0 (mean -1.1), gold +5.8; no probe (Master submitted the MCL fallback instead). |

## State (10-16 ~01:35; GPU block result below)

- The 0.961 OA block is rebuilt exactly from refs/ (scripts/lib.py: OOF f0-2 and gold-58 match the package to 0.0).
  LOA = 0.4 x 3x3 generalists + 0.6 x crop-fused OA experts (effb0 v2cor f0-4 + NFNet v2cor f0-2) + de-coupling (k 0.4 on MOA / PF).
- Package LOA (OOF f0-2): text 0.955, tfull 0.978, tsev 0.969, thr 0.814, un 0.902, gold 0.867, gold_rb 0.932, gold_rb_loa 0.962.
- Gold-58 cannot rank LOA arms: two studies no image reader sees (855003 contested, 559395 report-silent trauma knee) carry most of
  every LOA gold delta; without them the package is already 0.962.
- No trusted LOA read exists (trusted has negative rules only, = text's negatives). tfull / tsev / tord = trusted-style and
  host-definition-aligned reads ('moderate or large area (~1 cm) of > 50% loss; borderline graded negative').
- Text-visible errors are near the ceiling: status / attribute order already follows the expected test positive rates; remaining
  text errors are small focal posterior lateral-plateau defects (partly test-negative by the definition) and LM-tear coupling FPs.
- Post-hoc LOA knobs (w, de-coupling k, LM regressor, fusion spec / gate, MOA head as 2nd lateral reader): no held-out gain that
  survives gold; maxmix fusion = report-vs-gold trade-off. Closed (moderate: post-hoc, 3 folds + gold).
- Area (extent) reader on the deployed crop logits (C13, s20: 3 adjacent coronal windows): worse on 3 / 3 folds (S -3.45, tord -31,
  gold 0.0). No unused area signal in the current crops; an extent-TRAINED expert (G10) is untested but gets no support.
- Old expert variants re-read on LOA only (old sessions chose on the 3-OA mean): every variant is within ~1-2 slot-noise of 'one more
  copy of the deployed member'; the most consistent lead is the oedema auxiliary head (bml; b0 bags, 2 folds), never run on v2cor.

- Coupling (CPU): LM-tear knees carry 23% of LOA's text FP mass and the meniscus aux loss carries LM coupling into the LOA head
  (0.81 vs 0.75 without it), BUT gold has LM tears in 9 / 11 LOA positives vs 14 / 47 negatives: largely genuine co-occurrence.
  PF -> LOA interference (Labels Work's PF pilot) has two channels here: cross-compartment report wording (PF unsettled -> LOA
  unsettled 40% vs 14%) and the package de-coupling (LOA - 0.4 x 0.29 PF). Constraint: no MOA / PF target changes in LOA arms.
- De-coupling k 0.4 kept: gold rb_loa and meets-vs-below favour it; the status-level soft-gold read (sg) prefers less but is blind to
  coupling.
- GPU plan (PLAN.md, Master-confirmed, amended): P0 v2cor seed controls; P1 on f3 / f4: bml (oedema aux), trim0 (posterior slices),
  gradeH (definition head, LOA-only), gradeL, LOA loss x2, noaux (men_weight 0), PL last. CPU: inference-only k_eval 64 / trim 0 dumps.
- LB: ONE full-package probe (user 10-16): submissions/oct16_probe_loa_splice = 0.961 for 11 columns + Lateral OA at k_eval 64;
  the keep-LOA pair (oct16_probe_loa_961 / _ke64) is retired. Needs the user's go to push / submit.
- Size expectation: single-arm gains of ~+2e-3 text would be ~+0.002-0.004 LOA AUC on the LB (OA LB / text ratio ~1.4-1.8x on 10-11),
  below a keep-probe's +-0.006: only a bundle of several gains can show on the LB.

## How LOA gains could reach the test (working hypothesis)

Better within-status ordering by better / more independent image readers (more expert members, oedema supervision, definition-aligned
heads), judged on report reads with count-matched seed controls, and finally on a LOA keep-probe pair on the LB (user's approval).

## Next (10-16 ~02:00)

1. GPU (Master's GO, ~06:30+): P0 + top 4 P1 arms (queue.txt), then STOP -> scripts/s13_p1_report.py -> read to Master -> release held
   lines / P2 (#P2 lines pre-written) / PL.
2. CPU now: inference-only k_eval 64 / trim 0 dumps of the deployed effb0 expert on f3-4 -> scripts/s09_evaltime_read.py.
3. LB (user's go needed): the splice probe (submissions/oct16_probe_loa_splice); later the LOA bundle
   (submissions/oct16_loa_bundle: LOA-only extra members in the crop fusion, unit-tested).

## GPU block result (10-16 07:37-10:04, Master's GO; ~2.5 GPU-h; out/s13_p1_report.txt, s08_k16.txt, s18_ke64_f012.txt)

Pre-registered P0 + P1 on folds 3 + 4 (1 seed), each arm read as one more crop-fusion group in the 0.961 OA block vs a count-matched
seed control (v2cor_s1); score S = mean(text, tsev) delta; seed noise unit u = 2.0 (s2 vs s1) -> promotion bar S >= +2.0.

| change | S (f3 / f4) | gold LOA (rb_loa, jackknife) | verdict |
|---|---|---|---|
| one more effb0 v2cor seed member | +1.9 / -1.9 | - | a wash |
| v2cor + bml (oedema aux) | +1.3 / -4.7 | +9.7 (+7.1, +6..+12) | no gain; as a full expert PF / LM gold -10 / -9 |
| v2cor k16 (16 train windows, eval 64) | -0.1 / -10.3 | +3.9 (+9.5) | no gain |
| v2cor gradeH (definition head), v1 / T head | +3.6 / -2.4; +1.5 / -0.5 | +9.7 / +7.7 (+2.4) | no harm, LB-sized |
| v2cor silL (silent LOA calibrated) | -1.3 / +1.8 | +21.3 (+9.5, +15..+23) | no harm, LB-sized |
| inference-only k_eval 64 (f0-2, deployed structure) | -2.6 / +0.2 / +0.0 | +13.5 (+16.5, +8.5..+14.9); 5-column sum +15.9 | report-vs-gold trade-off |

Verdict: no LOA change improves the report reads; the gold-direction changes (silL, gradeH, k_eval 64) can only be decided on the LB.
Single-member gold deltas sit inside the seed-pair gold noise (-7.7), so gold alone is not evidence here.
LB: splice probe prepared (submissions/oct16_probe_loa_splice: 0.961 + LOA at k_eval 64; kernel-only; pair retired); optional later bundle probe
(0.961 + loa_extra {silL, gradeH} 5 folds; k_eval 64 and C11 separate choices) only if option (b) passes on f0-2 (PLAN.md; ~130 GPU-min;
read = scripts/s19_bundle_read.py; deploy code in submissions/oct16_loa_bundle). Count-matched evidence so far = f3 only: S +6.7,
gold +17.4 (rb_loa +4.7); f4 has no 2nd seed control (ledger 10:43 correction).

## Older learnings, re-read for LOA (sceptical audit, 10-16 01:00-02:10)

| learning (source) | status for LOA now | why |
|---|---|---|
| compartment crop expert transfers to the LB (oct10_oa E1 +0.048 OA sum; v2cor E3) | VALID | LB-proven; the 0.961 LOA is 0.6 expert |
| de-coupling k 0.4 for LOA (C1', nested text3 + strat3) | VALID | gold rb_loa and meets-vs-below favour it (s11 / sg sweep); separate PF de-coupling loses text |
| crop-level fusion of the OA experts (oct13 A9) | VALID | robust gold +27.7 holds (Fusion jackknife); maxmix / gates = report-vs-gold trade-offs (s05 / s06) |
| 'nothing stacks on v2cor' (clean, init, iso, posup, oaonly; oct10_oa) | PARTLY OUTDATED | judged on the 3-OA mean (text3); re-read on LOA alone: all within ~1-2 slot-noise of a member copy (s02), so 'no LOA gain detected', not 'no gain' |
| oedema aux (bml) closed (oct10_oa 09:43) | REOPENED | closed for PF loss on text3; on LOA alone every read up on f0 / f1 (b0 bags) -> P1 arm on v2cor |
| px384 / split / lse / top2 | OPEN, low | LOA up on f1, down on f2 (focal fold); lse / top2 gold gains ride on 2 studies |
| de-averaging beats attention on gold (A67) | OUTDATED for decisions | LB E2 < E1 on the OA sum; for LOA the gold gain is the 2 dominant studies |
| R50 expert adds nothing (3-pipe recipe) | VALID (LOA too) | s02: R50 f0-2 slot text -0.4 |
| Codex LOA reads as a validation set | DEAD | failed its gold gate (10-09) |
| text over-states LOA vs public by +0.050 (oa_research, main-only) | VALID as a warning | probe P0 would re-measure it for 0.961 |
| PL blend = a LOA improver (oct12_oa_pseudo pl_s038 f3) | OPEN, last | 1 fold, leaky teacher; student partly redundant with its teacher package |
| per-site / per-vendor calibration (oct10 M5 / M7, Fusion P5a) | VALID (do not) | report-fitted offsets lose held-out |
| mirror TTA, MOA head as lateral reader (A37, s04) | CLOSED | both lose |
| 'LM coupling is an error' (ERRORS P1) | NUANCED | gold: LM tears in 9 / 11 LOA positives vs 14 / 47 negatives |
| 'focal posterior lesions = the LOA weak spot' (A59 / s07) | NUANCED | largest text error, but the host rule (>= ~1 cm) makes many test-negative; s12: no edge pile-up |
