# oct12_oa_pseudo: pseudo-labels (PL) for Medial / Lateral / PF OA (SEALED 2026-10-12 19:03)

Started 2026-10-12 17:38 BST (user goal, 24 h; user_directives.md). Results in time order: ledger.md; ideas and ranking: ideas.md.
Plan source: work/pl_research/README.md ("Pseudo Label Research", sections 0, 2-4, 9, 9f).

## Final state (SEALED 2026-10-12 19:03 by the user, after ~1.5 h of the planned 24 h)

GPU arms run (all effb0 v2cor experts, 20 epochs, seed 0, scored at epoch 19, leaky option-1 teacher T_x): pl_neg03 f3 (buggy =
negatives-only blend), pl_2h f3 + f4 (separate teacher head), pl_s038 f3 (blend). No seed references, no control, no leak check.

| arm (vs the v2cor expert, same fold) | fold | tm3 | trusted2 | sg3 | thr3 | gold3 | reading |
|---|---|---|---|---|---|---|---|
| pl_2h, text head, alone | 3 / 4 | +0.0063 / -0.0003 | +0.0043 / +0.0013 | +0.0047 / +0.0055 | +0.0120 / +0.0078 | +0.0059 / -0.0011 | uncertain-cell ordering up on both folds; explicit cells up on f3 only |
| pl_2h, in the 0.955 recipe | 3 / 4 | +0.0049 / +0.0005 | +0.0032 / +0.0002 | +0.0025 / +0.0039 | +0.0105 / +0.0054 | +0.0033 / -0.0021 | same |
| pl_s038 (blend m0 / n0.3 / o0.8), alone | 3 | +0.0080 | +0.0005 | +0.0029 | +0.0066 | +0.0137 | gain = Lateral OA (text +0.017, gold +0.046, sg +0.006); MOA / PF flat |
| pl_neg03 (bug: negatives-only blend), recipe | 3 | +0.0020 | -0.0012 | 0.0000 | +0.0004 | +0.0066 | no consistent gain |

Learnings (evidence strength):
1. The separate teacher head (user's first choice) is the most consistent design so far: on 2 folds it improves the ordering of the
   uncertain / borderline cells (sg3, thr3 meets vs below, un3 unsettled vs negatives), the cells PL is meant to fix, and on f4 the
   student beats its teacher there. Its explicit-cell text / trusted gain (f3) did not replicate on f4. Weak-moderate (1 seed per
   fold, leak unmeasured). Estimated honest effect on the LB: below one probe's resolution (+0.0003..0.0005 macro) unless the
   uncertain-cell gain is what the expert-read test rewards most (gold: PF unsettled 7/8 positive, below 1/8).
2. Blend targets act mainly on Lateral OA (1 fold). Weak.
3. PL students learn faster early and do not overfit; references catch up by ep 17-19; no signal for longer training of the 2-head
   student (curves.py; moderate on 2 folds). 30-epoch pair never run.
4. Filtering by teacher disagreement: dropped (out/disagree.txt; moderate): teacher-low 'meets' are real focal / posterior lesions.
5. Confident teacher / model predictions are right 96-100% on 26,852 trusted cells; models are under-confident at the top
   (out/calib.txt; strong). Confident-only PL arms were queued, never run.
6. Validation: text labels are noisy; read tm3 + trusted2 + sg3 + gold3 together. pl_neg03 shows why: +0.002 text, -0.001 trusted.
7. Process: dump trained targets before reading a result (a default-threshold bug disabled the blend in the first arm; caught by
   scripts/targets_dump.py; trainers now log per-status target means). Never pkill runners by name (pidfile).

How to continue (queue.txt keeps the planned arms, in priority order):
- pl_2h_txt f3 (control: 2nd head on text targets), v2cor_s1 f3 / f4 (seed noise of every metric), pl_s038 f4, pl_s038_2h f3.
- Stage C leak check for the 2-head design: in_m_h3 / in_e_h3 (8 inner teachers, hold=3) -> scripts/build_nested.py --hold 3 ->
  pl_2h_T2L / pl_2h_T2H twins.
- Then 2-head variants (pl_w 0.5 / 2, lmse / rank losses), cross-training teachers (T_ne, T_b), blend family (u7, a1, conf, tie),
  main student pl_m_s038, 30-epoch pair. Deployment (folds 0-2, effb0 + NFNet experts, package recipe refit) only after f3 + f4 +
  stage C agree; LB needs the user's go.
- Runner: scripts/run_queue.sh (gpu_go.txt gate, slots_max.txt, runner.pid); evaluation: scripts/eval_student.py, curves.py,
  targets_dump.py, diversity.py.

## Question

Can OOF teacher targets (PL) make the OA columns of the best package (submissions/oct11_oa_3pipe, LB 0.955) better on the
expert-labelled test? There, OA = 0.4 x the 3 pipelines' OA (b) + 0.6 x the mean of v2cor crop experts (effb0 5 folds, NFNet-L0 folds
0-2) + de-coupling. The cheapest student is the effb0 v2cor expert (9.5 min per fold alone on the GPU), the member with the largest
weight in the OA recipe.

## Validation regime (pre-registered 17:50)

Teacher OOF -> student on the same folds (option 1 of pl_research 9f): fast, mildly optimistic; the size of the leak is measured once
(stage C). Gold-58 and the LB are leak-free.

| stage | what | judged on | pass rule |
|---|---|---|---|
| A (CPU) | teacher compositions and alpha tables as targets (scripts/teacher.py) | teacher OOF panel; target AUC on gold-58 (veto); target density per finding x status | target >= text on gold |
| B (GPU) | student arms on fold 3, then fold 4 (both never used for OA choices); refs: v2cor seed 0 (exists) + seed 1 (same fold) | package-level OA on the fold: recipe(b, student) vs recipe(b, v2cor) [replace], and recipe(b, mean(v2cor, student)) vs recipe(b, mean(v2cor, v2cor_s1)) [add, count-matched]; metric tm3 = mean(text3, md3) (LB-validated for OA: LB gain ~1.4-1.8x the text gain); also alone; secondary thr3 / un3; gold3 veto | f3: tm3 >= +0.002 vs the matching ref -> f4; <= 0 -> drop or change. f3+f4: both folds > 0 and mean >= +0.0015, gold3 not below -0.005 |
| C (GPU, once) | leak check on fold 3: 4 inner experts on 3 folds each -> honest expert-only teacher; honest and leaky students | leaky minus honest tm3 | <= ~0.002: trust stage B |
| D | deployment folds 0-2 (confirmation folds for PL: never used for a PL choice) + package recipe refit; LB probe = the 0.955 package with OA experts changed | folds 0-2 OOF recipe tm3 vs 0.955 recipe; LB 12 x dLB = OA AUC sum change (+-0.012) | LB needs the user's go |

Label noise (user 18:31): text labels are inaccurate. Gold-58 vs v1: meets 20/22 positive (LOA 5/7), explicit negatives ~4% positive,
unsettled 6/10 (MOA) / 3/9 (LOA) / 7/8 (PF), below ~2/12, silent ~4/49. text3 / md3 only score explicit cells and cannot see the
uncertain cells PL changes; text scored backwards when a change moved the label definition toward image reads (synovitis: text -0.06,
public +0.048). So every decision is read on four references: tm3 (explicit text cells), trusted2 (hand-checked cells, MOA / PF), sg3
(soft-gold status-level AUC over all cells) and gold3 (expert reads, tiny), plus the LB. A text gain that trusted2 / sg3 / gold3 do
not share is flagged, not accepted.
Metrics (scripts/pl_lib.py panel; evaluation labels are always the original v1 statuses, never PL targets):
text3 = meets vs denied / structure_normal; md3 = meets vs denied; thr3 = meets vs below; un3 = unsettled vs denied / normal
(the cells PL moves most); gold3 = gold-58 (58 studies, 47 OA positives; veto only, it misled twice on the LB).
Noise: expert alone text3 seed sd ~0.002 per fold; in the recipe ~0.0006-0.001 (OA session). Seed pair oa_crop_b0 / b0_s1 on fold 1:
tm3 -0.0005 alone and in the recipe, gold3 -0.004 alone / -0.0001 in the recipe. Measured again on fold 3 with v2cor_s1.
Why un3 / thr3 matter for PL: gold-58 rates by v1 status are MOA unsettled 6/10, LOA 3/9, PF 7/8 positive; below 0/2, 1/2, 1/8. The test
rewards ranking unsettled above clean negatives (un3) and meets above below (thr3); text3 / md3 never see those cells.
Regime iteration (user 17:55): stage C gives leaky vs honest deltas for the same design; if they agree (and gold does not
contradict), keep screening in the leaky setting.
A student is a hidden ensemble of its teacher: also report the teacher's own OOF (T_x = recipe with 4 crop experts) on the fold.

## Teachers (stage A, out/stageA.txt; 5-fold OOF means)

| teacher | text3 | md3 | thr3 | gold3 | target gold3 (st3/7 / u7) |
|---|---|---|---|---|---|
| b = 3 pipelines | 0.9601 | 0.9551 | 0.8423 | 0.9076 | |
| T_pkg = recipe(b, v2cor) | 0.9668 | 0.9635 | 0.8552 | 0.9096 | 0.922 / 0.924 |
| T_b = b + de-coupling | 0.9612 | 0.9577 | 0.8482 | 0.9131 | 0.926 / 0.930 |
| T_e = v2cor alone | 0.9651 | 0.9613 | 0.8509 | 0.9021 | 0.921 / 0.924 |
| T_x = recipe(b, v2cor + b0 + lse + m_csp) | 0.9662 | 0.9641 | 0.8566 | 0.9143 | 0.926 / 0.930 |

Text targets alone score 0.892 on gold; every PL target 0.915-0.930.

## Layout

- main/: copy of experiments/v1/oct10_oa/main (oa_train.py + PL options pl / pl_a_exp / pl_a_oth / pl_head / pl_head_w).
- scripts/: pl_lib.py (loaders, recipe, panel), teacher.py (stage A), run_queue.sh (runner; gpu_go.txt, slots_max.txt), copies of
  oa_eval.py, reeval.py, two_member.py, member_eval.py, make_oa_recipe.py, oa_stage.py.
- refs/: OOF + gold of the members (copies): sub_b0 / sub_b0_s1 / effb0_lr1e-3_R, nfnet_e14, radimagenet_e20, oa_crop_v2cor / b0 /
  lse (5 folds), v2cor_nfnet / v2cor_rad (folds 0-2), m_csp (5 folds, oct11_meniscus).
- data/: loc_pred, loc_pred_v2 (localiser predictions, copies), teacher_<T>.csv (+ _gold).
