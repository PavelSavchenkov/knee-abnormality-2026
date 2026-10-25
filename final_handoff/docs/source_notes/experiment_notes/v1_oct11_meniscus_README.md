# oct11_meniscus: better Medial / Lateral Meniscus for pipeline #1

Started 2026-10-11 22:55 (user goal, 24 h; user_directives.md). Time-ordered results: ledger.md; ideas and priorities: ideas.md.
GPU: the Synovitis Expert session is the master; arms start only while gpu_go.txt holds 1 (set on its go).

## Question

How to raise pipeline #1's (main effb0 R2, LB 0.940) Medial / Lateral Meniscus AUC on the expert-labelled test? Public LM 0.944
(keep-only probe, +-0.006); MM unprobed. Starting point: the OA expert's meniscus aux heads (experiments/v1/oct10_oa, v2cor),
which already beat main.

## Metric (scripts/men_eval.py; the same panels for every run, whatever labels it trained on)

Pooled out-of-fold, epoch 19; gold = mean logit of the fold models. Panels per finding: text (v1 meets vs denied / normal), md
(meets vs denied), thr (meets vs below = tear vs degeneration), lookat, cons (cells where v1, v3, lixin and steven agree:
favours no training-label source), v3soft / v1soft (soft labels over all studies; each favours its own source), gold-58
(direction / veto). Comparisons are count-matched (user): an expert alone vs one main model per fold; main + expert vs main + a
main seed; N-member blends vs N-member seed blends. Paired study bootstrap (x1e-3, 95%) and folds won.

## Baseline (5 folds pooled, ledger 23:20)

| | text MM / LM | count-matched pair vs main + main seed (text mean x1e-3) |
|---|---|---|
| main R2 (3 seeds) | 0.967-0.970 / 0.943-0.945 | seed pair -1.0 (noise ~+-1.5) |
| OA v2cor expert's meniscus heads | 0.9714 / 0.9513 | main + v2cor +4.6..+5.4 [+2.9, +7.4], 5/5 folds |

## Current state (2026-10-12 14:30)

- GPU block 10:12-14:24 (handed back to Fracture Contusion; queue empty). 19 GPU arms: screens on folds 3 + 4, px288 confirmed on
  folds 0-2; seed replicate measured (same-config spread ~+-2 x1e-3 on 2 folds).
- Result: the px 288 meniscus expert (m_csp) is the model-side gain (5 folds, below). Gains sit on small LM tears: radial +0.017,
  posterior root +0.022, body +0.015 AUC vs main.
- No gain detected (2 folds, 1 seed each; vs the m_cs base or m_csp; within the seed spread): new bag rules + split + men 1 (m_cs =
  v2cor), avgmax pooling (also at px 288), notch bags (but the most diverse 2nd design), soft targets, location / degeneration aux
  heads, meniscus-only loss (px 288, f3), magnification beyond px 288 (crop 0.8 at px 288, f3).
- Closed post-hoc (5 folds): eval-time window filters (FS significantly worse), checkpoint averaging, blend weights, de-coupling,
  label-source switch, head refit variants (+0.6..0.9, weak).

## Recommendation (13:25; 5 folds, count-matched, z-aligned logit means; text = v1 meets vs denied / normal, x1e-3)

Pipeline #1 meniscus columns = mean of main effb0 + OA v2cor expert's meniscus heads + the px 288 meniscus expert (runs/m_csp):
- vs 3 main seeds (the 0.940 pipeline's model type, count-matched): text mean +6.7 [+4.5, +9.1] (MM +3.1, LM +10.4), cons +7.2
  [+4.8, +9.7], 5/5 folds; gold LM 0.933 vs 0.912, MM 0.972 vs 0.977 (n.s.). ~+0.013 MM + LM AUC sum, ~+0.001 full LB.
- of which the new expert vs a main seed as the 3rd member: +2.3 [+0.9, +3.7] (5/5; LM +3.7, md +3.1); vs the existing b0 expert
  as the 3rd member: +1.3. m_csp alone vs v2cor: LM 0.9562 vs 0.9513 (+2.4 text mean).
- m_csp = men_train.py split=1 px=288 (cartilage-centred bags, trim 0.05, sagittal 64 mm, men 1 / OA aux 0.5, effb0, 20 epochs);
  test-time path deploy/men_stage.py (verified on gold-58 from DICOM: corr 0.9996 / 0.9986).
- In the LB 0.955 package (3 pipelines + effb0 / NFNet v2cor experts): its experts' meniscus heads are the free gain (+3.5 vs 2
  seeds, folds 0-2); m_csp adds only +0.5 there (saturating).
- Weak / optional: head refit on frozen features (+0.6..0.9); b0 expert as an extra member.

## Layout

- main/: men_data.py (configurable bags: compartment / tibial-cartilage centres, slice rules, trim, notch bags, mirroring),
  men_train.py (meniscus expert: BagNet, split heads, OA aux, location / degeneration aux, spatial pooling, init), copies of the
  pipeline-1 code (data.py, labels.py, model.py, train.py) and of the OA expert (oa_data.py, oa_train.py) for reference.
- scripts/: men_eval.py (panels, blends, bootstrap), screen.sh (standard per-fold screen report), scan_runs.py (old runs on the
  panel), site_patterns.py, case_quotes.py, location_auc.py, coupling.py, blend_weight.py, bag_coverage(2).py, check_bags.py
  (v2cor bag equivalence), men_viz.py (crops by eye), ablate_cpu.py (eval-time window ablations on the CPU), run_queue.sh.
- refs/: OOF / gold of every old pipeline-1 run (main_*) and OA expert run (oa_*), checkpoints for init / ablations.
- data/: localiser v2 predictions (loc_pred_v2, copied from oct10_oa), bag_cache/; out/: tables and images; runs/, logs/.
