# Investigate #2: input for the 2 finals (2026-10-22 ~12:30 UTC; OAI: no)

Source: work/alldata_gap_2026-10-22/inv2.md (all checks + scripts). Written for Master's final pick.

## Findings that matter for the pick

1. No conversion or inference error in the all-data packages.
   - safe3 == the fold e2e form key by key (only file lists differ); same epochs / schedule; train 4,349; labels unchanged.
   - Kaggle T4 at scoring layout (final118 on gold-58, 4 FAST chunks) == local; every member == its training-time dumps (corr >= 0.996).
   - So public differences between the all-data packages are model draws (seeds / PL) + selection, not bugs.
2. Public seed luck in the fold-probe wins.
   - Every recipe's fold runs share ONE seed across folds, so each probe win was one seed draw.
   - Gold seed rows of the merged forms: Fracture v3c_fs 0.947 vs 0.917; Contusion all 0.953 vs 0.958.
   - Scaled to ~375 public studies: ~+-0.0004 macro per bone column per seed draw = the size of the merged wins.
   - v2 + merged wins = a chain of LB-selected forms -> its public score is inflated by selection; re-seeded all-data gives that back.
3. Estimates: fold-avg counterpart ~0.9647 +- 0.0002; safe3 ~0.9650 +- 0.00015 -> all-data ~+0.0003 on public;
   expected private advantage larger (no selection on all-data seeds; more data per model).
4. w5b_mcl (0.25 of the MCL logit) is the only unstable recipe (same-fold seed 0 vs 1 OOF corr 0.86-0.93; all-data s103 / s104 another
   mode; s102 weak). safe3 shipped s101-s103. Packages with many w5b seeds (P1: 9) fix it. Label-free noise index: MCL 2.3x and
   Baker's 1.6x noisier with 3 seeds than the fold form; other columns equal or lower.
5. Fold + all-data / plain + PL blends add seed diversity (seed-seed logit corr 0.979 > fold-fold 0.963): more averaging lowers variance on
   private. Fusion relevance follows model counts (pooled S); effect measured negligible (corr 0.998+).

## Reading today's public scores

- Public read noise between differently seeded / PL packages: ~+-0.0004-0.0007 macro (seed draws) on top of the public-split noise.
  Hidden-order gaps of a few positions inside one band are NOT evidence; only a band change or a gap > ~0.0005-0.001 is.
- Routing choices (P2 / P4) were selected on local reads -> some selection exposure; plain blends (P3, P5's plain part) less so.

## Advice for the 2 finals

- Do NOT pick v2 fold-avg (0.964): lowest public, and its public includes selection luck; less data per model.
- Final 1 (max-safe): P1 plain 93 (0.965) over safe3: the same form with strictly more averaging (w5b 9 seeds fixes the only unstable
  recipe; bfs_rep / oa_v2cor / sd1b 5 = the highest-noise columns), no PL risk. Prefer P1 to safe3 even if its hidden order is
  slightly below (inside seed noise).
- Final 2 (max-expected): the largest blend among P3 / P4 / P5 that scored within the runtime margin, unless one of them is clearly
  (> ~0.0005, e.g. a band) above the others. Default P5 (130: P4 + plain extras = most averaging, both plain and PL twins), else P3
  (117: plain + PL 'both' blend, least routing-selection exposure). Pick P2 (87, PL-routed) only if it beats P1 by a clear gap.
- Runtime check before picking: private rescoring re-runs the notebook. final118 (118 models) shows COMPLETE with NO public score
  (likely timeout / error): confirm each candidate's scored minutes leave a real margin under 540 (slow-VM +40-100 % seen).
