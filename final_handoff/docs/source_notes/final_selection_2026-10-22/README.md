# Final selection, 2026-10-22 (deadline 23:59 UTC)

Two finals. Kaggle counts the better of the two on private. OAI: no in every candidate.

## Candidates

| probe | package | models | ref | status (12:5x UTC) | gold-58 vs safe3w9 (x1e-3) |
|---|---|---|---|---|---|
| P1 | plain: safe3 + w5b 9, bfs_rep 5, oa_v2cor 5, sd1b 5 (inv1_a) | 93 | 57517691 | 0.965 in 175.7 min; hidden order #1 of ours | not in the hold (seed model +1.45e-4 private) |
| P2 | PL by group, generalists plain (best_b1) | 87 | 57520594 | pending since 10:21 | +2.5 |
| P3 | plain + PL 'both' blend (best_b6) | 117 | 57520062 | pending since 10:04 | +1.9 |
| P4 | P3 set, OA + generalists routed per column (best_b6c) | 120 | 57520478 | pending since 10:18 | +3.1 (routes partly picked on this gold) |
| P5 | P4 + plain bfs_rep / oa_v2cor / sd1b 5 (+ PL twins) (best_b6ca) | 130 | 57521751 | pending since 10:56 | +2.6 |
| safe3 | all-data 3 seeds per recipe | 81 | 57497980 | 0.965 in 190.5 min; hidden order #2 | 0 (base) |
| v2 | fold-average package | - | 57416173 | 0.964 | - |
| final118 | all-data 118 + sxm | 118 | 57497935 | COMPLETE, NO score (timeout) | not selectable |

Fast-VM finish: P2 ~13:40, P3 ~14:10, P4 ~14:25, P5 ~15:30 UTC. Slow VM: +3 h. A 540-min timeout shows by ~20:00-22:00. Every outcome is known before 23:59.

## Reading the LB (inv1.md, inv2.md, experts.md, c4.md)

- A full model redraw moves public by +-0.0004-0.0005. Close relatives (P1 vs safe3) ~1e-4; P2-P5 vs P1 ~2-4e-4. Only a gap of >= ~0.0005-0.001 is a signal.
- Public gaps carry over to private shrunk (about half or less). The max of 5 is inflated by ~1.2 sigma (winner's curse).
- v2's public is inflated by LB-selected one-seed fold wins (fold-avg ~0.9647 vs safe3 ~0.9650). Don't pick it.

## Consensus rule (every session agrees)

1. Only COMPLETED, scored submissions. Use lb_order.py for hidden-precision ties.
2. Final 1 = P1, the plain robust package. It dominates safe3: same form, more averaging, w5b 9 fixes the unstable recipe. It loses the slot only if a PL package beats it by > ~2 sigma (~0.001) AND another plain-type anchor exists (there isn't one).
3. Final 2 = the best PL / blend package by mechanism: P5 by default (max expected: PL where members are noisy, plain where diversity / label design matter, plus the extra plain seeds). Then P4, then P3, if P5 times out or scores clearly lower (> ~0.0005 below its siblings). P2 only if it clearly beats P1 and all of P3-P5 fail.
4. Never two PL siblings (P4 + P5 share most models, so best-of-2 gains ~nothing).
5. Runtime margin: Submission Timing says scoring is one run (public + private); Investigate #2 is cautious about a rescore. Prefer a final whose scored minutes leave a margin under 540.

## Sources

timing.md (Submission Timing), inv1.md (Investigate #1), inv2.md (Investigate #2), pl.md (Pseudo Labelling), experts.md (Experts V3), c4.md (box 2 / C4 reader record). Submission folders: submissions/oct22_probes/p*/ (description, ref, submit.log, submit_timing.json).
