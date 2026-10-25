# Final selection: runtime facts and rules (Submission Timing, 2026-10-22 12:3x UTC; OAI: no)

Source: Kaggle API (read-only) at 12:27 UTC, plus this session's timing work (work/infer_speed_2026-10-21/, work/alldata_final_2026-10-21/kaggle_costs.md).

## Today's and recent all-data submissions

| ref | package | models | submitted (UTC) | status 12:27 | scoring time | public |
|---|---|---|---|---|---|---|
| 57517691 | P1 plain max-safe (inv1_A) | 93 | 08:51 | COMPLETE | 175.7 min (Master) | 0.965 |
| 57520062 | P3 blend (best_b6) | 117 | 10:04 | PENDING (143 min so far) | - | - |
| 57520478 | P4 column-routed (best_b6c) | 120 | 10:17 | PENDING (130 min) | - | - |
| 57520594 | P2 PL-routed max-safe (best_b1) | 87 | 10:21 | PENDING (126 min) | - | - |
| 57521751 | P5 best-b6ca | 130 | 10:56:41 | PENDING (91 min) | - | - |
| 57497980 | safe3 (oct21) | 84 | 10-21 21:42 | COMPLETE | 190.5 min | 0.965 |
| 57497935 | final118 (oct21) | 118 | 10-21 21:40 | COMPLETE, NO SCORE | timed out (550 min after acceptance) | none |

P5 (57521751) was submitted at 10:56:41 UTC (Master's correction; my 12:27 API listing missed it).

Notes:
- P1 at 175.7 min vs the ~207 fast-VM projection = 0.85x, so it landed on a fast VM.
- final118 shows status COMPLETE with an EMPTY score: that is how the API shows a runtime timeout. It cannot be a meaningful final.

## When the pending ones will finish (read-ahead render; projections from kaggle_costs / t62 runs)

| ref | pkg | fast VM -> done by | slow (RA-run class) -> done by | worst = 540-min timeout -> outcome by |
|---|---|---|---|---|
| 57520594 | P2 87 | ~196 -> 13:37 | ~400 -> 17:01 | ~19:31 |
| 57520062 | P3 117 | ~245 -> 14:09 | ~450 -> 17:34 | ~19:14 |
| 57520478 | P4 120 | ~246 -> 14:23 | ~455 -> 17:52 | ~19:27 |
| 57521751 | P5 130 | ~273 -> ~15:30 | ~475 -> ~18:52 | ~20:07 |

- Every outcome (score or timeout) should be known by ~20:10 UTC (P5 last), or ~22:10 if a run sat ~2 h in Kaggle's scoring queue before starting (we cannot see whether a PENDING run has started). That is before the 23:59 UTC deadline, so the finals can be chosen from completed scores.
- Slow-VM risk is real today: about half of today's timed VMs were slow. On the worst class (like inv2) even an 87-model package can exceed 540 (~580 projected). A timeout costs only that submission.

## Is a completed score final?

- Kaggle code competitions run the submitted notebook on the full hidden test (public + private rows) once, at submission time. The private score of a COMPLETED submission is computed in that run and revealed after the deadline.
- So there is no later private rerun, and a completed score cannot time out afterwards. This competition looks single-stage: one hidden test, ~1,150-1,299 studies; nothing in our history suggests a stage-2 rerun.
- Caveat: I could not find the explicit rule text (Kaggle docs / forum search); it is the standard behaviour. The competition's Rules page is authoritative if it says otherwise.
- PENDING at the deadline: I found no authoritative statement on whether a submission still scoring at 23:59 UTC can be selected or counts. Treat it as unsafe: only select COMPLETE submissions with a score. With the times above, all should be complete well before 23:59.

## Final selections

- Standard Kaggle: up to 2 final submissions; this team's plan has always assumed 2 (OAI / no-OAI).
- Selection is in the competition's "Submissions" tab, by checking the boxes before the deadline (23:59 UTC).
- If fewer than 2 are selected, standard Kaggle behaviour is to auto-select to fill: the highest public-score submissions, I believe for code competitions too.
- Not verified from this competition's rules text; the UI states the exact rule next to the selection boxes.
- Errored / timed-out submissions (e.g. final118) have no score and should not be selected.
- Ties at 3 decimals: the hidden public-score order (submissions/oct16_syn_cnx_probe/lb_order.py, read-only API sort) separates equal displayed scores.
