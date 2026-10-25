# oct20_final_v2: FINAL fold-avg submission, STRICT (Master 10-20)

- Package: oct20_stack_all_v2 (every LB-proven improvement; LB_AUDIT.md) + fast.py 972cdfa1 (parent_release) + strict fast_predict_all.
- USER: "we absolutely do not need any fallbacks. we need robust inference pipeline that works or fails loudly"; then (to Brainstorm)
  "if we check things which are guaranteed and should be true for all studies, it is better to fail loudly. if we check things which are
  possible some study can fire ... then we can skip it".
- Raises (all GUARANTEED for a valid study; Brainstorm's code read): series-csv coverage; Basin header scans + finite outputs (per worker);
  main / main-SD pass; crop stages (OA / bone / synovitis) without exception; window outputs present; every part finite + covering every
  study; 2 GPUs for > 3 studies; any failed chunk.
- Not raised (POSSIBLE for a valid study): series-level skips of one unreadable series (printed); the MCL zoom member w5b_mcl ('mclz' bag is
  coronal-only) gets zero crops on a study without a readable coronal series and then outputs a finite constant logit (no check sees it;
  0 of 4,407 train studies lack a coronal series).
- Kernels: v1 knee-oct20-final-v2-strict (Basin counters dead in FAST mode: Brainstorm review; stopped by the user / superseded);
  v2 knee-oct20-final-v2-strict2 (fixed; test_strict.py 9 / 9) = the submission candidate.
- Gate: read_check.py kaggle_output_v2 8 (Kaggle FAST strict vs the local sequential run <= 1e-3; finite; strict lines; timing).

## Kaggle commit check of v2 (knee-oct20-final-v2-strict2 v1, 10-20 02:10)
- COMPLETE in 1,301 s (~22 GPU-min): FAST STRICT on the 8 edge studies in 2 chunks (849 s / 187 s; chunk 0 includes a one-off cold read of
  the crop-expert weights from the dataset mount, crops_load 441 s vs 21 s warm) + the 3 sample test studies (sequential).
- Strict: "STRICT (chunk 0)", "STRICT (chunk 1)", "STRICT (sequential <= 3 studies)": no fallback fired (152 parts checked; Basin
  series-level skips 0); no Traceback; submission.csv written (3 rows, finite, [0.035, 0.861]); parent RSS 2.2 / 2.4 GB.
- Parity:
  - Kaggle FAST v2 vs Kaggle SEQUENTIAL (bench_out_off ORIG, same 8 studies, the 11 non-Effusion columns): max |dp| = 0.0 exactly.
  - vs the LOCAL sequential v2: 2.5e-3 (LOA) / 1.0e-3 (Contusion), the same as local vs Kaggle sequential = cross-GPU noise (5090 vs T4) in
    the crop experts. read_check's 1e-3 local tolerance flagged it; Effusion vs local 1.2e-5.
- Runtime projection (stress model + this run): ~300-360 min + ~7 min cold load, limit 540.
- GATES: PASS -> ONE official submission (USER: "make at most one official kaggle submission (if all gates pass)").

## LB (10-20 05:37)
57416173: public 0.964 (new best), scored in 205 min. Team LB rank 31 / 6,249, last of 13 teams in the 0.964 band -> ~0.9635-0.9636.
