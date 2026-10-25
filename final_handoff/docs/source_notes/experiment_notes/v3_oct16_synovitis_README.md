# oct16_synovitis: Synovitis improvements that transfer into the 0.961 fold-average package

## SEALED 2026-10-16 ~22:25 (user directive ~22:15: conclude; learnings to Experts V3) - final state

- Handoff: HANDOFF_experts.md (learnings by evidence, seed rows, pending arms with exact queue lines, tools, deploy paths, open ideas).
- Leads, both on the LB as splice probes of the 0.961 fold package (Synovitis only; the 11 other columns = 0.961's own Kaggle check):
  - ConvNeXt-T in both effb0 crop slots: G gold +7.2 [+3.7, +9.9] vs a both-slot seed swap -3.6; local fused gold-58 0.8937 vs 0.8865.
    submissions/oct16_syn_cnx_probe, 57283240 (accepted 21:01 UTC; est 288 min).
  - Fat-sat-only image heads (m_fs05 + n_fs05): G gold +4.8 [0, +9.9]; local gold-58 0.8901 vs 0.8865.
    submissions/oct16_syn_fs_probe, 57279368 (accepted 18:30 UTC; est 257 min).
  - Reads (Master): lb_order.py REF 57154196 (hidden order of the tied 0.961 display). Results go to each probe's README Result table.
- Not run, handed to Experts V3: x_mcnx_fs (ConvNeXt FS-only image head), x_cnx_dp2, x_cnx3e-5 (queue lines marked #HANDED).
- Closed: label / context targets, rules text-aux, PL (user-dropped), px288, ConvNeXt lr 1e-4, extra effb0 crop designs (saturated).
- Runner stopped, gpu_go 0. Folder kept as is (runs/, refs/ for reuse by copy).
- FINAL (user, direct, 10-16 ~22:50: "seal the folder, handoff all your progress and learnings to the experts v3 session, inform the
  master session, then quit"): HANDOFF_experts.md complete (sections 1-7, incl. where the levers apply in the package); Experts V3 and
  Master informed; session ended. Detached Kaggle timers (submit_and_time.py) of both probes keep writing submit.log /
  submit_timing.json in the probe folders until scored.

## Original header

Started 2026-10-16 00:45 (user goal, user_directives.md; 24 h). Session 'Synovitis V3' (knee-abnormality-2026-fe). GPU owner: Master
(experiments/v2/oct14_text_labels/MASTER.md); this session is last in the GPU order and works on CPU until 'GPU GO'.
Files: ledger.md (time-ordered, every number), ideas.md (ranked ideas), VALIDATION.md (synovitis read practices), scripts/, out/.

## Question

Synovitis in the 0.961 package (submissions/oct13_f5_win) is G = 1/3 image heads (m_prob05 main effb0, n_prob05 NFNet; Codex-read
image head) + 2/3 crop experts (s_fs, s_fc125 on effb0 and NFNet; localiser crops of pouch / Hoffa / popliteal, FS windows, Codex read
target w 1 + text aux w 0.5), folds 0-2 each, crop part fused at crop level. What change, transferable as a drop-in into that package,
improves the hidden-test Synovitis AUC?

## Prior learnings (line-by-line read of oct10_synovitis, oct11_synovitis_crops, oct12_syn_g, v3 sessions; sceptical notes in brackets)

1. Supervision moved the public score most: text -> Codex image reads (public ~0.87 -> ~0.93), more reads and raw-probability targets
   (+0.012). Model / crop changes after that moved held-out reads +11..+15 but public only ~+0.004 (G) or < +0.012 (P4 / P6).
2. Codex training-style reads score 0.81-0.83 AUC on gold; careful reads 0.878. G scores 0.887 on gold (this session, A2): the models
   already beat their labels and sit at the careful reader's level.
3. Reader copying: a stronger image weight in the main pipeline raises reads and lowers gold; in the crop expert the image target must
   dominate (w_img 0.25 -14.6 reads). Multi-task text targets denoise the main pipeline's image head (image-only gold 0.785).
4. Crop experts: FS-only windows +4.3 (3 / 3 folds; the eval ablation shows non-FS / coronal windows carry nothing); 13 further
   screens (bags, losses, aug, init, crop size, context, per-bag readout, attribute heads) found nothing better. Curves plateau by e15-17.
5. Encoder diversity beats seeds (NFNet as 2nd member +4.5 over a seed pair); R50 adds nothing (leave-one-out); the 3-pipeline
   crop-only entry (with R50) was <= the specialist on the LB.
6. Gold FPs: effusion-positive knees without synovial thickening / debris (often trauma); gold FNs: synovitis without effusion (Hoffa).
   Reports discount synovitis in traumatic effusions (A4), Codex does not, but a post-hoc trauma discount hurts gold (A4, 10-10).
7. v3 lineage (Labels Work / Fusion): v1 Synovitis text labels are the weakest of all classes; rules-set text Synovitis gives
   independent Codex +44.6 (main text head, f3); calibrated silent Synovitis raises main's text-head gold strongly (being re-checked
   with a gold jackknife); a label-trained main as a 3rd G image head adds nothing; post-hoc fusion / gating is saturated.
8. No local metric reproduces the synovitis LB order; clear-cut reads ranked P6 > P4 > base right, gold-58 did not. Public synovitis
   comparisons are ~+-0.01 AUC noisy; a synovitis-only probe reads +-0.006.

## Status (20:45): GPU phase since 10:05 (Master's GO); card with Lateral OA V3 until ~21:50

- Probes (both splice designs on the 0.961 fold package, Synovitis only, each on the user's direct go):
  - FS-only image heads: submissions/oct16_syn_fs_probe, 57279368, scoring (ETA ~23:50).
  - ConvNeXt-T in both effb0 crop slots: submissions/oct16_syn_cnx_probe, local gates PASS (gold-58 +7.2 fused), Kaggle check
    running (kernel v1 since 20:32), then submit.
  - Order read for ties: lb_order.py (server-side public-score sort, validated on known pairs).
- Next GPU (~21:50+, Master's order): ConvNeXt-T FS-only image head x_mcnx_fs f0-2 (queued, gpu_go 0), then ConvNeXt variants.

- Plan: PLAN.md (pre-registered; Master-confirmed with amendments; PL stage 1 / 2 rules from the user). Queue: queue.txt (gpu_go 0).
- Tools: scripts/read_arm.py (member vs refs mean + band, pair read, G-operator swap vs the seed-row swap, gold jackknife, lookat
  negatives, Effusion guard), scripts/gold_flips.py (per-gold-study contributions + reads + report excerpts), scripts/build_pl.py.
- Trainers: syn/ (crop expert; + llrd / drop_path / holdout), mainp/ (m_prob05 trainer + fs_only). CPU smokes / data checks pass.
- Probe designs (prepared only): PROBES.md.

## Results so far (CPU; details and numbers in ledger.md)

- A1: G's 6 members are near-equal and saturated (leave-one-out <= 1.6 reads).
- A2 / A8: G (gold 0.887) is at the careful Codex reader's level and agrees with each training read MORE than the reader agrees
  with itself (catch copies: G vs read 0.88-0.90, read vs re-read 0.81). Held-out Codex AUC is capped by reader noise: Synovitis
  decisions rest on gold-58 (+ jackknife) and the LB.
- A4: reports discount synovitis in traumatic effusions, Codex does not; a post-hoc trauma discount lowers gold -> closed.
- A5: every text Synovitis label set is weak on gold (0.67-0.78); rules is the best Codex-independent set.
- A6: member gold peaks ~e13 while reads rise (late reader copying), absorbed in G -> epoch choice closed.
- A7: G is weakest on Hoffa-signal-without-effusion (crops 812 > image heads 748) and inside effusion-positive knees.

## GPU results (10-16 10:05-; details ledger.md; gold x1e-3 with leave-one-study-out range; Codex = no-harm, capped)

- POSITIVE LEAD: fat-sat-only image heads. m_fs05 (main effb0, m_prob05 recipe + fs_only) member gold +16.1 [+8.7, +23.0] vs a 2-seed
  baseline; n_fs05 (NFNet, 3 FS slots) +16.7 [+9.9, +21.1] vs the deployed head (one seed). Both swapped into G: +4.8 [0.0, +9.9] (share
  expectation ~+5.3); the gain survives at its share in every smaller ensemble (absorption ladder). Splice probe staged:
  submissions/oct16_syn_fs_probe (user approval needed).
- POSITIVE LEAD (gold-only, one seed): ConvNeXt-T crop expert (lr 5e-5, s_fs design). f3+f4 screen: reads +5.4, gold +5.4. f0-2: member
  gold +26.9 [+14.3, +32.9] vs a 2-seed s_fs baseline (Codex flat). Swapped into s_fs in G: +8.4 [+5.0, +11.2] vs the seed swap +3.6 (share
  expectation +4.5 over the seed mean; observed +4.8). Ladder: survives at every size. As an added member or a 3rd pipeline: 0.0 / +2.4.
  With both FS image heads: +15.5 [+7.4, +18.6] (sum of singles +13.1). Next: ConvNeXt in the s_fc125 slot (running).
- CLOSED (gold, 1 seed): report-context synovitis targets for G's members (text-aux v1syn image head -10.8; blended image target -47.8;
  post-hoc text blend negative at every share); rules text-aux (no gain detected, f3); px288 (no gain detected, f3+f4); PL with G as
  teacher (member +19.7 but G +0.0 = imitation; then dropped by the user); G saturation for extra effb0 crop designs.

## How to resume / GPU GO checklist

1. On the Master's 'GPU GO' (card verified empty by the Master): `echo 1 > gpu_go.txt` (slots_max 1).
2. Runner (repo root): `setsid nohup bash experiments/v3/oct16_synovitis/run_queue.sh > experiments/v3/oct16_synovitis/logs/runner.log 2>&1 &`
   (writes runner.pid; stop only by that PID; to pause, set gpu_go 0 and let the running arm finish).
3. PL release watcher: `setsid nohup bash experiments/v3/oct16_synovitis/scripts/release_pl.sh > /dev/null 2>&1 &` (builds
   data/img_syn_pl.csv after r_fs / r_fc125 f3-4 and releases the pl_fs lines).
4. Event Monitor: `tail -n0 -F logs/events.txt | grep --line-buffered -E "start|end|gave up|PL|FAILED"` + the 10-min heartbeat.
5. Reads (scripts/, f3 / f4 screens vs refs; f0-2 arms in G):
   - `read_arm.py x_px288 --folds 3 --refs r_fs,r_fs_s1 --pair r_fc125`
   - `read_arm.py pl_fs --folds 0,1,2 --refs dep:s_fs --pair dep:s_fc125 --slot s_fs --leaky` (+ gold_flips.py; gold-only gate)
   - `read_arm.py m_fs05 --folds 0,1,2 --refs dep:mprob --pair "" --slot mprob --seedrow mprob=ref:m_prob05_s1`
   - Labels Work's test: `read_arm.py ref:sx_fc125_v1syn --folds 0,1,2 --refs dep:s_fc125 --pair dep:s_fs --slot s_fc125`
6. `scripts/status.sh` = the 10-min snapshot. 'GPU FREE' to the Master when the queue is done and the card is verified empty.
