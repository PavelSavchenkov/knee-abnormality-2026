# oct16_experts: Experts V3 — the main experiment session (v3 lineage)

- from_master/INDEX.md: everything the Master session produced for this session (public reviews 10-13 / 10-17 with the sampling / crop
  catalogue, lineage_oct16.md, IMAGE_LABELS_NEXT.md, OVERNIGHT_2026-10-16.md, per-class package OOF table, public single-model and
  external-data facts). New Master research outputs land there directly.

Started 2026-10-16 22:00 BST (user /goal, 48 h -> 2026-10-18 22:00). design.md = the user's job description (do not edit).
User 10-16 ~22:05 (via Master): this session is the main workhorse; Lateral OA V3 and Synovitis V3 concluded and handed over
(their HANDOFF_experts.md); Master aids / double-checks (send every pre-registration and every read). GPU: mine since 22:35 (Master's
GO); no per-arm GO needed; keep the card busy.

Base: the 0.961 fold-average package submissions/oct13_f5_win (+ LOA at k_eval 64, LB-positive, now part of the fold recipe). No
all-data training (user). PL de-prioritised (user). Every official submission needs the user's explicit go in THIS session.

## Afternoon 10-17 (12:00 -> ; details in ledger.md, prior_art.md section 16)

- User rules: U19 LB probes apples to apples (fold members only on the 0.961 fold package; Q14a / Q14b / s5 zoom probes withdrawn);
  U21 judge a one-member change by its member-alone gain vs a paired seed first, then transfer vs share (SD1 / SD1b corrected to LEADS);
  U22 expand stop-grad series heads (main, OA expert, Basin, crop experts); U23 coupling read next to every accuracy read; U24 region-
  restricted class heads.
- Learned: a crop / route de-couples a class only if it EXCLUDES the other class's evidence (Z1 notch: ACL de-coupled + better; W5b MCL
  zoom: couples MORE with fluid classes; OA compartment bags de-couple across compartments only; routing with the whole knee view = no
  de-coupling). SD1's gain is multi-task (no de-coupling); SD1b de-couples. Side flags and post-hoc Contusion <- Fracture: no lever.
  LP1 (OA expert px 384): no gain detected (member alone and swap).
- GPU queue: sfd3 (sag-FS series dropout, U20 b) -> sd1s12 (SD1 seed) -> oasd (OA SD) -> mes (strict routing + cross-class stop-grad
  aggregator) -> sdcg (per-class series groups) -> P1b (PL OA student, 5 folds) -> bsd_nf / bsd_r50 (Basin per-slot heads) -> bone pair.
- LB 10-17 ~15:20: s5-Q6 (s5 + post-hoc MCL <- Contusion b 0.2) = public 0.963, NEW BEST -> MCL follows gold; dc2m enabled; MCL zoom
  probe (fold base) live.
- 10-17 ~16:15: MCL zoom probe (W5b, fold base) SUBMITTED by Master on the user's direct go (ref 57315944; result ~19:40 BST).
- LB 10-17 ~16:00: s5-Q8 (LM <- ACL b 0.1) = 0.963 too (rank 1). Combined probe cancelled (user U28: per-column gains add). Next single-
  column s5 candidate built: submissions/oct17_probe_s5_eff_dec (Effusion <- PF OA b 0.3; non-gold PASS on the Q6 + Q8 base, gold agrees;
  offline gate PASS). Ranked after it: Q9 (LM <- MM on top of Q8; report-led, gold-negative), ACL <- Effusion (tiny).
- Probes to propose (fold package only): ACL zoom 40 mm (gated, recommended); MCL zoom (only if s5-Q6 reads gold-like); P1b after its
  5 folds. Built, ranked below P1b (tomorrow): M1 ConvNeXt LOA-only (submissions/oct17_probe_loa_m1, gates not run). Nothing is submitted without the user's go in this session.

## Night summary (10-16 22:00 -> 10-17 06:00; details in ledger.md, per-class rows in ideas.md / PROBES.md)

What won on the LB tonight (Master's probes, before / around this session): ConvNeXt Synovitis crops (57283240) and silL, a 2nd
differently trained LOA member (57284893), both above 0.961 on the fold package. LOA k_eval 64 won on the fold package but not on s5.

What this session found (GPU screens, package-level reads in the shipping form; 1 fold unless noted):
- Failed recipes (directions stay open): xF / explicit-cell OA stages, ligament expert, ConvNeXt OA swap, per-bag window budget, PF /
  meniscus / Baker's zooms, MCL tight zoom with 'below' as negatives, training-side de-coupling aux (8 pairs), per-series heads (2
  recipes), merged multi-view experts (effb0 and ConvNeXt), ACL notch-view member (failed its 2nd fold), window dilation.
- Why generalist arms rarely help: a better generalist reaches the package only through ACL / MCL (and 40 % of OA / menisci); experts
  absorb the rest. ME2 (ConvNeXt, 9 views) is the best single model so far (+1.7 text vs our main refs) but adds nothing to the package.
- LOA: decide on full-package gold beyond a count-matched seed member. A seed member alone lifts LOA gold +11.6; silL adds +5.8 more;
  other differently trained members looked similar on only 2 gold fold models (gradeH / bml: weak), and the ConvNeXt OA member's
  LOA edge went +9.7 (3 models) -> -1.9 (5 models): gold reads need all 5 fold models; silL (5 models) is the one well-supported member.
- MCL: four independent local sources point the same way (de-couple MCL from Contusion / trauma context; gold-led): post-hoc Q6, DC1
  aux, ME1, the MCL zoom W5b (member gold 0.986 vs 0.948). 'below' cells as weighted negatives stop a rare class from learning.
- Attention: temperature 0.25 sharpens the OA expert without loss and shows the expected planes per class (menisci leave axial, MM
  goes to peripheral sagittal slices, OA positives to coronal).

## Waiting for the user's go (10-17 ~04:00; Master's order for today's slots; nothing is submitted without your go HERE)

| # | probe (folder) | one assumption | status | expected |
|---|---|---|---|---|
| 1 | Q8 LM <- ACL de-coupling b 0.1 (submissions/oct17_probe_lm_dec) | is the LB 'report-like' or 'gold-like' for LM coupling | commit run + gates PASS | +0.0004..0.0006 if gold-like, ~-0.0002 if report-like |
| 2 | **WITHDRAWN 10-17 12:20 (user: LB probes must be apples to apples - fold members are probed only on the 0.961 fold package; never submit)** Q14a s5 + silL LOA member (submissions/oct17_probe_s5_silL) | does the fold-package LOA win (silL 57284893) transfer to the all-data s5 = the final package | local PASS; Kaggle commit run v1 COMPLETE, SUBMIT GATE PASS (non-LOA vs Kaggle s5 <= 4e-6, LOA Kaggle vs local 0.0012, est 281 min) | 0..+0.001; <= s5 drops silL from the final package |
| 3 | Q6 MCL <- Contusion de-coupling b 0.2 (submissions/oct17_probe_mcl_dec) | training-free MCL de-coupling (4 independent local sources agree) | commit run + gates PASS | +0.0003..0.0005 gold-like, ~0 report-like |
| 1s / 3s | Q8s5 / Q6s5: the same de-couplings on the all-data s5 package (submissions/oct17_probe_s5_lm_dec, oct17_probe_s5_mcl_dec) | do they transfer to the final (all-data) package (k64 did not) | built 08:25; Kaggle commit runs COMPLETE, GATES PASS (base = s5's Kaggle check exactly, decouple 2e-16, 0 failed, runtime = s5) | gold-58 LM +5.0 / MCL +6.8 on s5's check |  **SUBMITTED by Master on the user's go in Master's session (10-17 ~11:40): s5-Q8 57307925, s5-Q6 57307950; results ~15:10 BST (Master corrected 12:57)**
| 4 | **WITHDRAWN 10-17 12:20 (user: LB probes must be apples to apples - fold members are probed only on the 0.961 fold package; never submit)** Q14b s5 + seed LOA member (submissions/oct17_probe_s5_s1) | only if Q14a > s5: labels / diversity vs averaging | local PASS; commit run v2 COMPLETE, SUBMIT GATE PASS (LOA 0.0022, est 291 min; v1 transient CUDA error in s5's Basin) | decides what the all-data retrain adds |
| 5 | Q12 Synovitis ConvNeXt crops + effb0 kept, crop share 0.75 | on top of the LB-won ConvNeXt Syn | NOT built (slot-filler, near coin flip; build only if a 5th slot would otherwise go unused) | small |
| 6 | Q13 fold-package seed splice | mechanism only | not built | - |
| 5b | W5b-MCL: MCL zoom member at 0.25 (submissions/oct17_probe_mcl_zoom) | does MCL on the LB follow a gold-led, oedema-sensitive, trauma-decoupled reader; only if Q6 reads gold-like | local PASS; Kaggle commit v1 SUBMIT GATE PASS 06:12 (non-MCL <= 3e-6, MCL Kaggle = local, est 282 min) | gold-like +0.0006..0.0009, report-like ~-0.0001 |
| 6a | Z1c-ACL: ACL 28 mm notch-zoom member at 0.4 (submissions/oct17_probe_acl_zoom28) - ALTERNATIVE (reports +3.2 vs +2.7 is inside noise +0.56; gold-58 lower with 24 ACL positives) | the LB test of the attention-derived zoom (U16) | built 09:33, local PASS; Kaggle commit SUBMIT GATE PASS (est 270 min; Kaggle gold-58 ACL 0.9890) | ~+0.0003 |
| 6b | Z1-ACL: ACL 40 mm notch-zoom member at 0.4 (submissions/oct17_probe_acl_zoom) - RECOMMENDED ACL probe | the LB test of the attention-derived zoom method (user U16) | built 08:50; local PASS; Kaggle commit SUBMIT GATE PASS (est 317 min, slow machine; Kaggle gold-58 ACL 0.9939) | ~+0.0002 (ACL near ceiling) |
| 6s | **WITHDRAWN 10-17 12:20 (user: LB probes must be apples to apples - fold members are probed only on the 0.961 fold package; never submit)** s5 versions of the zoom probes: ACL 40 mm (submissions/oct17_probe_s5_acl_zoom) and MCL (oct17_probe_s5_mcl_zoom, only if s5-Q6 reads gold-like) | do the zoom members help the final (all-data) package | built 12:00; local checks after LP1 (~14:00); then Master's Kaggle steps | smaller than on the fold package (s5 averages 5 seeds) |
| 7 | Q7 Fracture <- Effusion | demoted (inside its null) | commit run PASS | ~0 |

User 10-17 ~12:20 (standing probe rule): LB probes are apples to apples. Fold-trained members are probed only on the 0.961 fold package
(read vs 0.961 by hidden order); all-data members only against the all-data package (all-data training is prohibited for now). Post-hoc /
inference-only transforms with no new model stay valid on s5 (s5-Q8 / s5-Q6). Q14a / Q14b / both s5 zoom probes are withdrawn. Whether
silL / zoom members enter the all-data package is decided only by the end-of-competition all-data retrain (user-owned).

## Files

- design.md (user), user_ideas.md (user suggestions ledger), ledger.md (time-ordered: every pre-registration, read, decision, review),
  prior_art.md (everything tried before, by class and lever; built from a full read of every experiment folder), ideas.md (ranked).
- Runner: run_queue.sh + queue.txt (NAME KIND SRC FOLD key=value ...; kinds oa / bone / syn / mainp via scripts/launch.py; kind 'dump'
  = per-crop OA dumps after training, SRC = epoch), gpu_go.txt, slots_max.txt, runner.pid (stop only by this PID), logs/events.txt.
- Trainers (copies; experiments stay self-contained):
  - oa/ = LOA V3's OA expert (oa_train.py: keep_bags, out_w, k_bags, tgt_csv, pl) + llrd / drop_path + init_full / ft_mask (xF stage);
  - bone/ = Backbone Sweep's bone expert copy (hn_w, sharpen, llrd, drop_path, px, backbone, series);
  - syn/ = Synovitis V3's crop expert + ligament bags (mcl, notchsc), acl / mcl outputs, out_bag per-output bag restriction;
  - mainp/ = the m_prob05 main trainer + fs_only (image heads).
- Data copies: data/loc_pred (v1), data/loc_pred_v2, data/loc_pred_v3 (bone), loc/pred (localiser v2 for syn bags), data/tgt_v3.csv,
  data/img_syn_prob.csv, data/synovitis_labels_codex.csv, data/img_attrs.csv.
- Refs (copies of other experiments' outputs): refs/pkg961 (members.parquet, oof_f5_win.parquet = f5_eff AND f5_win variants: always
  .xs('f5_win')), refs/oa/* (OA expert OOF / gold), refs/bone/*, refs/men/* (m_csp etc.), refs/syn/*, refs/ckpt/oa_crop_v2cor (deployed
  OA effb0 checkpoints), refs/exp_* (source configs for launch.py). oa_refs/ = LOA V3's exact OA-block rebuild data (win/ crop dumps,
  config.json, oa_fuse_recipe.json, members.parquet).
- Reads (scripts/): oa_member_read.py (member vs effb0 seeds, own pooling), m1_slot_read.py (SWAP of the effb0 group in the deployed OA /
  meniscus crop fusion, 5 columns, damage checks + gold jackknife), m2_men_member.py (extra meniscus model vs a seed), bone_member_read.py
  (fc_eval panel vs refs), syn_reads/read_arm.py (+ slib / ladder / combo: Synovitis member / G-slot reads), lib.py (exact 0.961 OA block
  rebuild), s08 / s19 / s22 (LOA V3 reads), dump_oa.py, e1_subgroups.py / p1b / p1 (CPU analyses), l1_lig_crops.py.

## State

See ledger.md (latest entries) and ideas.md (ranking). Arms so far: A1 ConvNeXt FS-only image head (Synovitis; held for the FS-heads
probe 57279368), A2 ConvNeXt-T OA expert lr 5e-5 (f1 screen), A3 bone hn_w (fallback), A4 xF stage on the OA expert (f0-2), F1 same-code
ConvNeXt OA 2e-4 rerun. CPU closed: P1 PF trochlea crops, m_csp as a package meniscus member.

## Resume

From the repo root: `setsid nohup bash experiments/v3/oct16_experts/run_queue.sh > experiments/v3/oct16_experts/logs/runner.log 2>&1 < /dev/null &`
with gpu_go.txt = 1 (the card verified free). Heartbeat: Monitor `while true; do sleep 900; echo heartbeat; done` (re-arm at expiry; cron
never fires in this host). Event monitor: `tail -n0 -F logs/events.txt | grep --line-buffered -E "start|end|gave up|empty"`.

## Status 10-17 02:15 (Experts V3)
- Probes READY (gates PASS; each needs the user's explicit go in this session): Q8 submissions/oct17_probe_lm_dec (first), Q6
  submissions/oct17_probe_mcl_dec (after Q8). Q7 check run on Kaggle (demoted). Q9 built, conditional on Q8.
- GPU queue (queue.txt; reads auto via scripts/auto_reads.sh / auto_reads2.sh / read_sd1diag.sh): ME1 merged multi-view f1 -> DC1 + dc0
  -> SD1 -> W5a MCL x2 -> ME5 x3 (attention diagnostic) -> S-D1 f1 -> bone f1 (A3, A6, FR1, W8) -> LP1 f0-4 (filler).
- Rules: package-level deepen bars, one fold at a time, three nulls, config audit (scripts/audit_configs.py) before queue blocks.
