# PL seeds for the final all-data package (2026-10-21)

User (via Master 17:0x): finish all all-data models first, then as many pseudo-label (PL) seeds as possible. OAI: no.
PL seeds ship only if the user decides. Swap rule (Master OK): a PL seed replaces the highest-numbered plain seed of the same recipe.

## Design (evidence: experiments/v2/oct12_pseudo README + review_learnings.md, oct16_experts ledger P1c-P10, p2b_dpl)

- Teacher = honest OOF of the current global-best FOLD pipeline, per column in its LB-best form (`teacher_v1.py`; forms per fold in
  `teacher_v1_src.json`). Folds 3-4 need the teacher fold runs (Master queued them behind the all-data lines).
- Target = 'ct' blend (oct12 K6r): (1 - a) y_plain + a p_T, a = 0.3 explicit report cells / 0.7 other stated / 1.0 silent.
  y_plain = the recipe's own plain targets from its own code; p_T = teacher quantile-mapped onto the recipe's own OOF scale.
- Tier A (the only design with LB evidence): sub_b0 (ct) and nfnet ('ctns': no teacher on Synovitis), oct14 trainers, NO code change
  (existing external-label input: `labels=ext:NAME` / `target=NAME` -> data/text_labels/external/NAME.csv). Local only.
- Tier B: bone b3c_rep / b3l_rep / b3n_rep / bfs_rep, oct16 trainer's existing `tgt_csv`; Opus-read cells keep the image read. Boxes.
- Not PL (negative evidence or no deployable path): R50 (shared-teacher R50 students hurt), syn crop / image-head recipes (their deployed
  heads train on image reads; P6), OA / LOA / MCL experts (P1c-P5, P7-P10 negative on gold), silsep (v4b is the LB-won label).

## Files

- `teacher_v0.py` (plumbing only), `teacher_v1.py` (parts/ per fold, --check parity, --assemble).
- `targets.py` (tier A), `targets_bone.py` (tier B); outputs read-only + md5 in `manifest.txt`, reports `targets_*.txt`.
- `check_tier_a.py`, `check_tier_b.py`: launcher dry diff (only seed + the target key differ) + target dump through each trainer's own code.
- `targets_c.py` (tier C: PL only in the shipped column), `check_tier_c.py`.
- `finalize_stage.sh` (STAGE boxes = tiers B + C -> data/pl/READY_v1.flag; nfnet; sub_b0), `wait_stage.sh` (waits for a stage's teacher
  runs, then finalizes). `queue_lines_v1.txt`: all lines (tier A 14, tier B 28, tier C 49).

## Status

- 10-21 17:2x: v0 plumbing targets + proofs PASS (tier A x2, tier B x4). Teacher v1 f0-2 built; parity: Baker's fused3 = pkg961 <= 5e-6,
  Synovitis study-level form vs shipped cnx Spearman 0.986-0.988, MCL SD-2 = stored refit. f3-4: all columns build except Effusion /
  Synovitis (wait for the teacher fold runs).
- 10-21 17:2x: Master queued the 22 teacher fold lines behind each GPU's all-data tail and the 28 tier-B lines (box 1 seeds 201/203/205/207,
  box 2 202/204/206; gated on data/pl/READY_v1.flag); tier A goes into the oct14 queue after READY. Proofs accepted, no GPU smoke needed.
- 10-21 19:0x: USER OK'd cutting the low-value all-data tail (Master): queues = kept all-data -> teacher lines -> PL -> '# ===== LOW-VALUE
  ALL-DATA TAIL'. b3n_rep f3/f4 dropped (f3-4 bone teacher = the shipped 2-member fusion). Per-group READY: boxes (tiers B + C; needs only
  Contusion / Fracture / ACL / MCL / LOA, complete now) after the sxm verdict; nfnet after ef3* f3/f4; sub_b0 after the oct14 F3/F4 +
  cs_fc125 f3/f4. Tier C (z1_acl, w5b_mcl, b3g, sd1b, bsd_nf, oasep, oasep_nf; PL only in the shipped column) prepared + proven (v0).
- Teacher v1 partial assembled 19:1x: explicit-cell AUC on f3-4 in line with f0-2 (no broken f3-4 builder).
- Tier C caveat: earlier PL on MCL (P2 / P5) and LOA (P1c / P4 / P7 / P9) was negative on gold; it exists because the user asked for max PL seeds.
- 10-21 21:0x: sxm WON Contusion (+0.00013); PL teacher kept SXM=0 for every stage (Master's option (a): ~1e-4 LB teacher difference,
  tier B starts ~1 h earlier). Stage boxes PASS 8 / 8 (CSVs 21:04, READY_v1.flag 21:12; teacher re-assembly bit-identical). Waiters
  for nfnet / sub_b0 running (wait_nfnet.log / wait_sub_b0.log).

## v2 (USER 21:3x, supersedes tiers A / B / C; all v1 lines held / killed by Master)

- USER: "take OOF pseudo labels (predictions of teacher models, like in full package) then blend them with original labels (text- or
  image-derived, no difference), then train student models" for EVERY model of the package; R50 last; watch gold-58 plain vs PL per model and
  at package level (blend of the original and PL packages may be submitted later).
- targets_v2.py: all 28 final recipes (form effc_off). Text cells by report status (0.3 / 0.7 / 1.0); image reads 0.3; unread image
  cells = teacher (now trained). Inputs only through existing file keys (no trainer edits). Not covered: oct14 syn img_eff (no key; not
  shipped since EF3), bone location heads (no teacher column). oct14 basin lines carry syn_eff=0 (the blend already holds its values).
- check_v2.py (per recipe proof), finalize_v2.sh (s1 = no Synovitis / Effusion teacher needed; s2 = the rest), wait_v2.sh (s2 waiter),
  make_lines_v2.py -> queue_lines_v2.txt (133 box lines gated by READY_v2_s{1,2}.flag; 63 local lines inserted on READY; r50 last),
  pl_audit.py (v2 run audit), pl_gold.py (gold-58 plain vs PL: per recipe + linearised package + 50/50 blend; rough guide).
- P6 correction (USER asked): P6 was ONE Synovitis crop student on 1,144 uncertain reads, 2 folds, 'no gain detected' - not a loss and
  not the user's design. The closest test of the user's design = oct12 inner world (3x3 students, LB +0.001 not separable).
