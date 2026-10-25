# oct15_labels: label manipulations for the 0.961 fold package (SEALED 2026-10-16 ~13:05)

SEALED. Nothing runs here: the runner and post-watch are stopped, gpu_go.txt = 0, and the queue holds only commented (deferred /
pruned) lines. The GPU went to Lateral OA V3 at 07:37 via Master. No Kaggle submission, upload or kernel push was made from this
experiment.

Scope (user /goal 10-15 13:55, 24 h): xF fine-tune + R50 + labels for the full 0.961 fold package (submissions/oct13_f5_win). Renamed
~18:05 to a pure label study: label sets for the generalists (main effb0, R50, NFNet), single-rule tests, package swaps, and the
expert hand-off. Master assigned the GPU. Research mode: no submission candidates unless the user asks.

## Bottom line

No label change tested here robustly improves the 0.961 package. The deciding test was gold-58 and robust gold, with a jackknife,
against a pool of v1 runs. Nothing from this folder is ready to deploy.
- What holds:
  - Calibrated silent Synovitis improves TEXT-trained Synovitis columns (main, crop-expert text head: +75..+128 gold). The 0.961
    Synovitis operator G uses image heads trained on Codex reads, and gains nothing from it (0.961-like test, 2 folds).
  - Label changes cost other classes through the shared backbone: Baker's silence -> Effusion, PF 'unsettled' -> Lateral OA, OA / fluid
    label sets -> menisci. This is the main lesson for the expert sessions (HANDOFF.md lesson 7).
  - The x-mask stage (xF) on R50: a small Codex Effusion gain on 4 / 4 folds (models not trained toward Codex), weak-moderate evidence.
    The main xF stage is neutral inside 0.961 (window fusion / img_eff absorb it).
- What failed (each with its pre-registered read):
  - Label sets for the generalists, from scratch, 1 seed per fold:
    - vs the v1 pool, the 6 calibrated sets average gold9 +3.1 but robust gold +1.0 (f3-4); the 13 sets on f3 average ~0;
    - single runs at +5..+7 are ~1.5 sd of one-run noise;
    - consv3 is the only set positive on both golds on both folds, but its gold read is not clean (v3's prompt saw gold summaries).
  - Label STAGES (4-epoch fine-tunes on new targets): flat on independent reads. NFNet stages fail NFNet's own validation.
  - Package grid, all three generalists relabelled (f1, 7 cells): no cell passes. The best is the user's "3 different label sets"
    (main consnr / R50 vag15 / NFNet v4b): gold9 +3.2, holds under the jackknife, but only +2.3 over a seed draw.
  - Single rules on v1 (5 folds or a pre-registered stop):
    - Baker's silent 0.15 -> 0.036: the gain is ONE gold study (...761231739312); Effusion cost −10.4.
    - PF 'unsettled' -> 0.85: PF +4.9 ± 5.7, fragile and fitted-to; Lateral OA cost −13.8.
    - Fracture "stated-normal bone -> absent" (± fracture facts by assertion): +34.4 / −8.7 and +5.2 / +10.8 on f3 / f4, inside the ±20
      noise of a single run. The rules set's Fracture +56.6 is a single f3 run and stays unexplained.
  - Multi-head (the user's idea; one backbone, heads v1 / v4b / vag15): about as good as one label model. Averaging the heads adds
    nothing over the best head.
  - Stability sweep (bs 16, bs 32 lr x2 / x4, EMA 0.999, 12 epochs; seed pairs on f3): no recipe beats the deployed one. 12 epochs is 35%
    faster but loses gold.
- LB probes: the 0.944 replica package and a Baker's-column splice are built and gated locally (submissions/oct15_label_probes); nothing
  is uploaded. Forecasts on gold: a Baker's splice gives dLB ~+0.0014 only if the one-study gain were real (the user decides); a PF
  splice gives ~+0.0006 (not worth a slot).

## Learnings (methodology; VALIDATION.md has the full list)

1. Baseline = a pool of same-recipe v1 runs (here sub_b0, sub_b0_s1, tl_v1, tl_v1s1), never the deployed member alone. The deployed
   sub_b0 was a weak gold draw (−5.6 vs the pool on f3) and inflated "label sets are better members". Show the pool's own spread
   (one run ~4 on gold9, ~±20 on a single class) next to every delta.
2. Jackknife every gold delta (leave one gold study out). One study faked all Baker's gains, at member and package level.
3. A same seed does not pair runs here: GPU nondeterminism makes a same-seed rerun as different as a new seed (OOF corr 0.976 vs
   0.972). A one-fold "code drift" offset flipped sign on the next fold. Never conclude from one fold of 2-3 runs.
4. Gold-58 is the same 58 studies on every fold, so gold does not replicate across folds the way OOF reads do. Per-class gold noise is
   large (~20 per run for Fracture / Lateral OA). Decide on pooled or multi-fold reads, with pre-registered stop rules (they worked).
5. Read the classes whose labels did NOT change (backbone interference), in every single-class arm.
6. Reference bias: trusted / lookat report reads are built on v1 labels. Their silent negatives reward lowering silent targets
   one-sidedly (use *_ns). lookat labels some vague cells positive (e.g. PF 'unsettled'), so it is fitted-to for rules that raise them.
   Codex reads have a test-retest cap: Effusion is at the cap for every main and package, Synovitis only at G level. Codex is fitted-to
   for every Codex-calibrated set.
7. A target-level mechanism (per-status means) can mislead: v4b's silent targets are fitted per study. Check per study (gold_contrib.py)
   before claiming a mechanism.
8. Runner gotchas: free GPU memory is floored to whole GB, so desktop apps can block a 29 GB arm. A post-run dump racing a large arm makes
   the runner skip it. `pgrep -f` matches your own wait loop.

## Where things are

- HANDOFF.md: input for the per-expert sessions (OA / menisci, bone, Synovitis, new ACL / MCL and fluid experts). It has a
  "re-baseline, read first" note at the top, lessons 1-8 (7 = cross-class costs, 8 = the user's PL de-prioritisation), and per-expert
  findings.
- VALIDATION.md: living validation practices. REREAD.md: verdict table of earlier claims. ledger.md: every read in time order.
  user_directives.md: user directives (verbatim where given). ideas.md, LABEL_SETS.md, INTEGRATION.md: idea ranking, label-set
  pros / cons, package-integration caveats.
- Read scripts (scripts/):
  - rebase_read.py (sets vs the v1 pool + jackknife), drift_read.py (current vs old code);
  - rule_read.py (single-rule arm per fold; --pool, --compare, --watch neighbour classes), jk_audit.py (claims), gold_robust.py
    (contested cells, jackknife), gold_contrib.py (per-study contributions);
  - grid_pkg.py (package grid + gold9 jackknife), label_screen.py, mh_read.py, syn_g_read.py (Synovitis 0.961-like test),
    stability_read.py, retest.py (Codex test-retest), fx_split.py (Fracture rule split), build_targets.py (vsf / fx groups).
- Runs (runs/, ~30 GB): tl_* mains on label sets and single rules, r50_* / nf_* label arms, st_* stages, sx_fc125_v1syn (Synovitis crop
  expert, f0-1), rs_* stability sweep (f3), tl_v1 / tl_v1s1 (current-code v1 pair, f3-4). Targets: data/targets/<set>_f<k>.csv.
- Probe packages: submissions/oct15_label_probes (README there; local only).

## Open / deferred (not started)

- Label seed replicates (v4b s2 f3-4, vag15 s2 f4, consnr s2 f3-4): deferred by Master until after Lateral OA V3 / Synovitis V3
  (commented in queue.txt).
- The full Synovitis test (blended Codex x context image targets + a cross-fitted text / image head blend) belongs to Synovitis V3,
  which has the sx_fc125_v1syn files.
- Weak leads only: consv3 (gold not clean); the unexplained f3 rules Fracture +56.6.
- The Baker's LB probe and any upload / push need the user's explicit, per-item approval.
- Pseudo-labelling / self-distillation is de-prioritised in every session (user 10-16).

## History (before the seal)

Started 2026-10-15 ~13:50 as oct15_xF_finetune_plus_labels (user /goal, 24 h; scope then = xF fine-tune + R50 + labels). Renamed
~18:05 (user): the session is now FULLY focused on label manipulations and label-driven improvements: the from-scratch label screen,
component attribution per class, an 'ultimate' text-label set, propagation of the winners to R50 / NFNet, and the expert hand-off
(HANDOFF.md). Research mode (user 17:50): no submission candidates / gates / uploads until the user asks. Baseline = the 0.961 FOLD
package (submissions/oct13_f5_win). Master session (Master) distributes the GPU and owns the v3 lineage. Directives: user_directives.md;
results in time order: ledger.md; ideas and ranking: ideas.md; validation practices: VALIDATION.md; package-integration caveats:
INTEGRATION.md; expert hand-off: HANDOFF.md.

Concluded before the rename (xF stage work, ledger 15:08-16:45): main xF stage neutral inside 0.961 (its image-side gain is absorbed
by the window fusion / img_eff; drop-in loses through logit widening); R50 xF stage a small consistent positive (exact-path Codex
Effusion 4 / 4 folds), kept as a finding; label sets as 4-epoch STAGES flat on independent reads; NFNet stages fail (NFNet's own
validation drops). The label lever showed up FROM SCRATCH (tl_consnp: report reads above the refs band on 2 clean folds).

## What works, per class x per model (HISTORICAL, 10-15; member level vs the old 3-refs mean; x1e-3; folds in brackets)

> Superseded by the Bottom line above: these numbers are vs the 3-refs mean (incl. the weak sub_b0 draw) and were never jackknifed.

EVIDENCE STRENGTH (21:45, Fusion H3 noise ceiling): every main cell below is 1 seed x 1-2 folds (v4b: 1 seed x 5 folds). The same
set's deltas do not replicate f3 -> f4 (Spearman over 6 sets: tr9ns -0.03, menisci -0.31, OA -0.49, gold9 +0.20, text9 +0.54), and
with ~11 sets screened '2 / 2 folds above band' happens by chance for ~1 set. Read per-set / per-group rankings as WEAK until the
seed-2 replicates (v4b / vag15 / consnr f3-4) land. Robust so far: silence-only sets (nm05, sil) and txt / rules fail on several
reads at once; the calibrated-silent -> fluid / Synovitis gold + attention link repeats on every set and fold tested.

| class / group | main effb0 from scratch | R50 (stage xF) | NFNet | experts (hand-off only, user 20:05) |
|---|---|---|---|---|
| ligaments (ACL, MCL) | every calibrated set + on lookat (partly the one-sided silent reward, 20:50: v4b lookat +7.5 -> lookat_ns +4.1, trusted +1.6, gold +0.8 over 5 f); vag15 f3 lookat_ns / trusted / gold 2.4 / 3.2 / 3.2; consnr (no rules) trusted +2.5 vs consnp -1.5 (f3-4); package dilutes (3x3 mean) | xF: ACL -0.2, MCL text +4.7 (f3) | label stage fails (own validation drops); from scratch untested | no expert today -> NEW ACL / MCL crop expert (design note) |
| menisci (MM, LM) | only the v1-silence family (vag15 f3) positive on every read (lookat_ns / trusted / gold 2.1 / 2.6 / 4.3); v4b / consnp / consnr neutral (lookat +1.9 / +1.8 / +2.6 f3-4); silence-only sets, v5, rules hurt (-3..-5); LM fold noise +-10 (consnr LM f3 -5 -> f4 +15) | xF: LM gold +14 (f3) | - | OA experts' meniscus heads: targets near v1 / v4b |
| OA (MOA, LOA, PF) | v4b lookat +5.2 -> lookat_ns +3.4 / trusted +1.4 / gold +5.3 (5 f), consnp +4.1; vag15 f3 gold +21.9; PF: rules component hurts (ungraded chondropathy 0.30); removing it (consnr) fixes PF gold on 2 / 2 folds (+2.1 vs -23.0) | xF: LOA text +6 (f3), package gold-test LOA +7.7..+11.6 | - | OA experts: v4b-style MOA / PF targets, never rules PF |
| fluid (Effusion, Baker's) | Effusion: no independent gain under any set (Codex gains = reader agreement); Baker's + under every set; consnp fluid lookat +5.3 vs v4b -2.0; without rules (consnr) fluid lookat / trusted drop to 2.5 / 2.4 (f3-4; R-biased toward rules); label-free attention (non-FS mass Effusion / Baker's, seed noise 0.01 / 0.02): v1 0.194 / 0.216, v4b 0.144 / 0.184, v5 0.133 / 0.165, consv3 0.155 / 0.167, consnp 0.183 / 0.146, consnr 0.181 / 0.158; st_x stage 0.146 / 0.160 with no package gain (mechanism only); grp1 dropped 20:26 (gold fluid v4b +10.3 vs consnp +5.2 + attention favour v4b-style fluid) | xF: Codex Effusion + on 4 / 4 folds (exact path) | - | img_eff (synovitis crop experts) w 0.3; NEW fluid expert (design note) |
| Synovitis | calibrated sets: gold +79..+117 (member); as a 3rd G head no add | - | - | synovitis crop experts' text part on rules / consnp (rules Codex +44.6 independent) |
| Fracture / Contusion | rules: Fracture gold +47.7, Contusion text +12.7 (f3) | - | - | bone experts on rules / consnp F-C |

Mechanism of v4b's gain (vag15 = v4b's vague-cell calibration alone, f3; one fold): (a) the vague-cell calibration gives the report-read
gains on every group incl. menisci (vag15 tr9 +2.7, menisci trusted +2.6 vs v4b -8.3) and OA gold (+21.9); (b) v4b's calibrated silent
cells give the fluid / Synovitis gold (Baker's gold +11.5 vs vag15 -48.3, Synovitis Codex +62.6 vs -9.2) and the fat-sat attention shift
(Effusion / Baker's non-FS mass v4b 0.150 / 0.159, vag15 0.189 / 0.241, v1 0.197 / 0.207). Silence ~ normal for structural findings,
not absence for fluid / synovitis -> vsf (a everywhere, b only on Eff / Baker's / Syn / C / F) on f3-4.


REREAD (22:10, REREAD.md): every past conclusion re-read with image truth primary (gold-58 per group + Codex only where independent).
Validation note (20:50): report trusted / lookat references hold silent negatives but no silent positives, so they reward lowering
silent targets one-sidedly; primary reads for label arms = trusted_ns (silent negatives dropped) + gold9 + per group (VALIDATION.md).

## Question

Which fine-tune stages and label sets for the 0.961 package's generalist members (main effb0 sub_b0 f0-4, RadImageNet R50 F20 f0-4)
improve the full 0.961 package most? First cheap stage screens of all reasonable options, then from-scratch training of the most
promising diverse ones; deliver 5-fold members that drop into oct13_f5_win (build + gates; Kaggle submit only on the user's direct go).

Out of scope: NFNet stages (all hurt NFNet in oct12_pseudo), experts and synovitis image heads (later directions), PL teacher targets
(user ~14:04), all-data checkpoints (user ~14:00). Allowed labels (user ~14:01 / ~14:04): all text-based sets (v1, v3, oct14_text_labels
variants incl. consnp / consv3), all image-based reads (Codex Effusion / Synovitis, Codex Baker's; never the *_gold runs), combinations.

## Prior evidence (oct12_pseudo inner world: outer fold 3, 60% data, 3x3 without experts)

- xF = 4-epoch fine-tune of each trained checkpoint on explicit v1 cells only (meets / denied / structure_normal); Synovitis / Fracture /
  Contusion keep all cells. Main lr 1e-4 cosine; R50 lr 5e-5 / head 1e-4 const. B0 0.943 -> B0_xF 0.945 LB (+0.002, the inner
  world's most reliable gain). The all-cells control (same budget) is negative: the mask is what works. Stages widen logits ~1.3-1.45x.
- Closed there (do not repeat without a new reason): 'below' cells (xb), agreement filters, silent weight 0.25, teacher-confidence
  mask on R50, longer stages, R50 3x lr, every NFNet stage, uncertain-cell down-weighting from scratch.

## How 0.961 differs from that setting (-> new screens, ideas.md)

1. Experts own most columns: OA x3 / menisci = 0.4 3x3 + 0.6 OA experts; Fracture 0.1 / Contusion 0.0 3x3 (bone experts);
   Synovitis = G (image heads + crop experts). The generalists matter for ACL, MCL, Effusion, Baker's (fully) and 0.4 of OA / menisci.
   The inner-world xF gain sat in Effusion / Synovitis / Contusion / MCL -> expect less; screen st_x9 (stage only on the 9 owned columns).
2. Effusion = 3x3 window fusion + img_eff (Codex-trained) blend; Baker's = window fusion. Stage gains on Effusion overlap them.
   Image-read stages (st_x_cx, st_cx_consnp) read only on the package swap.
3. 80% data per model (not 60%): stronger inits, maybe a different best stage strength -> lr screens (3e-5 / 3e-4).
4. New label sets (consnp / consv3 / v5 / rules / txt / site / own; label-level image-ref mean 0.865 (v1) -> 0.898 (consnp) /
   0.902 (consv3)): the mask won because v1's non-explicit values are noisy; calibrated values may make those cells useful -> stages
   with label sets on all cells, x mask with label-set values, v1-explicit + label-set elsewhere.
5. Recipe constants (OA / F-C / img_eff z-alignment) were fitted on the deployed OOF; a 1.4x wider member shifts the 3x3 weights and
   the expert weight -> refit / spread-matched reads (info).
6. Window fusion (Effusion / Baker's) uses the members' per-window outputs: not in the local read (caveat; window dumps for finalists).

## Validation (pre-registered ~14:00, approved by Master ~14:03 with amendments)

- scripts/pkg_read.py main=RUN@E [r50=RUN@E]: package swap on folds 0-2 (pkg_lib 0.957 recipe + img_eff, no window fusion, fixed recipe
  constants = the drop-in) vs the deployed row; noise row = main -> sub_b0_s1 (full seed replicate). Identity swaps read exactly 0.
- Epochs fixed: main e3 (last of 4, 0-based), R50 e4 (1-based). Curves / median-of-best = info only (no selection on the read folds).
- xF arms PASS: tr9 (trusted mean on ACL, MCL, MM, LM, MOA, LOA, PF, Effusion, Baker's) > 0 on >= 2 / 3 folds and above the seed row;
  Codex Effusion, tr|Effusion, Codex Baker's each >= -1.5; gold-12 >= -2 (veto; any owned finding < -15 -> manual read); beats its
  all-cells control.
- Label arms PASS (Master): Codex Effusion > 0 on >= 2 / 3 folds AND above the seed row; tr|Effusion >= -1.5; no-damage guard tr9 >=
  -1.5; gold veto as above; better than st_x on the same reads. Codex Baker's secondary (~100 vague studies per fold; seed |max| 5).
  Report metrics are v1-derived (info); v5-family / consensus sets carry a weak 2nd-order dependency on fold K's v1 labels through the
  generalists' OOF ranks (report metrics may be slightly optimistic; Codex / gold unaffected); sets with v3 have an unclean gold read.
- Screens on fold 1 first; promoted arms run folds 0 / 2, then folds 3-4 (deliverable + untuned-fold confirmation, same sign).
- Discount best-of-N (~20 arms: +2..3e-3 gold by chance); decide on reports / Codex, gold = veto.

## Deploy form: spread-matched '_sm' checkpoints (adopted 15:10 after the W1 read; approved by Master)

A mechanism fix, not a tuned parameter: the x mask widens main's logits (Effusion / Baker's ~1.9x, OA 1.4x), which re-weights the
3x3 mean and breaks the z-aligned recipes (img_eff, OA, F-C) whose constants assume the old scale. scripts/spread_fix.py maps each
fold model per finding to the deployed member's mean / sd on that fold model's own OOF (label-free; a = sd_dep / sd_arm, b = mu_dep -
a mu_arm), baked into the checkpoint (main cls_w / cls_b, Basin bin_out Linear + reg_w; exact, window logits move with it). Folds 3-4
use their own OOF; constants are frozen in the checkpoint. Applied identically to every arm (R50 included); every arm is read drop-in
AND _sm. It adds no label dependency (provenance unchanged).

## Member-by-member gating (user 16:25 / 16:38)

A change that improves one member's signal in its class operator can be applied to every model feeding that class, but runs are spent
sequentially, with a read after each:
1. Fold 1, member 1 (cheapest or most likely): member signal (scripts/member_signal.py) + one-swap package read. Stop if flat / negative.
2. Fold 1, the other backbones one at a time: member read + cumulative fold-1 package read (1 -> 2 -> 3 members). Stop at the first
   member that does not hold (e.g. main + R50 xF was not additive).
3. A second fold for the members that passed (all-members package on an untuned fold).
4. Folds 3-4 only for deployment (they double as the untuned-fold confirmation of the single-member reads).
Each OOF study is scored by its own fold's 3 models, so a complete package read on one fold costs 3 runs, not 15.
Expert-only effects -> HANDOFF.md.

## Provenance of every arm's targets (user 10-15 via Master: report validation accordingly)

Reads: R = report-derived (trusted / lookat / text, tr|, tr9), C = Codex Effusion / Synovitis / Baker's, G = gold-12. Every arm read
carries a provenance line; verdicts rest on the reads INDEPENDENT for that arm; biased reads are shown, labelled 'fitted-to'.

| targets | R | C | G |
|---|---|---|---|
| v1 / st_x / st_all (v1 values, x mask or all cells) | own source (biased) | independent | independent |
| v5, v5ns, site, own, v4* | mild (explicit cells v1, re-ranked inside bands in v5) | fitted-to for Eff / Syn / Baker's (calibrated on training-fold Codex reads); other classes on generalists' OOF ranks (weak 2nd-order fold dependency, user: acceptable) | slight (v4b fix gold-motivated) |
| txt | mild (fitted on the reports' own explicit cells) | independent | independent |
| rules | independent (hand rules, no fitting) | independent | independent |
| consnp (v1 + v5 + txt + rules + site) | mild | fitted-to for Eff / Syn / Baker's (through v5 / site) | slight |
| consv3 (consnp + v3), v3 | mild | as consnp | NOT clean (v3's prompt saw gold-58 summaries): per-class gold = weak signal only |
| Codex-target arms (cx_*: st_x_cx, st_cx_consnp) | as their base set | fitted-to for Effusion / Synovitis / Baker's: judge those on tr| and G, never on C | as their base set |
| nan_sfc (st_x9), lr variants | as st_x | independent | independent |

Consequences: for Codex-calibrated sets a Codex Effusion gain counts only together with tr|Effusion >= -1.5 AND gold Effusion not
vetoed; for consv3 / v3 gold is not independent; every label arm also shows per-class rows for the classes whose labels it changed most.

## 10-min review routine (heartbeat Monitor; user 13:55 + 14:30)

1. Progress: runner events, finished arms -> pkg_read (+ eval_arm single-member rows); ledger entry per read, provenance line first.
2. Validation bias check (VALIDATION.md): for each new read, which references are fitted-to for that arm; look at the
   status-stratified reads (cxv / cxe), spread / refit variants, seed and stage-noise rows; revise VALIDATION.md when a read teaches
   something (log line).
3. Code check: new options / target files (target_check.py), configs of running arms (logs/*.log first lines), anything that could
   make an arm test something other than intended.
4. Re-rank ideas.md and queue.txt (ROI: underexplored first, cost, simplicity, evidence); prune arms whose siblings already decided
   them; add new ideas from the reads; consider a pivot.
5. GPU: keep it busy (slots / gb tokens), never start GPU work outside the runner without room.

## Layout

- main/ (deployed sub_b0 trainer + oct14_text_labels options: tgt_csv, ft_mask x, ft_keep, init, tgt_nan_w0; added here: tgt_partial,
  ft_tgt_keep, oof / gold written after every eval, last epoch always saved), basin/ (deployed NFNet / R50 trainer + init / ft_mask /
  ft_keep / tgt_csv), scripts/launch.py (arm = deployed source config + overrides), run_queue.sh + queue.txt + gpu_go.txt + slots_max.txt.
- weights/sub_b0, weights/r50_F20: the 0.961 package's checkpoints (md5-identical to submissions/oct13_f5_win/package). refs/: OOF /
  gold of the deployed members and seed replicates. data/: members.parquet + recipes (package rebuild), targets/ (label sets from
  oct14_text_labels + scripts/build_targets.py: nan_sfc, v1x_*, cx_*, h_v3, grp1; consnr / vag15 built by Master in oct14_text_labels, record copies in scripts/build_consnr.py, build_vag15.py), site / aux tables. cache -> v0/oct8_baseline_v2/cache (Basin data).
- scripts/pkg_read.py (package swap read), eval_arm.py (single-member read vs references), target_check.py (CPU target check per arm).
