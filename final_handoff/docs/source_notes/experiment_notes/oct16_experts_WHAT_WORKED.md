# What carried the gains, per column (Experts V3, 10-18 ~12:45; for U65 'leaner but stronger regime')

Written for Master / Brainstorm's request 15 (the final all-data plan). Each line: the lever that carried an LB-proven or locally
supported gain, the mechanism we can show, and what did NOT add (with the test). Evidence and numbers: ledger.md (read ids), PROBES.md.
Base = the 0.961 fold package (oct13_f5_win: 3x3 generalists main / NFNet / R50 + crop-fused experts + window fusion).

| Column | What carried the gain (LB status) | Why it worked (shown mechanism) | Did not add (test) |
|---|---|---|---|
| MCL | W5b coronal 48 mm MCL zoom member (fold LB 0.962); Q6 post-hoc MCL <- Contusion de-coupling (s5 LB 0.963); zoom + Q6 stack 57325143 above the zoom probe; Q21 = + SD-2 generalist series-group heads + b3g's area MCL (submitted 57352127) | an area-only view isolates the ligament; reports couple MCL to bone bruises that gold does not -> removing the coupling ranks on the ligament itself | axial zoom inside a shared trunk (damages it); MCL-only expert; ConvNeXt zoom (overfits); pseudo-labels; masks; more MCL arms after Q21 (gold ~saturated, Brainstorm 6) |
| ACL | Z1 notch 40 mm zoom member (fold LB above the base); Q16 = Z1 + b3g area ACL (57346196: LB above Z1 by hidden order, est +0.00011 = the FINAL ACL form; first LB support for b3g) | the notch crop sees the ligament without the rest of the knee | b3g variants (v4b targets / split trunks: no headroom, ACL at ceiling, Brainstorm 11) |
| Lateral OA | silL = the v4b calibrated-label 3rd group (LB-won); Q15 = per-series SD heads in every LOA group + silsep (submitted 57352132); oadual (FS / non-FS separate trunks) = local lead (f1 pass, f2 running) | LOA follows GOLD: the wins come through report-invisible positives (calibrated labels / per-series heads rank them up); per-series heads also de-couple LOA from the lateral meniscus | per-class LOA-only experts and LOA-only crops (need the co-supervision of the other OA / meniscus outputs); k_eval 64 (won on the fold package, ~0 on all-data); px 384 / 0.25 mm cache; multi-zoom; pseudo-labels (P4, P9); BML aux (text below a seed); meniscus masks |
| Lateral Men. | Q8 post-hoc LM <- ACL de-coupling (s5 LB 0.963; b 0.1 = the gold optimum) | ACL-torn knees inflate LM scores (+0.9 logit on LM negatives); the coupling is in the image members too and is not the notch in view (lmfe) | magnification / native resolution (small tears do not move); no-notch LM crop (weaker); root crops; LM label / coupling work (Q9 family, DC1) |
| Medial Men. | (deployed fusion; no LB-proven addition) | MOA cue is GENUINE co-occurrence on gold; misses = peripheral / ramp / small tears, shared by every member | px-384 MM crop members (any seed adds the same small gold, = averaging; one small-tear study carries it); seg-pretrained init, os16, dual trunks, magnification, 0.25 mm cache; post-hoc MM <- MOA (one study) |
| PF OA | Q20 = per-series SD heads in both OA groups (weak, ungated LB) | (small text gain; gold flat) | zoom / resolution (acquisition limit); sharpened vague-cell labels (leaky gain, text down); trochlea bag (redundant: the pf crop already holds it); lower de-coupling k (robust gold down) |
| Medial OA | (saturated on gold) | moderate / partial-thickness lesions on the OA grade boundary | all levers closed; MOA k 0.1 = an inference-only tweak (Master) |
| Synovitis | ConvNeXt crop slots in G (fold LB above 0.961) | G already beats a careful Codex re-read on gold; errors sit in moderate-effusion knees | FS-only image heads (LB lost); post-hoc Syn <- Eff; other VLM readers; model variants (supervision-limited; BS-4 second reads CLOSED by the user 10-18: no Codex quota); masked crops (read 113: Codex worse) |
| Effusion | window fusion (in 0.961) | fusion part alone = the column | img_eff w stays 0.3 (mixed: gold / reports favour 0.2, Codex + Opus image reads favour >= 0.3; only an LB read changes it); no GPU arm |
| Contusion | FS-only bone member in the fused column (gated probe) | small isolated bruises are the only model-side misses | ConvNeXt bone member (fold 2 null); bone SD heads; mes splice; blanket C <- F |
| Fracture | (none beyond the package) | the oedema link shows as false positives, not missed fractures | impaction-only convention target (not learnable from ~131 cells); oedema-free reader; post-hoc F <- C |
| Baker's | (not worked on here) | - | site offsets = Master |

Cross-class lessons for a leaner regime:
1. Area-restricted members (a crop that shows only the structure) carry the ligament gains (MCL, ACL). A merged multi-area expert (b3g)
   adds on top. Whole-knee generalists still anchor every column.
2. Post-hoc de-coupling of report co-occurrence (Q6 MCL <- Contusion, Q8 LM <- ACL) is the cheapest gain: inference-only, LB-proven.
3. Label calibration (v4b) helped only where the host follows gold over reports (LOA). Pseudo-labels failed 11 times.
4. Resolution / magnification never helped: the scans are acquired at ~0.6 mm, the 0.4 mm cache already oversamples.
5. Per-series (FS / non-FS) heads, and now separate trunks, help OA through report-invisible positives; they did not help menisci.
6. Single-fold reads need a same-recipe seed twin: two seeds of one MM crop differ as much as most 'arms' (mmhr vs mmhr_s1).
7. Final assembly must re-fit the coupled terms on the final members: OA de-coupling k per column, img_eff w, Q6 / Q8 contexts, crop-
   fusion aligns (PROBES.md 'FINAL-ASSEMBLY RE-FIT RULES').

## 10-21 (12:41): the image-label (Opus / Codex / EF3) round - what carried the gains (yardstick per class = training target + referee)

| Column | What carried the gain (LB status) | Why (shown) | Did not add (test) |
|---|---|---|---|
| Contusion | all-4 v3c member swap (b3c / b3l / b3n / bfs_rep on Opus v3c where read, old target elsewhere): LB-best, hidden order above v2 (~+0.0005 band); b_v3 slot alone ~+0.0003. sxm (ConvNeXt syn-form, masked Opus img heads) as an ADDED member at w 0.3: submitted 10-21 (local cross-fit +2.2 (1.4) over all-4, twin -0.2) | the label effect: every v3c member beats its old-target seed twin; the masked img head carries the label signal at ~32 % coverage | v3e labels over v3c (all-4 f1-2 +0.6 (1.9)); a 2nd FS member (bfsn_rep, bfsl_rep); oversampling (b3e_pp); masked heads on the bone expert (b3m_v3f); cont_fs w re-fit (picks 0.2 vs shipped 0.25, +0.1) |
| Fracture | v3c bone members + the FS member bfs_rep spliced at w 0.75 (submitted 57475191, scoring) | local 5-fold cross-fit +13.7 (3.8); FS member positive on 2 seeds (seed 1's f1 was a high draw) | SIF bone members; a 2nd FS member; sxm added on the probe form (~0 on f1 / f2) |
| Effusion | EF3 crop img_eff slots (LB-won, ~+0.00015); form (c) = + the main and NFNet base slots on EF3, R50 shipped (submitted 57479153, scoring) | the base (70 % of the column) carries most of the label gain: step 2 +10..+24 vs the crop-only +1.6; seed-robust in both slots | form (b) (+ the R50 slot): stronger locally but a systematic gold drop (survives a seed swap; the EF3 class signal of the Basin slots, not window selection) |
| Synovitis | - (the shipped crop slots already train Codex img_syn heads; 100 % of the column) | - | single-head Codex form (tie) |
| Baker's | - (HELD) | Codex over-calls report-negative studies vs Opus image+report re-reads (Master's check) | bk1 main-recipe splice (LB-flat); Codex-led Q25 |
| PF OA | - (HELD at v3g, 1231 reads) | the effb0 OA member fails f0 at every label count (-12.5 / -10.4) | OA effb0 / NFNet members, the syn-form PF head (as a member: +1.0 (2.0) vs twin -0.7) |

Lessons (10-21):
1. The USER's standard recipe (sample as before; the new label where read, the member's old target elsewhere) beat every variation on
   bone experts (oversampling, doses, masked heads); masked img heads help on a NEW syn-form member at low coverage (sxm).
2. Read each arm vs its same-recipe old-target twin (step 1), then in the shipped form (step 2), then cross-fit (step 3); the twin row
   separates the label effect from member diversity (sxm: +2.2 vs twin -0.2; PF sxpg: dense gain = twin's = diversity).
3. Sparse yardsticks (~200 Opus reads / fold) carry ~+-10 seed noise on single-fold reads; quote pooled folds. EF3-size yardsticks are stable.
4. Kit hooks need a code-path mini-run of the package's own stage (an exported-but-not-run Basin ckpt errored eff_c v1 on Kaggle);
   one-column Contusion probes move MCL through the post-hoc DEC (tiny, +0.4e-3 MCL AUC) - gates check MCL via decouple().
