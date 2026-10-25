# 5. Learnings: lineage, what worked, what didn't

Numbers are public-LB AUC (mean of 12 per-class AUCs) unless marked. Local numbers marked "x1e-3" are AUC differences on our
held-out reads. Full detail lives in the copied notes under `docs/source_notes/` (experiment READMEs, `WHAT_WORKED.md`,
`PROBES.md`, `lineage_oct16.md`).

## 5.1 Lineage

| step | package | LB | main lever |
|---|---|---|---|
| 0 | effb0 2.5D, 5 folds | 0.938 | baseline R2 (3-slice windows of every series, per-finding attention head, text v1 probabilities) |
| 1 | + Synovitis = logit mean with Effusion | 0.940 | Synovitis +0.024 AUC |
| 2 | + Synovitis from a head trained on Codex image reads | 0.944 | Synovitis +0.048 AUC |
| 3 | 3 backbones (effb0, NFNet-L0, RadImageNet R50) x folds 0-2 | 0.948 | cross-backbone ensemble (+0.008 vs 0.940, the largest single lever) |
| 4 | + step 2 on all 3 backbones | 0.952 | |
| 5 | + OA compartment crop experts (60 % share) + LOA de-coupling | 0.955 | OA summed AUC +0.036 |
| 6 | + Synovitis "G" (image heads + synovitis crop experts) | 0.955 (higher hidden) | |
| 7 | + menisci from the OA experts' meniscus heads + bone experts (Fracture / Contusion) | 0.957 | |
| 8 | + window / crop-level fusion + Effusion image-read head (w 0.3) + folds 3-4 | 0.961 | fusion ~+0.003; folds 3-4 ~0 |
| 9 | all data, 3-5 seeds per config | 0.962 | +0.001 |
| 10 | single-column LB wins on the fold package: LOA (silsep / per-series heads, Q15), Synovitis ConvNeXt crop slots, MCL (zoom member + SD heads + b3g + Q6), ACL (notch zoom + b3g), LM Q8 | fold stack v2: **0.964** | expert members + post-hoc de-couplings |
| 11 | + Contusion all-4 (Opus v3c targets), EF3 Effusion crop slots, Fracture v3c_fs, sxm | each above v2 by hidden order (0.964) | Claude Opus / Codex / Qwen image-read targets |
| 12 | the merged form retrained on all data (safe3, 81 models) | **0.965** | |
| 13 | + w5b 9 seeds and extra seeds (P1, 93) / + pseudo-label twins (P3, 117) | **0.965** (P1 above safe3 and P3 by hidden order) | finals: P1 + P3 |

The lever split up to 0.961 (lineage_oct16):
- backbone ensemble ~+0.008;
- image-read labels ~+0.006;
- experts ~+0.006;
- fusion ~+0.003;
- all data +0.001.

## 5.2 What worked, by class

| column | what carried the gain (LB status) | mechanism we could show |
|---|---|---|
| Synovitis | image heads trained on Codex image reads (+0.048 AUC); synovitis crop experts (pouch / Hoffa / posterior recess); ConvNeXt-T in the effb0 crop slot | reports say little about synovitis; the image reads are a much better target |
| Effusion | 3x3 window fusion; image-read Effusion head at w 0.3; EF3 crop slots (Codex + Qwen + Opus target, ~+0.00015 LB) | the base (70 % of the column) carries most of the label gain |
| OA (M / L / PF) | compartment crop experts (localiser bags, v2cor coronal-by-tibial-cartilage) at 60 % share + de-coupling | the crop isolates the compartment |
| Lateral OA | silsep = per-series heads + calibrated v4b labels (silent cells low); oasep / oasep_nf groups (Q15) | LOA follows GOLD: wins come through report-invisible positives |
| MCL | coronal 48 mm MCL zoom member (w5b_mcl); per-series-group heads (sd1b, bsd_nf); b3g area MCL; Q6 de-coupling from Contusion | an area-only view isolates the ligament; reports couple MCL to bone bruises that the image labels don't |
| ACL | 40 mm notch zoom member (z1_acl); b3g area ACL | the notch crop sees the ligament without the rest of the knee |
| Lateral meniscus | OA experts' meniscus heads; Q8 de-coupling from ACL (b 0.1) | ACL-torn knees inflate LM scores (+0.9 logit on LM negatives) |
| Medial meniscus | OA experts' meniscus heads (fusion) | misses = peripheral / ramp / small tears shared by every member |
| Contusion | bone crop experts; FS-only member; all-4 members on Opus v3c targets; sxm (masked Opus image heads) at w 0.3 | the label effect: every v3c member beats its old-target seed twin |
| Fracture | the same bone members (v3c) + FS-only member at w 0.75 | local 5-fold cross-fit +13.7 x1e-3 |
| Baker's | 3x3 window fusion | |

Cross-class lessons:
1. **Area-restricted members** (a crop that shows only the structure) carry the ligament gains; a merged multi-area member (b3g)
   adds on top. Whole-knee generalists still anchor every column. A crop de-couples a class only if it EXCLUDES the other class's
   evidence.
2. **Post-hoc de-coupling of report co-occurrence** (MCL <- Contusion, LM <- ACL) is the cheapest gain: inference-only and
   LB-proven. Re-fit its constants whenever the context column changes.
3. **Better training targets** beat model changes:
   - image reads where the reports are vague (Synovitis, Effusion, Contusion, Fracture);
   - calibrated text where the column follows gold (Lateral OA).
4. **Cross-backbone ensembles**, not seeds. One all-data model per backbone ~ 3-4 fold models. Seeds beat folds at equal N, but
   the gap shrinks with N. Fold models decorrelate more (Spearman 0.904 vs 0.941).
5. **Fusion at window / crop level** (geo / prob mean over windows or crops with a top-k relevance gate) helped Effusion, Baker's
   and the experts. It is mostly filtering. Beyond that, post-hoc fusion is saturated.
6. **Final assembly must re-fit coupled terms on the final members**: OA de-coupling k, img_eff weight, Q6 / Q8 contexts,
   crop-fusion aligns.

## 5.3 What did not work (with the test that closed it)

- **Pseudo-labels**: ~11 failed attempts on gold / LB.
  - The inner-world 3x3 students gave +0.001 LB, not separable from noise.
  - The final PL v2 twins tied plain at 3 decimals; they lowered Lateral OA and bone on gold and raised MCL / PF / Effusion.
- **TTA** (plain + 4 affine views): 0.940 = 0.940.
- **Folds 3-4** on top of 0-2: ~0 for ~60 min of runtime.
- **Resolution / magnification / native-resolution caches**: scans are acquired at ~0.6 mm. The 0.4 mm cache already oversamples
  (PixelSpacing is the interpolated grid, not detail).
- **Backbone sweep**:
  - ConvNeXt-T main beat effb0 by +3.8 x1e-3 alone but added +0.2 inside the package;
  - DINOv2-S, RadImageNet DenseNet-121 and natural-image ViTs were weaker;
  - generalist gains get diluted by the expert-owned columns.
- **VLM classifiers** (Qwen 2B fine-tune):
  - weaker than the CNNs;
  - an Effusion member added < 1e-4 macro;
  - 17-105 min of Kaggle time per fold;
  - closed.
- **Feature-level KD** from fold teachers: fold feature spaces are unrelated (cross-fold similarity 0.04-0.32).
- **Text label sets beyond v1** (v2 facts, v3 forecasts, v5 fact heads, label-diverse mains, metadata / site post-processing): no
  robust package gain, except v4b for Lateral OA.
- **External OAI data** (MOAKS reads, 2,399 knees):
  - no column held on two folds;
  - DESS gradient-echo windows don't match TSE;
  - no final uses OAI.
- **Single-column LB losses**:
  - FS-only Synovitis image heads;
  - Effusion form (c) (main / NFNet base slots on EF3);
  - Baker's pairwise-site targets (Q25, -0.001); the bk1 Baker's splice was flat;
  - PF from the separate-head OA fusion (Q20, not LB-confirmed);
  - LOA k_eval 64 on all-data (sign flipped vs the fold package).
- **Image-only VLM reads of focal findings** (MCL, menisci, cartilage): near chance.
- **Opus synovitis second reads**: lowered the Codex target on gold.
- **Segmentation masks / mask-guided crops, bone SD heads, ConvNeXt bone member, a 2nd FS bone member, oversampling, label doses**:
  no gain detected on their reads.

## 5.4 Validation lessons

- **Metrics** (`code/common/metrics.py`):
  - `trusted` = AUC on hand-checked text cells (primary early on);
  - `lookat` = trusted + "look at" cells;
  - `text` = v1 meets vs denied;
  - `gold` = the 58 official image-labelled studies;
  - per-class image-read yardsticks later (`03_LABELLING.md` 3.6).
- **Gold-58 is the right direction but very noisy.**
  - Robust positives per column are only 9-35.
  - Seed-only swaps move the 12-column gold mean by +-0.0013.
  - The best of ~12 near-null arms shows +2-3e-3 on gold by chance.
  - Report a leave-one-study-out range, hand-read the movers, and never pick among many arms by gold alone.
- **Compare against same-recipe seed replicates**, not other configs. Per-run sd (oct7 main): trusted 0.0014, gold 0.0032, lookat
  0.0034 across seeds. Rare findings have ~4x more seed noise than a 2-run row suggests.
- **Read a new label or member against its same-recipe old-target twin** (the "twin row"). It separates the label effect from
  diversity.
- **One fold per arm, then re-rank**: deepen to more folds only for leads. Judge on folds that never picked the form (winner's
  curse).
- **LB probes**: change one column on a fixed base and read by hidden order.
  - Per-column gains add.
  - Probes must be apples to apples: fold members on the fold package, all-data members on the all-data package.
  - The LB is the final judge. Several local readers (text, trusted, Codex) were wrong in both directions.
- **Public LB noise**: a model redraw moves public by +-0.0004-0.0005, the same size as the all-data gain. Don't read differences
  below that between differently seeded packages.

## 5.5 Process lessons

- **Gates before every Kaggle action.** The fold kits, the merged v2 and the all-data finals each passed:
  - local reproduction;
  - strict state-dict loads;
  - a CPU setup test on the real datasets;
  - a Kaggle commit check with shared parts bit-identical to the reference;
  - local vs Kaggle <= 1e-3.

  This caught real bugs: stripped training-only heads, a Basin checkpoint exported but never run, file-listing pagination, a
  routing-overlay harness bug.
- **Fail loudly.** Strict mode with no silent fallbacks: a broken package must error in the commit check, not score quietly.
- **Runtime is a deadline risk, not a budget line.**
  - Estimate from scored runs, keep 2x margin, and submit finals early.
  - A deadline-day machine timed out an 87-model package that a fast machine runs in ~180 min (section 4.4).
- **Seeds for the finals**:
  - rank extra seeds by noise removed per Kaggle minute (worst-case seed-noise table);
  - one recipe (w5b_mcl) was much noisier than the rest (inter-seed corr 0.84 vs 0.92 between folds) and deserved 9 seeds;
  - the whole extra-seed tail was worth < 1e-4 LB.
- **Many parallel agents** (Claude Code sessions: Master, Experts, labelling chats, investigators) worked well. They need:
  - one GPU owner per card with a queue + `gpu_go` gate;
  - one place for decisions (ledgers);
  - every irreversible action (submit, delete, push) as its own short synchronous call, never detached inside a longer command.
- **Machine hygiene**:
  - cap pinned host memory (an unkillable pinned allocation crashed the desktop);
  - read big caches from JPEG copies (RAM thrash);
  - hardlink duplicate checkpoints;
  - never `pkill -f` a pattern (it hits other sessions); stop runners by their PID file.

## 5.6 If we did it again

1. Build the cross-backbone 3x3 and the image-read Synovitis target in week 1. They were the biggest early levers.
2. Spend labelling budget on image reads for the classes whose reports are vague (synovitis, effusion, contusion, fracture), with
   gold-checked yardsticks, before tuning models.
3. Add area-restricted expert members per structure, plus post-hoc de-couplings of report co-occurrence.
4. Keep one fold-average package as the LB base, probe one column at a time, and retrain on all data only at the end (with a
   noisy-recipe seed plan).
5. Treat Kaggle runtime as variable by 2x or more between machines, and submit the finals a full day early.
