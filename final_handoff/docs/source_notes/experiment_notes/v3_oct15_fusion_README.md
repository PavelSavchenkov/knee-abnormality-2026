# oct15_fusion: post-train fusion of model predictions (window / series / head level)

Goal (user 2026-10-15 ~19:35, 5 h, CPU only; GPU stays with the v3 master's sessions): improve post-train fusion of the package's
models at window / series / head level, per class or class group (anything identifiable at inference, no report text). Extend the
oct13 window-fusion idea (experiments/v2/oct13_deep_ensembling_test: geo|pm|g20|wmean on Effusion / Baker's, crop-level fusion in the
OA / bone / synovitis experts; part of the 0.961 and 0.962 packages) in every reasonable direction: attention weights, per-series
fitting, rankings, per-model weights at several coarseness levels, filtering vs fusion, why it works, lightweight cross-fitted
adapters / heads; >= 6 diverse ways; evidence from prior experiments; a GPU plan for later.

Research mode (user 10-15 17:50): no submission candidates, gates or uploads.

## SEALED 2026-10-16 ~07:00 - read this block first

Status: closed (user 10-16 ~07:00: "seal the experiments folder"). Nothing runs, nothing is open. No submissions were made (research
mode). The final word is this block plus sections 15 (jackknife), 16 (ranked learnings) and 16a (correction against the full seed
pool). Sections 1-14 below are the time-ordered record; some statements there were later corrected - see "Superseded" below.
Every number is in ledger.md. x1e-3 AUC; "gold" = gold-58, "robust" = gold with contested cells dropped; "Codex" = Codex image reads.

### What happened here

- 10-15 19:36-00:35 (user's 5-h brief, CPU): post-train fusion of the 0.961 / 0.962 package members at window / series / head /
  package level: decomposition of the oct13 window fusion, >= 14 fusion ways, mechanism, light cross-fitted heads (sections 1-9).
- 10-15 21:05-10-16 07:00 (user requests relayed by the v3 Master): labels and the shared backbone (head-level label screens, a
  partial-backbone screen, backbone x head grid), label-diverse vs seed ensembles per class / group, gold study by study, the
  jackknife standard, re-reads as Labels Work's seed / multi-head arms landed (sections 10-16a). Worked with the Master and Labels Work
  (experiments/v3/oct15_labels) by message; notes sent to Lateral OA V3 and Synovitis V3.
- GPU: only short guest blocks granted by the Master (feature / window dumps G1 / G1b / G4, two partial-backbone prototypes). The
  partial-backbone screen batch was dropped by the Master (01:00) when the GPU went to the new class sessions.

### Final learnings, ranked by evidence (details and numbers: section 16)

Solid:
1. The deployed window fusion gains by FILTERING (Basin's window set: no axial non-fat-sat, plus the top-k gate), not by fusing models.
2. Post-hoc fusion inside the package is saturated: >= 14 ways move it by <= 0.3; report-fitted vendor offsets and cross-class
   stacking hurt.
3. Gating: opt in per class on a coarse grid (Effusion k10, Baker's k20, Medial OA k10); global gates and learned heads do no better.
4. Image-target models (synovitis heads, crop experts) gain nothing from post-hoc gating; window-set levers must act in training.
5. Crop fusions hold on robust gold: OA (Lateral OA robust +27.7, PF OA +13) and Fracture (+16.7); Contusion is undecidable.
Moderate:
6. Mechanism: report-trained mains read a report-specific signal from non-fat-sat series; Codex-calibrated label sets move attention
   away from it.
7. Label effects live in the shared backbone features; head-level label screens fail validation; single-seed label deltas do not
   replicate across folds.
8. Against a 4-member v1 seed pool (both folds), the AVERAGE label set equals v1; v4b + vag15 beat it (all9 +5.2 / +2.9 robust, not
   fragile). Label sets robustly gain Medial OA and Synovitis and robustly lose menisci; the meniscus loss comes from OTHER classes'
   targets through the backbone (v1 meniscus targets do not protect).
9. Mains sharing a seed make correlated errors even with different labels: give each label set its own seed (3 seed contrasts agree).
   Window attention is interchangeable between mains; the label effect is in the window values.
10. One backbone with v1 + label heads (tl_mh3) buys about a 3-seed ensemble's macro gain at one model's cost; the same label sets as
    separate models are clearly better (menisci).
11. In the bare 3x3 a label-set main's edge mostly dilutes (ligaments wash out). Its lower correlation with Basin on vague Effusion
    cells is real (label-free), but the gold gain there hangs on one cell.
Weak / fragile (do not build on): per-class label patterns other than Medial OA / Synovitis; every Baker's label gain (one gold cell:
report general_normal, all readers ~0.1, gold 1); a Lateral-OA-specific gate; protocol-wise recipe weights.

### Superseded statements in sections 1-14 (do not reuse)

- "Label-diverse ensembles beat seed ensembles / the edge is better members / label ensembles keep gaining while seeds saturate"
  (sections 14, early summary): measured against the deployed sub_b0, a weak gold draw. Against the full seed pool: no general edge,
  seeds gain as much from averaging (16a).
- "Head-only re-fit on v4b targets recovers most of the from-scratch gain" (H2): mostly Synovitis; the head screen failed validation.
- Fragile under the jackknife (section 15): label-set PF OA loss and ACL gain, Effusion gain inside the 3x3, Lateral OA member-level
  gate, all Baker's gains.

### Evaluation rules that came out of this session (now used across the v3 lineage)

- Baseline = a multi-seed v1 pool from the SAME code as the arms, never one deployed seed (old-code mains only for gold reads:
  current code reads ~2.5 lower on reports at the same seed).
- Compare ensembles at equal N against seed ensembles of that pool; split any edge into members vs diversity.
- Every gold delta: with and without contested cells (scripts/gold_contested.py) AND the leave-one-gold-study-out jackknife (FRAGILE =
  sign flip or halving; scripts/h13_jackknife.py). Report reads are agreement with v1 and are biased toward v1-trained models.

### Hand-offs and ideas not done

- Lateral OA V3 and Synovitis V3 got 'what Fusion knows' notes (+ a correction withdrawing the Lateral OA gate claim). G3, a
  fat-sat-only main (fs_only flag in Labels Work's trainer; synovitis crops fat-sat-only +4.3 Codex), went to Synovitis V3, parked
  until the user OKs it.
- For later GPU time: label-set mains each with their own seed, picked by their own gold edge over a multi-seed v1 pool; meniscus
  protection by a meniscus-only member or a model whose other targets are v1. Low ROI / dropped: partial-backbone label screen (6-pair
  validation too weak), package main slot = v4b + vag15 (3x3 edge +1.1 robust), type-prior main.

### Where things are

- README sections: 1-9 fusion (decomposition, mechanism, saturation, knobs, transfer, >= 6 ways table, head adapters, prior evidence,
  reuse + GPU plan); 10-12 labels and backbone, REREAD, gold study by study; 13 handoff; 14 label vs seed ensembles; 15 jackknife;
  16 / 16a ranked learnings and correction. ledger.md = time-ordered log with every number.
- data/ (6.9 GB): copy of oct13 data (window / crop dumps, members.parquet, recipes), win_v3 / win_h dumps (G1 / G1b / G4), targets.
  weights/: sub_b0, sub_b0_s1, effb0_lr1e-3_R f3-4 copies. main_pb/ + refs_pb/: partial-backbone trainer copy; runs/pb_v1 / pb_v4b f3
  prototypes. out/: metrics parquets, *.md reports (h8_perclass, h9_perclass, h10_perclass, h12b_jackknife, h13_jackknife,
  h13b_jackknife_claims, gf_*), cache/.
- Key scripts (all `conda run -n rsna python experiments/v3/oct15_fusion/scripts/<name>`): wdata.py (caches), d1_decompose.py (fusion
  decomposition), d3_coarse.py (gate coarseness), p1_package.py (exact package map), gold_contested.py / gold_flips.py (robust gold,
  study-by-study flips), h8_labelens.py + h8_report.py + h8_seedpool.py (label vs seed ensembles), h8_seedxlabel.py (seed sharing),
  h9_labelwin.py / h9_attswap.py (attention x values), h10_3x3ens.py / h10_perclass.py / h10_effusion.py (inside the 3x3),
  h12_multihead.py / h12b_jackknife.py (multi-head), h13_jackknife.py / h13b_jackknife_claims.py / h13c_drop_bakers.py (jackknife).
  Imports resolve inside scripts/ (lib.py, fz.py are local copies); Labels Work's runs are read from experiments/v3/oct15_labels.

## Data and evaluation

- data/: copy of oct13_deep_ensembling_test/data (window dumps of the deployed 3x3 members: main sub_b0 on all windows, NFNet / R50
  Basin T_eval 24 mapped onto main's windows; crop dumps of the OA / bone / synovitis experts; members.parquet = fold-level logits
  of every member; recipes).
- scripts/wdata.py -> out/cache/win_f{0-4}.npz (OOF: one model per backbone per study, 4,349 studies, 657k windows) and win_gold.npz
  (58 studies x 5 fold models x 3 backbones), with per-window series metadata (plane, type, fat-sat, position).
- scripts/fz.py: vectorised segment softmax / top-k / pooling. lib.py, make_eff_recipe.py, wfuse.py: copies (package rebuild).
- References: Codex image reads (Effusion all 3,6k / report-vague cells, Synovitis, Baker's 516 report-vague) = image truth proxy at
  scale; report metrics (trusted / lookat / text per finding) = floor; gold-58 = small veto. Every fitted knob is cross-fitted
  leave-one-fold-out (LOFO) over the 5 OOF folds. Fold consistency reported as n+ / 5.
- Package relevance: ACL, MCL, Effusion (+ img_eff recipe), Baker's are 3x3-only columns; OA / menisci take 0.4 of the 3x3, Fracture
  0.1, Contusion / Synovitis 0.

## Directions (pre-registered 19:55)

| # | direction | question |
|---|---|---|
| D1 | decomposition | gain from cross-model fusion, or from attention filtering, or from probability pooling? |
| D2 | mechanism | why does it gain: which windows / planes are selected, which cells move (report status strata), agreement |
| D3 | pooling family + coarseness | continuous knobs (gate k, temperature, blend a) fitted globally / per group / per class / per class x backbone |
| D4 | series / plane level | two-level pooling (within series, then across series), per-plane weights per finding |
| D5 | rankings | rank-consensus relevance, rank-normalised values, study-level rank blends |
| D6 | learned light heads | cross-fitted MIL meta-head on per-window outputs of the 3 backbones; study-level stacker on window summaries |
| D7 | cross-class | relevance shared inside class groups (fluid, bone, OA / meniscus); cross-class stacking |
| D8 | per-study adaptive weights | backbone weights from agreement / attention concentration / Basin coverage |
| D9 | transfer | 1 model per backbone (OOF) vs 5 (gold) vs all-data seeds: do fitted knobs transfer? |
| G | GPU plan | what to train once the GPU is allowed |

Added during the session: E1 (expert crop fusion: fusion vs filtering), P1-P5 (package-level reads, groups), S1 / S2 (label-changed
mains, Master's request), D10 (context partial-out, mechanism), H1 / H1b / H1c (head-level adapters on frozen features), G1 / G2 (guest
GPU block for feature dumps, image-read head on the deployed members).

## Results (time-ordered record; final word = the SEALED block and sections 15-16a)

Time-ordered log with every number: ledger.md. x1e-3 AUC deltas. "Codex" = Codex image reads (Effusion all / report-vague cells,
Baker's 516 report-vague studies, Synovitis); report = trusted / lookat / text cells (v1); gold = gold-58. OOF = one fold model per
backbone per study, 5 folds unless stated. Evidence strength in brackets.

### 1. Where the window-fusion gain comes from (D1, D2, D4, E1)

Bare 3x3 (main sub_b0, NFNet, R50), deployed fusion geo|pm|g20 vs the deployed study mean, OOF 5 folds:

| variant (each vs the deployed study mean) | Codex Effusion | Codex Baker's | report lookat |
|---|---|---|---|
| probability instead of logit pooling, own attention, all windows | +1.5 (5/5) | +2.9 (5/5) | -0.0 |
| own attention + top-20 gate (filtering only, all windows) | +1.5 (5/5) | +1.3 | +0.2 |
| own attention restricted to the windows Basin sees, no gate ('ownmap') | +3.2 (5/5) | +6.6 (5/5) | -0.9 |
| ownmap + top-20 gate | +6.1 (5/5) | +8.1 (5/5) | -0.75 |
| consensus attention (geo of the 3 backbones) + top-20 | +6.2 (5/5) | +9.7 (5/5) | -1.3 |
| deployed geo|pm|g20 (consensus + prob-mean values) | +6.1 (5/5) | +9.1 (5/5) | -1.3 |

- Value fusion (prob vs logit pooling, z-scored / quantile values, rank blends) adds nothing on the 3x3 [solid: D1, D5].
- About half of the Effusion gain (70% of Baker's) is WINDOW-SET filtering. Basin uses only its slot series (sagittal, coronal,
  axial PD / T2 fat-sat; not axial non-FS / GRE, coronal T2 non-FS, duplicate series) and 24 slices per series. Off-map windows get
  s = -30 in the dumps, so the deployed 'consensus' attention silently restricts main to Basin's series. The top-k attention gate is
  the other half. Consensus across backbones adds ~0 on Effusion and ~+1.6 on Baker's [solid: D2, 5 folds].
- Main alone, slot series only: Codex Effusion +9.7 (5/5), Baker's +13.0 (4/5), lookat -2.6 (0/5; MCL -13, Effusion -8.9). Fat-sat
  only: Effusion +10.0, lookat -6.4. Series weighting (window-count normalisation) does not matter; equal series weights / noisy-OR /
  max over series lose [solid: D4].
- Crop experts, where every member sees the same crops: own-attention gating = the deployed consensus fusion (bone own|g20|P vs
  geo|g20|pm: text C +2.0 both, gold F +16.7 / +11.1); consensus without a gate ~0 [moderate: E1, 3 folds].
- Answer to "fusion of different models or filtering?": mostly filtering (window set + top-k gate). The window set comes from another
  pipeline's input design (Basin's slots), not from its attention.

### 2. Why it gains (D2b, D7, D7b, E1, S1, S2)

Window fusion corrects where REPORT-trained models put their attention. Report labels reward windows that predict what the
radiologist wrote (for Effusion: a weak 'some fluid' signal on non-fat-sat series that matches small / trace effusions the report
mentions, D10e); image-read labels (and, per earlier probes, the LB) reward windows that show the finding.
- The gain and the report cost sit in the same studies: the quartile with the most main attention on non-slot series carries Codex
  Effusion +4.0 of +6.1 and lookat -3.3 of -3.7 (Baker's +6.4 of +9.1); studies with >= 6 series and low main-vs-consensus attention
  agreement likewise [moderate: D2b partial substitution]. So per-study adaptive blending cannot separate gain from cost.
- Models trained on image targets do not gain from top-k gating / relevance changes: synovitis image heads (D7b: every alternative
  relevance -1..-8 Codex), synovitis crop experts (E1: every gate loses), oct13 R7 (image-head window fusion ~0). The fat-sat window
  SET still adds a little to a frozen-feature Codex Effusion head (H1c: +2.4 Codex, 4/4 models, fitted-to; lookat -5.1).
- The report-trained 3x3 Synovitis column gains from Effusion's windows (+6.6 Codex vs own-attention fusion +4.7, D7); the
  image-trained heads do not (D7b).
- Labels move the attention: v4b (Codex-calibrated vague cells) puts 0.150 of main's Effusion attention on non-fat-sat series vs v1
  0.202 (seed row 0.190); Baker's 0.190 vs 0.229 / 0.207; MCL / ACL / MOA unchanged [moderate: 1 v4b seed, 3 folds]. The window-fusion
  gain on top of v4b halves (Codex Effusion +3.7 vs +5.5..+7.0 for the v1 pair) (S1, S2).
- Replications (D10c, D10d): the sign pattern holds on the v1 seed row (s1: Effusion Codex-fit non-FS -0.56, Baker's -0.16; text-fit
  +0.34 / +1.10); for v4b the correction shrinks (Effusion -0.37 against +2.38 FS) or flips (Baker's +0.82). Basin's per-slot parts are
  near-collinear (its substitution logits are dominated by the study logit), so the split is informative for main only.
- Other classes (D10f, descriptive, gold-58 only): main's non-fat-sat part is relatively better on reports than on gold for Effusion
  (+208 gap), Fracture (+169), MCL (+132) and Baker's (+109), and relatively better on gold for Synovitis (-114) and Contusion (-73).
  Hypothesis: Fracture and MCL carry the same report-specific non-FS signal; untestable at scale without image reads for them.
- Caveat: Codex reads were made from contact sheets that favour fluid-sensitive series, so a part of any series-restriction gain on
  Codex can be shared view; gold-58 agrees in sign for Effusion / Baker's (+2.5..+3.7 / +9.1).

### 3. Inside the 0.961 / 0.962 package: saturated (P1, P2)

Every package recipe is linear in the 3x3 base logit and in the experts' fused terms, so a changed window form moves the package by an
exact linear map (p1_package.py, p2_experts_package.py; OOF folds 0-2 package of oct13_f5_win with every fused part, + expected gold).
- 3x3 window-form changes on top of f5_win: all within +-0.3 on trusted / lookat / text: Effusion k10 +0.15 Codex (gold -1.2), Baker's
  k10 +0.26; fat-sat-restricted Effusion / Baker's ~0 / -1.4; 3x3 OA fusion -0.3 reports (the OA experts' crop fusion already does
  it); menisci own-gate +0.07; ACL k5 half-blend lookat ACL +1.3 (3/3) / gold +2.5 but chosen in-sample [solid: 3 folds, exact].
- Expert crop-fusion forms: OA own|g0|P lifts PF lookat +1.8 (3/3) but trusted -0.48 (0/3), gold -1.8; bone own|g20|P = deployed
  (gold +0.6), others lose [solid: 3 folds]. The deployed forms stay.
- So the remaining value of window-level work is (a) new members (label-changed mains, new backbones) and (b) training-time use.

### 4. Fitted knobs: coarseness, learned heads, rankings, cross-class (D3, D5, D6, D7)

- Coarseness (D3; gate k x blend a, LOFO): per class > per group > global (mean over findings +1.71 / +1.60 / +1.05 on the
  image-ref objective). Stable per-class picks on all 5 held-out folds: Effusion k10 a1 (Codex +6.7), Baker's k20 a1 (+9.1), Medial OA
  k10 a1 (lookat +3.0 on the bare 3x3). MCL, PF, Contusion, Fracture must be able to opt out (any pick loses held-out). A coarse grid per
  class is the right granularity; continuous fits do not beat it [solid: 5-fold LOFO].
- Light learned heads on the 3x3 window outputs (D6; attention sharpness, consensus share, Basin-map prior, 30 series-type priors,
  backbone weights; global / group / class sharing; LOFO, L-BFGS): text-fitted heads move reports +0.1..+0.15 and gold macro +2..+3,
  Codex ~0. Fitted on Codex reads: Effusion +3.7 (fitted-to), below the unfitted top-20 gate (+6.1). The hard top-k gate is a
  nonlinearity the smooth heads do not reproduce; text targets cannot discover the image-read gain [moderate].
- Rankings (D5): Borda / reciprocal-rank relevance below the geometric-mean attention; rank-blends = logit blends [solid].
- Cross-class relevance (D7): group relevance loses for every 3x3 column; Effusion's windows help only the report-trained 3x3
  Synovitis column, not the image heads (D7b) [moderate].

### 5. Transfer to other members (S1, S2, D9)

- Label-changed main (tl_v4b, Master's request): the fusion still adds on top (Codex Effusion +3.7, Baker's +4.4 on f0-2; 5 folds +3.5 /
  +1.8) but about half of the v1 gain. The optimum is unchanged (k 10-20); no re-fit of the 'win' constants needed. v4b's Codex level is
  fitted-to (its vague cells were calibrated to Codex reads); gold-58 v4b 0.9325 vs v1 0.9292, win +0.4..+1.0 on every main.
- More models per backbone (D9: 2 main seeds): the gain holds (+6.8 / +12.0 vs +5.5 / +12.8 with one) -> expected to hold for the
  3-seed all-data package [moderate: 3 folds].
- Window-level fusion of a label mix (v1 + v4b) adds nothing beyond the per-main gains (S2).

### 6. Ways tried to improve post-train predictions (>= 6), with verdicts

| # | way (level) | best result | verdict [strength] |
|---|---|---|---|
| 1 | window-set filtering: slot / fat-sat series (window, series) D2, D4, D4b, D4c | main alone + slot + top-20: Codex Effusion 0.9537 (>= 3x3 study mean 0.9498; seed row 0.9510 vs 0.9507), below the fused 3x3; half of the deployed fusion's gain | explains the deployed gain; already inside the package (geo relevance) [solid] |
| 2 | gate k x blend a at global / group / class coarseness (window) D3 | per class best: Effusion k10 +6.7, Baker's k20 +9.1, MOA k10 lookat +3.0 (bare 3x3) | package-level ~0 (P1) [solid] |
| 3 | light learned fusion heads, 5 parameter families, 3 sharing levels (window -> study) D6 | text-fit: reports +0.15, gold +2..+3; Codex-fit Effusion +3.7 < unfitted gate | no [moderate] |
| 4 | rank-based relevance / values / blends D5 | all <= geo | no [solid] |
| 5 | cross-class relevance (window) D7, D7b, P3 | Effusion windows for the 3x3 Synovitis column +6.6 Codex; image heads -1..-8; as a G member -0.2..-2 | no [moderate] |
| 6 | expert crop-fusion form (crop) E1, P2 | OA own-attention prob pooling: PF lookat +1.8 but trusted -0.48 | deployed forms stay [solid] |
| 7 | context partial-out: subtract main's non-fat-sat-series signal (series) D10, D10b | Codex-fit Effusion +6.8 held-out; unfitted contrast on the package +0.9 Codex Effusion / +1.4 Baker's (3/3), lookat -1.4 | mechanism evidence; small trade-off knob [solid / weak gain] |
| 8 | head-level adapters on frozen window features (head) H1, H1b, H1c, G2, G2b | inner world: Codex-target head Codex Effusion +23 (4/4, fitted-to), in an inner 3x3 +7.5 for lookat -0.8; on the deployed members inside the package: +-0 (2 CV seeds), Synovitis head -0.5..-2.1; hard top-20 gate in training: worse everywhere | no package gain: the package is already image-aligned [moderate-solid] |
| 11 | per-group calibration offsets, groups identifiable at inference: vendor / field / protocol (study) P5a | report-fitted vendor offsets: text +1.35 (fitted-to) but Codex Effusion -10, Synovitis -12, Baker's -17, gold macro -10; Codex-fitted: Baker's +5.9 (3/3), gold Effusion -5..-14 | no; never fit group offsets on reports (report leniency differs by vendor / site) [moderate] |
| 12 | group-wise recipe weights (expert vs 3x3) by expert crop count / protocol / vendor (column) P5b | best Lateral OA +0.70 (3/3, protocol); others -0.3..-3.9 held-out | no [moderate] |
| 13 | disagreement-aware study fusion: mean - lam * sd over backbones, confidence weights (study) D8 | MOA lookat +1.4 (4/5), ACL +0.4 (4/5), rest <= 0 (3x3 Synovitis +3.9, unused column) | no package-relevant gain [moderate] |
| 14 | cross-class stacking of package columns, group / all 12 (column) P4 | text-fit: text -3.1..-3.4, gold macro -7..-12; Codex-fit: Codex Effusion +3.6 (fitted-to) but gold Effusion -27..-36 | no; learns co-mention patterns [solid] |
| 9 | label-variant mains at window level (window) S1, S2 | fusion holds on v4b at ~half the gain; no label-mix bonus | keep constants [moderate] |
| 10 | model count per backbone D9 | 2 seeds: gain holds (+6.8 / +12.0) | transfers to multi-seed all-data [moderate] |

D2b showed that the fusion's gain and its report cost sit in the same studies (non-slot attention quartile), so no study-level gate on
inference-time features can separate them (D8 and P5 confirm: no gain from per-study or per-group weights).

### 7. Head-level adapters on frozen window features (H1, H1b)

PL inner main models (in_m_h3 f0/1/2/4) with stored post-norm window features h of their held-out inner fold; the attention-MIL head is
re-fitted from its own weights (exact rebuild) with 5-fold CV inside each fold (3,420 held-out studies, 4 models).
- Plain fine-tuning on ~680 studies drifts below the original head (lookat -3.2, text -7.0): small-data drift; variants compared to it.
- Hard top-20 gate inside the training pooling: worse on every read (lookat -3.7, Codex Effusion -4.0, Synovitis -14). Gating is a
  test-time correction of report-trained attention, not a better training pooling.
- Learned per-finding series-type attention prior with text targets: exactly 0 (also D6): reports do not ask for it.
- Codex reads as the Effusion target (head only): Codex Effusion +23 / report-vague +25 on 4 / 4 models (fitted-to), lookat Effusion
  -5, other columns unchanged. Restricted to fat-sat series (H1c): +2.4 more (4/4), lookat -5.1 more. Re-fitting any head (text or
  Codex) halves its non-FS attention (0.230 -> 0.114 / 0.137), so the Codex gain comes from the value function, not the attention. In an inner 3x3 at blend w 0.3: Codex +7.5 / vague +9.5 for lookat -0.8, vs the window fusion +6.1 /
  +6.8 for lookat -4.7; together +11.2 / +13.6. An image-read head is the better trade-off and adds to the fusion.

- Deployed members (G1 guest GPU dumps of sub_b0 f0-4 with h; G2 / G2b on CPU; 2 CV seeds): a Codex-target Effusion head blended into
  the package's Effusion adds nothing (w .1: Codex -0.1 / +0.0, gold +1.2 = one pair; w .3: Codex -1.4 / -1.0); alone it is 12-14 below
  the package (package Codex Effusion 0.9728). The fat-sat variant: the same (w .3 gold -2.5). A Codex Synovitis head as an extra G
  member: -0.5..-2.1 Codex, gold -5..-13. The control (head re-fitted on text) moves Effusion back toward reports (lookat_nosil +4.5,
  Codex -4.6): the trade-off, not a gain. The package's image-target members (img_eff head, window fusion, synovitis image heads /
  crop experts) already carry what heads on the report-trained main's features can add; H1b's inner-world +7.5 had no img_eff.

### 8. Evidence from prior experiments that fits

- oct9 attention probes (experiments/v0/oct9_baseline_2nd/notes/attention_findings.md): main 'dilutes' attention on non-FS series for
  MOA (axial non-FS -25), fracture (-30), synovitis (coronal non-FS -28); proposed fixes type_prior / two-level attention (never run).
- oct7 research log: report-trained attention is already sequence-aware for fluid (FS lift 1.3, T1 0.44), but not enough (D10).
- Synovitis crop experts: FS-only windows +4.3 vs base on 3 / 3 folds (experiments/v1/oct11_synovitis_crops/ideas.md C9): a training-
  time window-set restriction that worked.
- PL session review 78: FS-only input diet 'about free' on OA / menisci / synovitis; plane-only diets cost 1-6e-2 per finding.
- PL noisy student with series dropout 0.2 (mbl_T3pH_ct_S): no report gain at 20 epochs (cx_syn +39) -> random series dropout is not
  the same as removing context series.
- oct13: the PL / clean-label stages and the window fusion fix the same image-read effusion errors (fusion gain on B0 +7.4 Codex vs
  +1.4 on K6r); the img_eff head (trained on Codex reads) +18.4 Codex at w 0.3. Same story as v4b here (S1, S2).
- LB: oct13_f5_win (all window / crop fusions) +0.004 over 0.957; the gains it was chosen on were image-read gains.

### 9. How to reuse the ideas, and the GPU plan (needs the Master's / user's go)

Reuse now, CPU, no GPU:
- For every new main (label sets, backbones): read the label-free attention diagnostic (mean attention mass on non-fat-sat series for
  Effusion / Baker's; s2_labelmix.py part A; v1 0.202 / 0.229, v4b 0.150 / 0.190) and the within-main fusion delta (s1_swap.py). Keep
  the 'win' constants (S1).
- Window fusion is a correction for report-trained members only: do not add gating / relevance changes to image-target members
  (synovitis image heads, crop experts' image heads) (D7b, E1).

GPU plan, ordered by evidence per GPU-minute (screens on folds 3 / 4 against the same-recipe refs; reads: Codex / gold first,
report metrics as the floor; package read on folds 0-2 only if a screen passes):

| # | arm | why (evidence) | cost |
|---|---|---|---|
| G1 | DONE 20:31-20:40 (Master's guest slot, 9 GPU-min): sub_b0 f0-4 OOF + gold window dumps with h -> data/win_h | enables head-level work on the real members | done |
| G2 | DONE on CPU: image-read heads on the deployed members' frozen h -> no package gain (section 7) | | closed |
| G3 | 'fluid evidence' main: from scratch with the fat-sat-only window diet (fs_only, ported by Labels Work, CPU-checked: 58% of windows kept) on v4b, read on Effusion / Baker's / Synovitis with the window fusion on top. PARKED until the user OKs training. Expected value LOWERED by G2: the package's Effusion is already image-aligned, so only a replacement-main read (all columns) or Baker's / Synovitis may gain | D4 / D4b / D4c: a filtered v1 main reaches the 3x3 study mean on Codex Effusion; D10: image truth subtracts main's non-FS part; synovitis crops FS-only +4.3 (3/3) | 1 arm x 2 folds x ~13 min |
| G4 | main from scratch on v4b with `type_prior` on (exists in oct15_labels main) + the attention diagnostic | v4b already moves fluid attention to FS (0.150 vs 0.202); a prior may finish the job; on report labels it is 0 (D6, H1), so only with image-like labels | 1 arm x 2 folds |
| G5 | gold-58 window dumps of the all-data seeds | confirm the fusion behaves as with fold models (D9: holds with 2 seeds) | minutes |

Update 22:46 (sections 14 / H8-H10 per class), for any later GPU time:
- To build an ensemble, train more mains on different good label sets, each with a NEW seed, rather than seed replicates of one set.
  Pick label sets that beat v1 on their own (v4b + vag15; the average set does not, section 16a). Diversity from label sets is worth
  about one seed, and pairs that differ in both diversify more (two seed contrasts agree).
- The main slot's label set matters most where the main is complementary to Basin: Effusion (silent / vague cells). Ligament gains
  of label mains wash out in the 3x3.
- Window attention can be shared across mains (interchangeable); spend compute on values / features, not on attention variants.
- tl_mh3 (label heads on one backbone) tests whether one backbone can carry the label ensemble; read it with h12_multihead.py.

Not recommended: training with a hard top-k gate (H1: worse on every read), type priors trained on report labels (D6 / H1: 0), distilling the fused predictions (oct13: PL / clean-label stages fix the same errors), random series dropout (PL
noisy student: no gain), more post-hoc fusion knobs (saturated, P1 / P2).

### 10. Labels and the shared backbone (user 21:05-21:30; H2-H7, with Labels Work's arms)

Head-level label swap (H2 / H2b; deployed sub_b0 fold models, frozen window features h from the G1 dumps, head re-fitted with CV
inside each fold; control = the same re-fit on the trainer's own v1 targets, which is neutral):
- v4b targets at head level: gold macro +10.3 vs +11.1 for the from-scratch tl_v4b (Effusion +11.2, MOA +10.9, LOA +9.7, Fracture
  +9.7, Synovitis +73; menisci not hurt, unlike from scratch). Codex 30-75% of the from-scratch gain (fitted-to). Report gains of the
  from-scratch v4b (ACL +3.7, MCL +3.6, MOA +2.5 lookat_nosil) are NOT reproduced: they need the backbone. So a label set's gold effect
  is mostly head-level, its report effect backbone-level.
- Inside the package (window fusion + recipes) the head-level swap changes nothing (gold macro -0.2, reports +-0.5), the same as Labels
  Work's from-scratch package swap. Member-level label gains are absorbed by the package's image-target members and experts.

Head-level label screen (H3; user: rank many label sets cheaply, but only after a strong confirmation): validation pre-registered by
the Master (views trusted_nosil / lookat_nosil / text / gold per group; Spearman >= 0.7, >= 80% pass / fail agreement, no sign flip
on clear pairs; screen band = v1 head re-fits on the 3 same-recipe feature sources). RESULT (17 set-fold pairs): FAILS on every view
x group (best rho: lookat_nosil menisci 0.83, fluid 0.79; tr9ns -0.48, gold9 +0.11). Noise ceiling: the same set's single-seed
from-scratch deltas do not replicate f3 -> f4 (rho over 6 sets: tr9ns -0.03, OA -0.49, gold9 +0.20, text9 +0.54), so no screen can be
validated against them at 0.7; the screen is more self-consistent (tr9ns 0.66, text9 0.71) but measures the head-level effect. Do
not rank label sets with it. A valid cheap screen needs >= 2 seeds per from-scratch set as the target and a partial-backbone stage.

Differently-labelled mains (H4 / H5 / H7; tl_v4b, tl_vag15, tl_vsf from scratch, f3-4):
- Routing classes to the member trained on 'their' labels (structural from vag15, fluid from v4b) won on f3 (which motivated it) but
  not on the clean f4 (tr9ns +1.9 vs members +2.7 / +2.8). The plain average of the two label-diverse mains is the robust form: tr9ns
  +1.4 / +4.2, gold9 +9.1 / +7.9 (f3 / f4) vs the same-compute 2-seed v1 average +1.3 / +1.6, +2.1 / +0.4: label diversity ensembles
  better than seed diversity at member level.
- Window level (H5, pre-registered 60-cell grid of attention source x value source x pooling): no cell beats the plain average
  held-out; per-group cross-fitted picks overfit.
- Inside the bare 3x3 (H7) the label pair's advantage over the seed pair mostly vanishes on reports (tr9ns +0.1..+0.2, lk9ns +0.3..+0.4)
  and keeps the gold / Codex direction (gold9 +1.7..+2.4, Codex Effusion +5..+8, fitted-to).
- Interference: vsf (vag15 targets on the 7 structural classes + v4b targets on fluid / Syn / C / F, one backbone) is below vag15
  (identical structural targets) on all 6 structural group-folds (gaps 1-3, ~ the refs band; 1 seed each) -> consistent with the
  user's point that other classes' labels move a class through the shared backbone; a seed replicate (scheduled by the Master) decides.

Generalisation (part 3 of the Master's request):
- Same family as 'experts always helped': the OA crop experts (OA + menisci targets on compartment crops), the bone experts (F / C),
  the synovitis crop experts and the image heads all train a backbone on a few classes; none is pulled by the other classes' targets,
  and every one of them beat the generalist's column. The interference here is the generalist-side view of the same effect.
- Post-hoc forms (tested here): per-class member routing (fragile, f3 only), plain averaging of label-diverse members (robust at member
  level, diluted in the 3x3), window-level attention / value crosses (no gain), head re-fits on frozen features (gold-level label
  effects at zero training cost; no package gain).
- Training-side forms (for Labels Work / GPU, not run): owned-class-only generalists (a main trained on the 7 structural classes, one
  on fluid / Syn / C / F), down-weighting the loss of non-owned classes when their labels change, class-group-specific last blocks, and
  the label-diverse 3x3 (main and R50 on different label sets: the Master's fold-1 package read).

### 11. REREAD (user via Master ~22:00: report references are v1-built; primary = gold-58 per group + Codex where the arm's targets were not Codex-calibrated; report reads = 'agreement with v1')

| conclusion (section) | rested on | gold / independent Codex | verdict |
|---|---|---|---|
| window-fusion gain = window-set filter + top-k gate; value fusion 0 (1, D1 / D2) | Codex (independent for v1 mains) | gold Effusion +2.5..+3.7, Baker's +9.1 for the fused forms | holds |
| window fusion 'costs' lookat Effusion -1..-6, MCL -4.6 (1, D1 / D3) | reports | gold Effusion +2.5..+3.7; MCL gold +2.3..+9.1 under geo-gated forms | reversed as a cost: it is disagreement with v1, not lower accuracy (gold n = 58: 1-4 pairs) |
| per-class opt-outs of the gate (MCL / PF / C / F) (4, D3) | reports | MCL gold +2..+9, PF -1.3..+9, F +11..+21 under gated forms (D1 / D5) | undecidable (gold favours opting IN for MCL / F; small n) |
| report-trained main takes a report-specific non-fat-sat signal (2, D10 / D10c / D10f) | text-fit vs Codex-fit (Codex independent of v1) | gold: non-FS part relatively weaker on gold for Effusion / Baker's / F / MCL (D10f) | holds |
| package saturated for 3x3 window forms (3, P1) | reports + Codex + gold | gold macro -0.2..+0.7 for every variant (OA win10 +0.7: Lat OA +5.8) | holds (small gold upsides undecidable) |
| oct13 crop fusions on robust gold (gf_experts.py) | gold | OA geo|pm|g10: Lat OA +13.5 -> +27.7 without the contested cell, PF +13; bone: Fracture +16.7 (no contested cells), Contusion +10.8 -> -1.5 without the 4 contested Contusion cells | OA and Fracture hold; the Contusion crop fusion is undecidable (its gold gain was contested pairs) |
| OA crop fusion form: deployed geo|pm|g10 stays; own-attention prob pooling better on reports (E1 / P2) | reports (E1), reports + gold (P2) | deployed wins gold Lat OA (+23 vs +7.7); P2 swap gold -1.8 | holds on gold; oct13's report gain (+0.74) is agreement only, its gold Lat OA +17..+23 carries it |
| learned light heads: no gain (D6) | reports + Codex | gold macro +0.5..+2.9 (text-fit) | weakened: small gold upside, not tested further |
| rankings / cross-class relevance: no gain (D5 / D7 / D7b) | Codex + reports | gold mixed (D5 rrf / borda macro +0.6..+2.6) | holds (Codex-led) |
| cross-class stacking harmful (P4) | text-fit + Codex-fit | gold macro -7..-12; Codex-fit gold Effusion -27..-36 | holds (stronger on gold) |
| per-vendor offsets fitted on reports harm image truth (P5a) | gold + Codex | gold macro -10 | holds |
| group-wise recipe weights: no gain (P5b) | reports only (chosen and read) | not read | undecidable |
| disagreement-aware fusion: no package gain (D8) | reports + Codex | not read | undecidable |
| hard top-k gate in training worse (H1) | reports + Codex (fitted-to for cx) | Codex -4.0 for the text head (independent); gold not read | holds on Codex; gold undecidable |
| head-level v4b swap: gold effect head-level, report effect backbone-level (H2) | gold (primary) + reports | gold macro +10.3 vs +11.1 | holds |
| head-level label screen fails validation (H3) | reports + gold views | gold rho 0.11; from-scratch gold9 does not replicate f3 -> f4 either (0.20) | holds |
| label pair: plain average > routing (H4) | reports (tr9ns) | gold9: routing f3 / f4 +11.2 / +4.0 (best member +6.1 / +2.4), average +9.1 / +7.9, seed pair +2.1 / +0.4; ROBUST gold (8 contested cells dropped; own9 vs the seed pair): average +4.9 / +3.5, routing +6.9 / +0.4 (f4 Lat OA +10 -> -21) | average holds on robust gold (both folds); routing fragile -> keep the plain average |
| window-level attention x value cross: no gain (H5) | reports (choice and read) | fluid pick: gold and tr_ns disagree | undecidable |
| label pair inside the 3x3: gain vanishes (H7) | reports | gold9 +1.7..+2.4 vs the seed pair | weakened: a small gold gain survives |
| image-read head on deployed members: no package gain (G2 / G2b) | Codex (fitted-to) + reports + gold | gold +1.2 / 0 / -2.5 | holds |

REREAD caution (Master ~22:00): gold-58 is one fixed set (K12fkd won on gold breadth and lost on the LB). Verdicts that rest on gold
ALONE (H4 routing on gold9; the gate opt-ins for MCL / PF / F) are LB hypotheses, not package decisions. Prefer forms where gold and an
independent image read agree; where they cannot be checked, keep the simpler form (plain average over routing).

### 12. Gold study by study (user ~22:00) and the backbone x head grid (H6)

- Tools: scripts/gold_flips.py (per-cell AUC contributions with v1 status / quote, image reads on gold, series facts), gf_runs.py
  (comparisons -> out/gf_*.md), gold_contested.py (contested cells; robust deltas), gf_robust.py.
- Explainable gold patterns: the window fusion's Fracture gain = focal fractures moved up (eminence, subchondral, impression,
  report-silent); the label pair's ACL / MCL gains = report-'below' low-grade / degenerative negatives moved down (vague-cell
  calibration matches gold's threshold) and grade II MCL sprains up.
- Contested gold: 8 / 696 cells contradict both the explicit report status and every image read (4 Contusion: a narrower gold
  definition). Single contested cells move a class by up to ~+18 (Lateral OA). Read gold deltas with and without them; a 'gain' that
  flips sign or halves without them is fragile.
- Gold anatomy of the deployed recipe (gf_package.py; package vs bare 3x3, all -> robust x1e-3): Synovitis +98 (report-silent
  synovitis up, report-claimed but gold-negative down: image-trained members agree with image-based gold), Fracture +50 (report-silent
  fractures up; impaction / osteochondral injuries that gold calls negative down, insufficiency fractures up), PF OA +24.5 (patellar
  chondropathy cells up), LM +14.9, MOA +9.3 -> +6.6, LOA +1.9 -> +12.8, Effusion +2.5 -> +7.8, Baker's +9.1, MM -2.4; Contusion +4.0 -> -3.0
  is the only contested gain. The experts' gold gains are explainable and robust.
- H6: the cross-class effect lives in the backbone features: under any re-fitted head, the vsf backbone stays below the vag15
  backbone on the structural groups (f3 -3..-5.6 per group; f4 small / mixed); fluid gold follows the backbone too. H2's head-level gold
  gain was mostly Synovitis; on the 9 owned classes it is ~+4.7 (5 models).
- Package design implications (task 3): (1) label sets interact through the shared backbone, so a label change on one class group must
  be read on every group, and a per-group mix inside one model (vsf) can lose to two models that each carry one label regime; (2) the
  cheap ways to keep the benefit without the interference are model-level: average label-diverse members (stable on robust gold) or
  give different backbones different label regimes (Master's fold-1 label-diverse 3x3); (3) heads cannot repair it post hoc; (4)
  specialists (experts, owned-class-only mains) avoid it by construction, the training-side form to test first if a label regime
  helps one group and hurts another.

### 13. Handoff (after this session)

- Partial-backbone screen: DROPPED by the Master (01:0x; its 6-pair validation would be weak and the GPU goes to Lateral OA V3 /
  Synovitis V3). Code stays (main_pb/, scripts/pb_launch.py, out/pb_jobs.txt, h11_pb_validate.py) with one prototype pair on f3
  (pb_v1 / pb_v4b) if anyone needs a cheap label screen later; validate it first.
- Watcher: scripts/watch.sh (Labels Work mains landing at e19, pb jobs done). Re-reads when they land:
  `scripts/h8_labelens.py` then `scripts/h8_report.py` (tl_v1 / tl_v1s1 give seed N 3-4; *s2 replicates fill the mixed pool);
  `scripts/h12_multihead.py` (tl_mh3 heads: one-backbone label heads vs separate mains vs seed N3).
- P6 pair_label (package main slot = v4b + vag15) still needs tl_vag15 f0 / f2 window dumps (not on Labels Work's schedule).
- G3 (fat-sat-only main): ported by Labels Work, parked until the user OKs it as training; handed to Synovitis V3 (00:5x).
- 00:5x (user via Master: GPU to the new Lateral OA V3 / Synovitis V3 sessions once ideas are exhausted): Fusion has no GPU-worthy
  idea left. The pb block is optional (its 6-pair validation is weak; worth it only if many more label sets get screened); P6 and G4
  dropped (low ROI). 'What Fusion knows' notes sent to Lateral OA V3 and Synovitis V3 (ledger 00:5x).

### 14. Label-diverse vs seed ensembles (user ~22:25 via Master; PRE-REGISTERED 22:08, before computing)

Question: do ensembles of mains trained on DIFFERENT label sets beat seed ensembles of the same recipe at equal N, and does label
diversity keep adding where seeds saturate?
- Pools per fold (from-scratch main arms of Labels Work, e19, read-only; whatever exists at compute time is listed in the result):
  LABEL pool (seed 0): f3 = v4b, vag15, consnp, consnr, consv3, v5, vsf, vsfe, rules, txt, sil, nm05; f4 = v4b, vag15, consnp, consnr,
  consv3, vsf. SEED pool (v1 targets): sub_b0, sub_b0_s1, + tl_v1 s0 / s1 when they exist. MIXED pool (user correction 22:20): label
  sets with different seeds (vag15 s2, v4b s2, consnr s2) when they exist. Recipe variants (effb0_lr1e-3_R) are NOT part of the
  question (user correction).
- Ensembles: z-aligned logit mean (per class, z over the fold's OOF; gold z with the OOF stats) at N = 2, 3, 4:
  (a) all label subsets of size N (mean and spread over subsets); (b) the label-free max-diversity subset (lowest mean pairwise OOF
  correlation of the members' z-logits); (c) fixed principled set v4b + vag15 (+ consnr, + consv3 for N 3-4); seed ensembles: all
  subsets of the seed pool. No gold / report-based selection anywhere.
- Levels: study logit average first (h8_labelens.py); window level next if time allows (the win operator with each member's own
  attention, and with consensus attention; label-pool attention x seed-pool values and the reverse), using the _sm window dumps.
- Reads: gold9 and gold9rb (contested cells dropped; both must agree), gold per group, Codex only for ensembles whose members are all
  non-Codex-calibrated (seed pool, rules, txt, nm05), nosil reports as 'agreement with v1'. Provenance per ensemble.
- Decisive comparison: label-N vs seed-N at equal N, per fold and level; the gain-vs-N curves for both pools; at the end, the bare 3x3
  with the best form in the main slot (NFNet / R50 deployed).

Results (H8 / H9 / H10; x1e-3; gold9 / gold9rb = contested cells dropped; reports = v1 agreement):
- Member level, study average (H8): at N 2 label pairs beat the seed pair on gold on both folds (f3 +9.1 vs +6.9, rb +9.0 vs +7.4;
  f4 +5.5 vs +0.9, rb +2.3 vs +0.3; vs the v1 main alone); reports mixed (f3 +0.5 vs +1.3, f4 +3.4 vs +1.9 tr9ns). Label ensembles keep
  gaining N 2 -> 4 (f3 +9.1 -> +10.9, f4 +5.5 -> +8.2; rb f4 +2.3 -> +4.9). The seed pool has one pair per fold so far (tl_v1 seeds
  pending). Single-model gold noise is large (seed swap +7.7 / -4.7). (Recipe-variant rows, effb0_lr1e-3_R, are not part of the question
  per the user's correction; they stay in out/h8_labelens.csv only.)
- Window level (H9): no window form beats the study average for these ensembles; label values with v1 attention > label attention
  with v1 values.
- Inside the bare 3x3 (H10): the advantage shrinks: N2 label - seed pair gold9 +2.4 / rb +1.1 (grid), +1.7 / +0.4 (with win); reports
  +-0.2. (N 3-4 comparisons need seed-N >= 3: re-run with the tl_v1 seeds.)
- Verdict: label diversity is a better ensemble axis than seeds at member level (moderate: 2 folds, gold-led, seed pool small);
  in the deployed 3x3 form most of it is absorbed. Worth testing where members are not diluted (an extra label-diverse main as a 4th
  pipeline, or label-diverse Basin members: the Master's fold-1 grid), and re-reading once tl_v1 seeds give seed-N >= 3.

Per class and per group (user ~22:25 via Master; scripts/h8_report.py -> out/h8_perclass.md; member level, study average, f3-4).
Yardstick y = Labels Work's per-class member seed noise (v1 seed pair, mean |gold diff| over f0-4; copied constants); groups = max of
this session's 3 replicate pairs and the classes combined as independent. Bar = 2 y on gold AND gold_rb, same sign on both folds.
- N 2 label pairs - seed pair, all9 (f3-4 mean): gold +3.4, rb +1.8 (y 7.7). Split: +3.0 / +1.4 because the label-set members are
  better alone, +0.4 / +0.4 from extra diversity. The ensembling gain over own members is about the same for label and seed pairs
  (+3.6 vs +3.2 gold9). So the label-pair edge is mostly "better members", not "more diverse members" (but see 16a: the members
  were 'better' mainly than a weak baseline).
- Beyond 2 y: only Synovitis (+64 on both golds, y 7.2), all of it member effect (the label sets fix v1's Synovitis labels); expert-owned
  in the package, info only. No class or group loses beyond 2 y.
- Below 2 y but same sign on both folds and both golds: Medial OA + (member effect, ~1 y), PF OA - (~0.6 y; the seed pair gains more
  from diversity on PF), Fracture - (member -, diversity +). ACL + is one-fold (f4) driven. Lateral OA rb flips sign (contested cell).
- Groups (gold / rb, fold mean): ligaments +6.1 / +6.1 (f4 only), OA +3.6 / +0.3, fluid +3.1 / +0.7, menisci +0.7 / +0.7, bone -2.2 /
  -1.4; none beyond 2 y.
- Label ensembles' diversity gain grows with N (all9 +3.6 / +5.1 / +5.8 at N 2 / 3 / 4); every group follows. Seed N >= 3 is needed to
  say whether seeds saturate earlier (tl_v1 seeds pending).
- Fold 3 only: pairs that differ in label set AND seed (mixed, 11 pairs with vag15s2) gain +4.9 from diversity vs +2.9 (label pairs,
  same seed) and +3.1 (seed pair); the two sources of diversity seem to add. One seed-in-label pair (vag15 + vag15s2): +1.3. Weak: one
  fold, one replicate set; re-read when v4bs2 / consnrs2 / f4 replicates land.
- Codex (independent members only: rules, txt, nm05 on f3): these label pairs do not beat the seed pair on Effusion / Baker's;
  rules-containing pairs gain on Synovitis. These are the weaker label sets; not representative of the pool.
- Practical: picking better label sets is what pays; diversity from labels adds about what a seed does at N 2, and more when combined
  with a seed change. In the 3x3 (H10) the edge was already small.
- Gold study by study behind these patterns (scripts/gf_labelens.py -> out/gf_labelens_f{3,4}.md; whole label pool mean vs the v1 seed
  pair; delta AUC by v1 status x gold label):
  - PF OA (-5 on both folds, robust too): the loss sits on gold-positive cells that v1 calls 'unsettled' with an ungraded "patellar
    chondropathy" ("Condropatía rotuliana") plus one focal grade-4 trochlear chondropathy; v1 'below' focal patellar chondrosis
    negatives move up. The label sets treat ungraded patellar chondropathy as weaker than gold does. This is the same cell type the
    deployed package gains on (section 12). A label rule that counts ungraded patellar chondropathy as PF OA would target it, but it
    would be gold-motivated: tag gold fitted-to for PF OA if anyone builds it.
  - Medial OA (+17 f3, +6 f4): gold-positive cells with v1 'conflict' cartilage descriptions (ICRS II erosion with mild compartment OA)
    move up; v1 not_mentioned / unsettled negatives move down. Explainable: the label sets resolve conflicting cartilage text closer to
    gold.
  - ACL (+6 f3, +23 f4): same direction on both folds: v1 'below' negatives (partial / low-grade / mucoid) move down, hedged tears that
    gold calls positive move up (as in section 12's fold-4 pair read).
  - Fracture (+10 f3, -7 f4): both folds raise report-silent gold-positive fractures and lower a hedged "bony avulsion" positive; the
    v1 'meets' positives go opposite ways. Not a stable direction.
- Attention of one model pair x window values of another, per class (Master relay; h9_report.py -> out/h9_perclass.md,
  h9_attswap.py -> out/h9_attswap.txt; label pair tl_v4b + tl_vag15, seed pair sub_b0 + sub_b0_s1, f3-4):
  - vs the plain 4-model study average: v1 attention x label values adds nothing on robust gold (all9 +1.9 / rb -0.1 with consensus
    attention); class moves follow whose VALUES are used (e.g. Synovitis follows the label models' values).
  - Values fixed, attention swapped: all9 moves <= 1 (consensus) and <= 3, fold-inconsistent (top-20 gate); no class beyond 1 y. The
    attention of v1 mains and label-set mains is interchangeable; the label effect is in the window values.
  - The top-20 gate at member level on all classes gives large swings both ways (Lateral OA up ~1.5 y on rb, Effusion / Lateral
    Meniscus down ~1.3-1.7 y). Without the Basin window-set filter, the gate is not a gain here, matching section 1's split.
- Inside the bare 3x3, per class (h10_3x3ens.py now stores per-class gold / gold_rb; h10_perclass.py -> out/h10_perclass.md; label
  pair v4b + vag15 vs seed pair as main; x1e-3 fold mean, member level -> 3x3 grid): all9 +7.2 -> +2.4 (rb +4.2 -> +1.1). Ligaments
  wash out (ACL +11 -> +1, MCL +10 -> -1: the Basin members dominate there); Medial OA keeps ~1/3 (+9 -> +3, main's share); Lateral OA
  +22 -> +5 (rb negative both levels: contested cell); Effusion GROWS (+0.6 -> +8.7, rb -1.9 -> +5.2; same sign both folds; < 1 y): the
  label-pair main complements Basin on Effusion better than the v1 pair. With window fusion Effusion stays +6.8, Baker's goes to 0.
- Why Effusion grows in the 3x3 (h10_effusion.py -> out/h10_effusion.txt, out/gf_h10_effusion_f{3,4}.md): the label-pair main is less
  correlated with the Basin members on Effusion (OOF corr 0.944 vs 0.964 f3, 0.945 vs 0.960 f4), mostly on report-silent / vague cells
  (0.911 vs 0.945, 0.920 vs 0.946). So it adds information where Basin and the v1 main agree most. Gold cells that move: v1-'unsettled'
  positives (effusion mentioned without a grade) and v1-'below' small-effusion positives go up. Two gold negatives that every image
  reader calls positive go down; they carry part of the all-cell gain (robust: f3 +3.9, f4 +6.5).
- Seed x label (h8_seedxlabel.py -> out/h8_seedxlabel.csv): each label-set main X (seed 0) paired with the v1 main of the same seed
  (sub_b0) vs a different seed (sub_b0_s1). Ensembling gain over the two members (gold9 / rb, x1e-3): f3 +2.5 -> +3.9 / +2.6 -> +3.8,
  f4 +3.5 -> +5.4 / +4.4 -> +5.4; paired per X: +1.6 (17 / 18 positive) / rb +1.1 (16 / 18); menisci +2.3, OA +2.6, fluid +1.5,
  ligaments -0.7. Mains that share the seed (same init and data order) make correlated errors even when their labels differ. All 18
  pairs share the same two v1 mains, so this is one seed contrast per fold (both positive): moderate. Labels Work's tl_<set> arms all
  use seed 0, so same-pool label pairs understate the diversity a real label ensemble (one seed per set) would get.

### 15. Jackknife of the gold claims (Master 10-16 ~01:15: the new standard)

Leave-one-gold-study-out, definition copied from Labels Work's gold_robust.jackknife: FRAGILE = a sign flip or |delta| halving when
one gold study is dropped; groups drop the study from every class at once; fold mean drops it from both fold models. Scripts:
h13_jackknife.py (label-pair margins, H8 per class, H10, seed x label) -> out/h13_jackknife.md; h13b_jackknife_claims.py (crop fusions,
member-level gate) -> out/h13b_jackknife_claims.md. x1e-3, fold mean, gold / gold_rb.
- NOT fragile (on both golds unless noted):
  - v4b + vag15 vs the seed pair, member level: all9 +7.2 / +4.2, ligaments +10.6, Medial OA +9.3 / +10.8, Synovitis +56.
  - All label pairs vs the seed pair: all9 gold +3.4 (rb +1.8 is fragile); Synovitis +64; Medial OA on rb (+9.4).
  - Seed x label diversity: all9 +1.7 / +1.1, menisci +2.1, PF OA +5.0.
  - OA crop fusion: Lateral OA rb +27.7, PF OA +13; the OA + menisci group +4.5 / +8.2. Bone crop fusion: Fracture +16.7.
  - Bare 3x3 with the label pair as main: all9 gold +2.4 (rb fragile), fluid group gold +6.6.
  - Member-level top-20 gate HURTS Effusion (-29 / -23).
- FRAGILE:
  - Every other per-class label-pair margin: ACL, MCL, menisci, PF OA (the 'loses on both folds' pattern), Lateral OA, Effusion,
    Baker's (one gold study carries all of the Baker's gain in every comparison), Fracture, Contusion.
  - The Effusion gain inside the 3x3 (rb +5.2): one gold-negative cell that every image reader calls positive (v1 'below', so not
    in the contested list) carries more than all of it. The label-free OOF evidence (lower correlation with Basin on report-silent /
    vague cells) stands; the gold gain does not.
  - Member-level top-20 gate on Lateral OA (+45 / rb +22): one study carries all of the rb gain.
  - Contusion crop fusion on rb (already undecidable).
- The Baker's artifact (Labels Work 01:3x; scripts/h13c_drop_bakers.py): one gold-positive Baker's cell (report general_normal, all
  image reads ~0.1) carries every Baker's label gain. Dropped on top of the contested cells (robust gold, fold mean): label pairs - seed
  pair all9 +1.78 -> +1.16; v4b + vag15 - seed pair all9 +4.18 -> +3.58 (holds), Baker's +5.4 -> 0.0; bare 3x3 grid all9 +1.09 ->
  +0.70, fluid +4.9 -> +3.1; 3x3 win all9 +0.45 -> +0.34.

### 16. What Fusion learned, ranked by evidence (closing; Master 10-16 ~01:00)

Evaluation rules that came out of this session (apply to every label / ensemble read):
- Baseline = a multi-seed v1 pool from the SAME CODE as the arms (for Labels Work's arms: tl_v1, tl_v1s1, ...), never one deployed
  seed. The deployed sub_b0 is a weak gold draw on fold 3; against it every label set looked better (16a). Old-code mains (sub_b0,
  sub_b0_s1) may join the pool for gold reads (code drift inside gold noise) but not for report reads (current code ~2.5 lower).
- Compare ensembles at equal N against seed ensembles drawn from that pool, and split any edge into members vs diversity.
- Every gold delta: with and without the contested cells, plus the leave-one-gold-study-out jackknife (section 15).

Solid (several folds, exact maps or image reads at scale; survive the jackknife where gold is involved):
1. The deployed window fusion gains by FILTERING: a hidden window-set restriction (Basin's slots: no axial non-fat-sat) plus the top-k
   gate. Value fusion, rank blends and consensus attention add ~0 (Baker's +1.6) [D1, D2, D4, D5; 5 folds; Codex at scale].
2. Post-hoc fusion inside the 0.961 / 0.962 package is saturated: >= 14 ways move it by <= 0.3. Report-fitted vendor offsets and
   cross-class stacking hurt [P1-P5; exact linear map, 3 folds].
3. A per-class gate opt-in on a coarse grid (k x blend) is the right granularity. Global gates lose (diffuse classes cannot opt out);
   learned / continuous heads do not beat the grid [D3, D6; 5-fold LOFO].
4. Models trained on image targets gain nothing from post-hoc gating or relevance changes (synovitis image heads, crop experts, oct13
   R7). Window-set levers must act in training (synovitis crops fat-sat-only +4.3) [D7b, E1].
5. The crop fusions hold on robust gold: OA (Lateral OA robust +27.7, PF OA +13) and Fracture +16.7; Contusion is undecidable
   [gf_experts, H13b].

Moderate (2-3 folds, one seed contrast, or mechanism evidence without a direct test):
6. Why filtering gains: report-trained mains read a report-specific signal from non-fat-sat series (e.g. trace effusions that reports
   mention and image reads deny). Its fitted weight is + on text and - on Codex; label sets with Codex-calibrated vague cells move the
   attention away from it [D10, D10f, S2].
7. Label effects live in the shared backbone features: re-fitted heads do not undo other classes' label changes, and a head-level
   label screen fails validation. Single-seed from-scratch label deltas do not replicate across folds [H2, H3, H6].
8. Label-diverse ensembles, against the full 4-member v1 seed pool on folds 3-4 (16a): v4b + vag15 beat it (all9 +5.2 gold / +2.9
   robust, not fragile; Medial OA +7); the AVERAGE label set does not (+1.4 / +0.5, fragile). Label sets robustly gain Medial OA and
   Synovitis and robustly lose menisci (-5, both folds). Label diversity adds about what seed diversity adds (f3 equal, f4 a bit more).
   Mains sharing a seed make correlated errors even with different labels (+1.7 / +1.1, not fragile): give each label set its own
   seed. Window attention is interchangeable between mains; the label effect is in the values [H4, H8, H9, H13].
9. Inside the bare 3x3 the label-main edge mostly dilutes (all9 gold +2.4, robust fragile); ligaments wash out. On Effusion the label
   main is less correlated with Basin on vague cells (label-free OOF), but the gold gain there hangs on one cell [H10, H13].

Weak or fragile (do not build on without image-read support):
10. Every per-class label pattern except Medial OA and Synovitis (PF OA loss, ACL gain, Lateral OA, Effusion); all Baker's label gains
    come from one gold cell (report general_normal, readers ~0.1) [H13, H13c].
11. A Lateral-OA-specific gate (member level: one study carries the robust gain); protocol-wise recipe weights (Lateral OA +0.70) [H9,
    P5b].

12. One backbone with label heads (H12 / H12b on Labels Work's tl_mh3: v1 + v4b + vag15 heads, averaged; folds 3-4, one seed each;
    jackknife; x1e-3 gold / robust): vs one v1 model (same cost) all9 +4.0 / +2.9 (gold not fragile, robust fragile), ligaments +5.4,
    Synovitis +90; vs a 3-seed v1 ensemble (3x the cost) all9 -0.3 / -1.5 (~equal), Synovitis +86, menisci -10.5 (not fragile); vs
    the same three label sets as separate models all9 -5.4 / -4.2 (NOT fragile), menisci -7.3 (not fragile). The fold-3 OA gain does
    not hold over two folds. So a multi-head backbone buys about a 3-seed ensemble's macro gain at one model's cost; separate label
    models stay clearly better (menisci). Shared features trade menisci for Synovitis (as in H6 / 16a); protecting menisci needs a
    meniscus-only member or a model whose other targets are v1. [moderate: 2 folds, one seed per fold]

Closed 10-16 06:5x. Nothing open in Fusion. G3 (fat-sat-only main) went to Synovitis V3. The partial-backbone screen was dropped
(code kept).

#### 16a. Correction after the full seed pool landed (03:05; tl_v1 / tl_v1s1 on fold 3)

Labels Work's own v1 mains (tl_v1 seed 0, tl_v1s1 seed 1: same trainer and labels as the label-set arms) give a 4-member seed pool on
fold 3 (sub_b0, sub_b0_s1, tl_v1, tl_v1s1). Scripts: h8_labelens.py / h8_report.py (out/h8_perclass.md), h8_singles_f3.py,
h8_pair_vs_seedpool_f3.py, h8_seedxlabel.py. x1e-3, fold 3, gold / gold_rb all9.
- The deployed sub_b0 is a weak draw on fold-3 gold: tl_v1 +5.5, tl_v1s1 +9.3, sub_b0_s1 +7.7 over it. Label sets (12, seed 0)
  average +6.1 over it, the same-trainer v1 pair +7.4: label sets - v1 = -1.2 / +0.6. So item 8's "better members" was mostly the
  baseline, not the labels. Per group, label sets beat v1 only on Synovitis (+74) and fluid (+9; Baker's part = one gold cell);
  they lose on menisci (-8.5) and ligaments (-4.9).
- Equal N, all label subsets vs all seed subsets (fold 3): N 2 +0.1 / +0.6, N 3 +0.3 / +0.6, N 4 +0.1 / +0.2. No label-ensemble edge
  on all9 gold. Gain from averaging (ensemble - own members): seeds +3.3 / +4.4 / +5.1, label sets +2.9 / +4.2 / +4.7 at N 2 / 3 / 4:
  seeds do not saturate earlier.
- One specific pair still beats seeds: v4b + vag15 vs the mean of the 6 seed pairs +4.9 / +3.9 (vs the same-trainer tl_v1 pair +3.4 /
  +3.4; vs the sub_b0 pair +7.0 / +4.9). Groups vs the seed pairs: OA +10.8, ligaments +7.5, fluid +5.0, menisci -6.5, Synovitis +69.
- Seed x label replicates with the same-trainer pair: (tl_v1s1 + X) - (tl_v1 + X) diversity +1.9 (10 / 12) / rb +2.3 (11 / 12). Two
  independent seed contrasts on fold 3 + one on fold 4 now agree: give each label set its own seed.
- Fold 4 (05:30, tl_v1 / tl_v1s1 f4 landed; h8_seedpool.py, h13_jackknife.py E / F), fold mean of f3-4, x1e-3, gold / gold_rb:
  - Label sets vs the 4-member v1 pool: all9 +0.9 / 0.0; vs the same-trainer pair -1.2 / -1.4. The average label set = v1.
  - All label pairs vs all 6 seed pairs: all9 +1.4 / +0.5 (FRAGILE). Not fragile: Medial OA +5.8 / +6.2 (gain, both folds), menisci
    -5.0 (loss, both folds; jackknife -6.1..-3.4), Synovitis +69. Fluid +8.2 / +5.8 is fragile (Baker's = the one-study artifact).
  - v4b + vag15 vs the 6 seed pairs: all9 +5.2 / +2.9, NOT fragile (jackknife 3.9..6.1 / 1.9..3.8); Medial OA +7.0 / +7.6; vs the
    same-trainer tl_v1 pair +2.9 / +1.6 (f4 robust ~0).
  - Gain from averaging: f4 label ensembles gain more than seeds (N 2 / 3 / 4: +4.3 / +6.0 / +7.0 vs +2.8 / +4.3 / +4.4), f3 equal:
    mixed, no general rule.
  - Labels Work's label-level look at the menisci loss (06:0x): sets whose meniscus targets equal v1 (vsf, vag15) lose MORE meniscus
    gold (MM -8.6, LM -3.9) than sets that change meniscus targets (MM -5.1, LM -2.0). So the cost comes through the shared backbone
    from other classes' targets (as in H6), alongside the Medial OA gain. Hand-off (Labels Work HANDOFF section 1, Master's fix
    05:55): v1 meniscus targets do NOT protect menisci; protection = a meniscus-only member, or menisci in a model whose OTHER targets
    are v1. Read meniscus gold in every OA-label arm.
- Labels Work's additions (03:2x; experiments/v3/oct15_labels/scripts/rebase_read.py, out/rebase_read_f3.txt):
  - Code drift on fold 3: at the same seeds, the current trainer reads ~2.5 lower than the old code on report reads (trusted_ns -2.9
    vs an old-code spread of 1.2; text -1.8); gold is inside noise (+2.5 vs 7.7); Codex mixed. The f4 pair checks it.
  - Re-baselined on the 4-member pool, gold9 f3: v5 +6.6, consnr +6.3, consnp +5.9, v4b +5.3, consv3 / vsf +4.2, vag15 +3.8, rules
    -14.3; the 13 sets average ~0. Pool members span -5.6..+3.6 around the pool mean, so single-set gold edges of +4..+7 are about one
    seed's spread.
Revised ranking: item 8 becomes "specific label sets (v4b + vag15) beat v1 seeds by ~+3-5 gold9 at member level (fold 3; fold 4 pending a
proper seed pool); the average label set does not; label diversity is worth what seed diversity is worth; vary seeds across sets".
