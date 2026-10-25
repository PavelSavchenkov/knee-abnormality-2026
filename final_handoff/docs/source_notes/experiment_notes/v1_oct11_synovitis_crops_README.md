# oct11_synovitis_crops: synovitis crop experts on pipeline #1

Started 2026-10-11 19:15 (user goal, user_directives.md; 24 h). Time-ordered results: ledger.md; ideas and priorities: ideas.md.

## FINAL (2026-10-12 19:20): Synovitis G, submissions/oct12_syn_g (57112379), LB 0.955 and above the 0.955 entry (new best)

G = equal logit mean of 6 members, folds 0-2: image heads m_prob05 (effb0) + n_prob05 (NFNet-L0) and crop experts s_fs + s_fc125
on effb0 and NFNet-L0 (no R50). In the 0.955 package (only Synovitis changed) it ranks above oct11_oa_3pipe by hidden precision:
team LB 71/74 -> 54/83 inside the 0.955 block, passing 22 teams with unchanged scores (~+0.004 synovitis AUC if evenly spread;
bounds 0..+0.011).

| folds 0-2, x1e-3 | reads | clear-cut | gold-58 |
|---|---|---|---|
| G vs m_prob05 (the 0.955's Synovitis) | +15.3 [+10.9,+19.9] | +5.3 [+2.7,+8.5] | +10.8 n.s. |
| G vs 4 specialists (2 seeds + NFNet + R50 image heads) | +9.0 [+5.1,+13.0] | +3.3 [+0.6,+6.3] | +10.8 n.s. |
| G vs 6 specialists (+ m_prob15 + m_cal05) | +8.0 [+4.3,+11.9] | +2.9 [+0.6,+5.7] | +7.2 n.s. |

Specialist ensembles saturate (0.916 -> 0.920 -> 0.921), so the gain is not a member-count effect. Open: G vs a size-matched
specialist ensemble on the LB (one submission). Caveat: no local metric reproduced the synovitis LB order; LB synovitis
comparisons are ~+-0.01 AUC noisy.

## 3-pipeline entry (2026-10-12 12:50): submissions/oct12_syn3pipe, submission 57106261: LB 0.954 (< 0.955)

Only Synovitis differs from the 0.955 entry, so the crop experts' public synovitis AUC is lower, but the size is open. Public LB
14:22: our 0.955 is 71st of the 74 teams at 0.955 (rank 198 of 5,828; 717 teams at 0.954), so its true score is near 0.9545. The
gap to the 0.954 entry is then (0, ~0.001) LB = a synovitis loss of (0, ~0.013) AUC: a near tie is possible. Leaving R50 out changes
the local numbers by less than the seed floor (effb0 + NFNet: +15.2 reads / +5.0 clear-cut vs +14.8 / +6.4 with R50), so R50
cannot be named as the cause.

The UI sort uses hidden precision (user 06:50): P6 > P4 > the specialist pair, so both crop-expert pairs beat it by < +0.012 synovitis
AUC each, and the order matches the local reads / clear-cut ranking (gold-58 misled: it favoured P4). User 07:00: extend the best
design (P6 = s_fs + s_fc125) to the 3 pipelines, 0.955-style. Done: 18 models (2 designs x effb0 / NFNet-L0 / RadImageNet R50 x
folds 0-2); entry = the LB 0.955 package with Synovitis = mean of the 3 pipelines' crop-expert logits.
| folds 0-2 vs m_prob05 (= the 0.955 package's Synovitis), x1e-3 | reads | clear-cut | gold-58 |
|---|---|---|---|
| 3-pipeline crop experts (the entry) | 0.9286 (+14.8 [+9.6,+20.5]) | 0.9941 (+6.4 [+3.1,+10.3]) | 0.877 (+2.4) |
| effb0 P6 / NFNet pair / R50 pair | +14.1 / +12.4 / +6.2 | +4.6 / +4.3 / +6.2 | 0.871 / 0.885 / 0.860 |
| 3-pipeline specialist image heads | +6.0 | +1.8 | 0.875 |
Reading: 12 x (LB - 0.955) = Synovitis change (+-0.012) + the hidden-precision sort order vs the 0.955 entry.

## LB RESULT (2026-10-12 01:57): both crop-expert candidates TIE the specialist pair (at 3 decimals; by hidden precision P6 > P4 > base)

| synovitis-only probe (6 models: 2 x folds 0-2) | public LB | public synovitis AUC | local reads vs the pair |
|---|---|---|---|
| baseline m_prob05 + m_prob05_s1 (submissions/oct11_synbase_pair) | 0.536 | 0.932 +- 0.006 | |
| P4 s_fs + s_codex (submissions/oct12_syn_p4) | 0.536 | 0.932 | +11.1 |
| P6 s_fs + s_fc125 (submissions/oct12_syn_p6) | 0.536 | 0.932 | +11.5 |
The crop experts are locally better on held-out Codex reads but not visibly better on the public test (each difference in
(-0.012, +0.012)). The 3-pipeline crop-expert extension was stopped (no measurable gain expected in a 12-finding package:
+0.006 synovitis = +0.0005 LB); the 0.952 package's synovitis (3-pipeline image heads) stays. GPU handed to Fracture Contusion.

## Key learning: local reads vs public synovitis (all synovitis LB evidence so far)

| synovitis from | held-out reads (folds 0-2) | gold-58 | public synovitis | source |
|---|---|---|---|---|
| R2 + effusion mix (0.940) | 0.870 | 0.784 | 0.872 (probe) | oct7 probe |
| imghead3 (700 reads, block rank) | 0.900 | 0.859 | ~0.92 (0.944 - 0.940) | oct8 |
| w05 (all reads, block rank) | 0.905 | 0.877 | ~0.93 (+0.012 vs ih3) | oct10 probes |
| prob05 = m_prob05 (raw read prob) | 0.914 | 0.875 | ~0.93 (= w05) | oct10 probes |
| m_prob05 + m_prob05_s1 (baseline) | 0.916 | 0.875 | 0.932 +- 0.006 | oct11_synbase_pair |
| P4 s_fs + s_codex | 0.928 | 0.883 | 0.932 | oct12_syn_p4 |
| P6 s_fs + s_fc125 | 0.928 | 0.871 | 0.932 | oct12_syn_p6 |
Public synovitis rose with held-out reads up to ~0.915 (label fixes: more reads, raw targets) and then flat-lined at ~0.932 while
reads kept rising to 0.928 (model / crop changes); gold-58 stayed at ~0.87-0.88 throughout. Reading: beyond ~0.915 the held-out
Codex reads measure agreement with the reader's own noise (training-style re-reads score 0.81 on gold), not with the host's labels;
further model work on the same targets is unlikely to move the public synovitis. The next lever is the supervision (new / better
reads, host-like labels, external synovitis grades), not the model.

## PRELIMINARY FINAL (2026-10-11 23:55): synovitis approach for pipeline #1

**Synovitis = probability mean of two effb0 crop-expert designs, folds 0-2 each (6 models), image-read output alone:**
1. **s_fs**: main/syn_train.py `fs_only=True` (other settings default): localiser v2 bags pouch (suprapatellar, 80 mm), hoffa
   (72 mm), post (popliteal, 100 mm), all planes, fat-suppressed windows per bag (non-FS only where a bag has no FS window),
   central 80% of present slices, k 8 train / 32 eval windows per bag, 224 px, attention pooling over all bags; outputs img_syn
   (raw Codex read probability, w 1) + aux read effusion / text synovitis / effusion / Baker's (w 0.5); 20 epochs, epoch 19.
2. **s_codex**: reader-aligned bags = the 4 views the Codex reads came from: `bags=pouch,hoffa,post,notch,gutter fs_only=True
   planes=Sagittal,Axial k=5`, evaluated with k_eval 64 (all windows; equal quality, order-independent).
Test-time path: deploy/syn_stage.py (DICOM -> mirror detector -> localiser v2 -> bags -> crops -> experts), verified on gold-58.

| folds 0-2 pooled (x1e-3 paired bootstrap) | reads | clear-cut | gold-58 |
|---|---|---|---|
| specialist pair m_prob05 + m_prob05_s1 (= the LB 0.536 baseline probe: public synovitis 0.932) | 0.9164 | 0.9886 | 0.8746 |
| **s_fs + s_codex (P4)** | **0.9277 (+10.9 [+5.3,+16.5])** | 0.9907 (+1.7) | **0.8829-0.8853** |
| single: m_prob05 / s_fs / s_codex | 0.9138 / 0.9256 / 0.9188 | 0.9877 / 0.9921 / 0.9845 | 0.875 / 0.857 / 0.879 |
Within-site gain (site-centred AUC) +11.3 = the pooled gain. Design pair +6.1 over one expert vs an expert seed pair +1.7.

What did not help (one fold, one seed unless noted; vs the base of the time): per-plane bags -5.6; 12 epochs -6.8; dropout 0.4
-7.0; Codex attribute heads (2 designs) -10.0 / +1.2 (gold -17 / -23); per-bag readout -6.2 / -16.4; coronal back into the reader set
-15.9; specialist-encoder init -7.6; clamp -2.3; no erasing -0.9; whole-knee bag -3.9; text-heavy loss (w_img 0.25) -14.6; crop
1.25 +0.8 reads / -4.3 clear-cut / +45 gold (weak lead); k_eval 64 neutral (3 folds); epoch averaging neutral.

Kaggle: one probe planned (P4, synovitis-only, 2 x 3 fold models, reads 12 x (LB - 0.536)); needs the user's request (quota
after 01:00 BST); details probes.md.

## Question

Can a synovitis expert trained on localiser crops of the synovitis regions (OA-session idea: compartment-crop experts, LB +0.048 OA
sum) with the Codex image-read targets (Synovitis-session idea: image head on the raw read probability, LB 0.952) beat the current
pipeline-1 synovitis specialist (m_prob05), alone or blended with it, apples to apples (a 2-member blend vs m_prob05 + its seed
replicate)?

## Metric (scripts/compare.py)

Folds 0-2 pooled. Held-out reads: AUC vs the Codex read (p >= 0.5) on validation-fold studies (2,150 with a read, never trained on
in that fold); clear-cut reads (p >= 0.85 vs <= 0.1: less reader noise); gold-58 synovitis test-like (mean logit of the 3 fold
models; 27 positives, SE ~0.06: veto only). Paired study bootstrap vs m_prob05; noise floors: seed replicate (single) and
m_prob05 + m_prob05_s1 (pair). Reads alone reward copying the reader: a gain needs clear-cut reads up and gold not down.

## Layout

- main/: syn_data.py + syn_train.py (crop expert), train.py / data.py / model.py / labels.py (main pipeline = m_prob05 code).
- loc/: localiser v2 predictions for every slice (pred/), weights (copied from oct10_synovitis/loc, originally oct10_oa).
- data/: Codex export snapshot, img_syn_prob.csv (m_prob05 target), img_attrs.csv (read attributes, scripts/make_attrs.py).
- refs/: predictions of m_prob05, m_prob05_s1 and other oct10_synovitis runs (for self-contained comparisons).
- scripts/: run_queue.sh (queue.txt), watch.sh, compare.py, eval_syn.py / blend_syn.py / paired.py (copies), bag_viz.py, make_attrs.py.
- runs/, logs/, out/.

## Results (folds 0-2 pooled; x1e-3 paired study bootstrap vs m_prob05; details ledger.md; probes probes.md)

| synovitis from | reads | clear-cut | gold-58 | d reads | d clear-cut | d gold |
|---|---|---|---|---|---|---|
| m_prob05 (specialist; LB 0.945 on the 0.944 base) | 0.9138 | 0.9877 | 0.8746 | | | |
| m_prob05_s1 (seed floor) | 0.9143 | 0.9883 | 0.8710 | +0.5 [-3.9,+4.8] | +0.6 | -3.6 |
| s_base crop expert (pouch / hoffa / post, all series) | 0.9213 | 0.9926 | 0.8614 | +7.4 [+2.1,+12.8] | +4.9 [+1.6,+9.0] | -13 |
| **s_fs** (same, FS windows per bag) | **0.9256** | 0.9921 | 0.8566 | **+11.8 [+5.1,+17.8]** | +4.4 [+0.7,+8.5] | -18 |
| s_codex (reader-aligned: FS, sag + ax, + notch / gutter) | 0.9188 | 0.9845 | 0.8793 | +5.0 | -3.2 | +4.8 |
| m_prob05 + m_prob05_s1 (pair floor) | 0.9164 | 0.9886 | 0.8746 | +2.6 | +0.9 | 0 |
| m_prob05 + s_base | 0.9213 | 0.9916 | 0.8746 | +7.5 | +3.9 | 0 |
| **s_fs + s_codex (probe candidate P4)** | **0.9274** | 0.9903 | **0.8853** | **+13.5 [+7.4,+19.6]** (vs pair +10.9 [+5.3,+16.5]) | +2.6 | +10.8 |

Screens (one fold unless noted, vs the base on that fold; expert seed noise ~2.5 reads per fold): per-plane bags -5.6 (f0);
12 epochs -6.8 (f1); k_eval 64 neutral (3 folds); torch.compile ~5% faster only.
Why the OA-style windows still work (eval-time ablations, attention): the model down-weights useless windows itself (attention
pouch 0.53 / hoffa 0.35 / post 0.12); coronal and non-FS windows add nothing at test time; FS-only training moves the budget to
useful windows and makes the Hoffa bag 3x more important. Error review (ledger 20:30): misses are Hoffa-signal positives with
small effusions; false alarms small effusions with poor fat suppression; gold false positives large simple effusions.
