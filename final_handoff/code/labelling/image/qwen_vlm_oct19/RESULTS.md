# Open-VLM image labelling pilot (Master, 10-19): results

USER (verbatim): "also, consider investigating open LLMs which fit on 5090 for inference and using them for image-based data labelling
(you can label several hundreds of images, then double check if labels look sane, then try to train something on them, etc),
investigate which class needs it the most. you can choose to LLM-label certain series, certain frames, certain zooms, etc -- test what
works best"

Setup:
- Pilot set out/pilot_studies.csv: 299 non-gold train studies with a Codex synovitis read and a shipped 0.961 OOF row (folds 0-2);
  200 random + 99 enriched for rare text positives. Never tuned on gold; gold-58 (out/gold_studies.csv) gets one check at the end.
- Views: the image-labelling bundle's contact sheets (views/<index>/, render_views.py).
- Read: score.py = one forward pass per (study, finding). score = P(Yes) / (P(Yes) + P(No)) at the first answer token, fp32 head,
  thinking off. Prompt = INTRO + sheets + the host definition (+ optionally the Codex prompt's reading guide).
- References (evaluate.py):
  - text = text v1 probability >= 0.5;
  - trusted = hand-checked text cells;
  - codex = Codex image reads >= 0.5 (Effusion / Synovitis);
  - the 0.961 OOF on the same studies, for scale.
- OAI: no (competition images only).

## Pass 1: Qwen3.5-9B bf16, viewset A (2 sheets per finding), definition only (16:16-16:42, 3,588 reads, 0.42 s / read, 18 GB)

| finding | VLM vs text | OOF vs text | VLM vs trusted | VLM vs Codex | OOF vs Codex | p sd |
|---|---|---|---|---|---|---|
| Effusion | 0.740 | 0.859 | 0.851 | 0.834 | 0.960 | 0.084 |
| Baker's | 0.748 | 0.962 | 0.935 | | | 0.058 |
| Lateral OA | 0.612 | 0.898 | | | | 0.072 |
| Contusion | 0.597 | 0.917 | | | | 0.123 |
| Fracture | 0.549 | 0.797 | | | | 0.043 |
| PF OA | 0.548 | 0.880 | 0.597 | | | 0.067 |
| Synovitis | 0.544 | 0.732 | | 0.703 | 0.916 | 0.060 |
| ACL / MCL / MM / LM / MOA | 0.49-0.51 | 0.84-0.95 | 0.49-0.58 | | | 0.06-0.10 |

Reading:
- The zero-shot 9B sees fluid: Effusion 0.83 vs Codex, Baker's 0.94 vs trusted.
- It sees almost nothing focal: ligaments, menisci and cartilage are at chance.
- Synovitis, the target class, is only 0.70 vs Codex.
- Scores are compressed (sd 0.04-0.12) and yes-biased (mean P(yes) 0.6-0.9 for most findings).
- As a labeller this is far below Codex and below the trained models on every column. Not usable as a training target yet.

Sanity check, by eye (sanity_pairs.py, Effusion pass 1, 1_sag_fs, 3 highest / 3 lowest VLM scores):
- Highest #38 / #129: large suprapatellar effusions (Codex 0.95, text 0.97). Correct.
- Highest #67: a large posterior fluid collection, i.e. a Baker's cyst, with no joint distension (Codex 0.15). The VLM confuses Baker's
  cyst fluid with effusion.
- Lowest #93 / #282 / #192: no distension (one Baker's cyst, one normal, one noisy scan). Correct.
- The extremes are sane. The middle carries the error, and scores are compressed.

Add-on to the 0.961 OOF on the same studies (addon_check.py: z(OOF) + w * z(logit VLM)). INFO only: in-sample over a 5-point grid,
pilot n, no selection correction.
- Synovitis vs Codex: 0.9157 -> 0.9208 at w 0.25 (+5.1e-3).
- Effusion vs Codex: 0.9599 -> 0.9615 at w 0.25 (+1.6e-3).
- Baker's vs trusted: 0.9968 -> 0.9977.
- ACL / menisci / MOA / PF: flat or worse.
- So the fluid reads carry some information the package lacks, toward the Codex reader. This is weak evidence; it says nothing about
  gold or the LB.

## Which class needs image labels most (Brainstorm 10-19 gold-58 aggregates + the pilot)

| class | need (why labels limit it) | reader feasibility (pilot pass 1) | verdict |
|---|---|---|---|
| Synovitis | highest: 79 % of reports silent; Codex reads lifted public ~0.87 -> ~0.93 (+0.012 more from raw probabilities); Codex gold 0.878 vs text 0.672; one read per study (189 second reads) | 9B zero-shot 0.70 vs Codex (target needs >= Codex-level reads, or independent second reads to average) | target class; needs a better reader config |
| Baker's | small: public ~0.99; Codex covers 516 studies in one text cell; 38 % reports silent | 0.94 vs trusted | second; Q25 (pw toward Codex pairs) LB read tonight decides whether image reads transfer |
| Effusion | low: model 0.980 > Codex 0.891 on gold; VLM Effusion member closed (< 1e-4) | 0.83 vs Codex | no |
| Fracture | headroom #3, but host convention + few positives | 0.55 vs text | no, unless a gold check shows the reader sees fractures |
| LOA / PF / Contusion / menisci / ligaments | label- or convention-limited; Codex LOA panel failed feasibility | chance to 0.61 | no |

Next: improve the Synovitis reader. Then label the pool only if the best config reaches about the Codex level on the pilot and on the
one gold-58 check.

## Pass 2: Qwen3.5-9B, viewset A + the Codex reading guide, 7 findings (16:55-17:10)

| finding | pass 2 VLM | pass 1 VLM | 0.961 OOF |
|---|---|---|---|
| Synovitis vs Codex | 0.784 | 0.703 | 0.916 |
| Effusion vs Codex | 0.861 | 0.834 | 0.960 |
| Baker's vs trusted | 0.948 | 0.935 | 0.997 |
| Baker's vs text | 0.806 | 0.748 | 0.962 |
| LOA / PF / Contusion / Fracture vs text | 0.50-0.60 | 0.55-0.61 | 0.80-0.92 |

The guide helps the fluid findings. It does not help the focal / cartilage ones. (The live LOA 0.85 at n=63 was noise.)

## Blend with the 0.961 OOF, out of sample (blend_check.py)

Method: w picked on 2 OOF folds and applied to the 3rd, 200 permuted-VLM nulls; gains x1e-3 AUC.

| VLM read | Synovitis vs Codex: cv gain (null p95, p) | Effusion vs Codex | Baker's vs text |
|---|---|---|---|
| pass 1 alone | +0.6 (1.8, 0.065) | -0.8 | 0.0 |
| pass 2 alone | +0.7 (0.9, 0.060) | +1.3 (0.7, 0.035) | +0.6 (0.4, 0.040) |
| mean of pass 1 + 2 (prompt ensemble) | **+3.5 (1.7, 0.025)** | +1.0 (0.2, 0.030) | +0.1 |

- First signal above its null: averaging two prompt variants of the same model gives Synovitis +3.5e-3 toward Codex, out of sample.
- Caveats:
  - the reference is Codex, i.e. the same reader family the Synovitis models are trained on;
  - vs text, the blend is 0 / negative (text is the weakest Synovitis reference);
  - gold is not read yet (one check at the end).
- Every further config (frames, Codex sheets, few-shot, Lingshu, 27B, thinking) joins the average as it lands.

## Passes 3-4 (17:10-17:32) and blends with more configs

| config | Synovitis vs Codex | Effusion vs Codex | Baker's vs trusted |
|---|---|---|---|
| pass 3: 9B single frames at 2x (F) + guide | 0.629 | 0.793 | 0.953 |
| pass 4: Lingshu-7B (medical, Qwen2.5-VL base), A + guide | 0.71 | 0.65 | ~0.93 |

Cross-fitted blend gains (x1e-3, Synovitis vs Codex; null p95; p):
- passes 1 + 2: +3.5 (1.7; 0.025);
- 1 + 2 + 3: +2.4 (1.6; 0.03);
- 1 + 2 + Lingshu: +1.2 (2.2; 0.07);
- 1 + 2 + 3 + Lingshu: +2.7 (2.1; 0.04).

Effusion: +0.8..+1.3 for the 9B-only averages; Lingshu removes it.

Reading:
- The weaker readers (single frames for Synovitis; Lingshu overall) dilute the average.
- Picking a subset on these 299 studies is selection.
- So the subset is tested on an independent set.

PRE-REGISTRATION (17:3x, before any data of the bak_* passes): REPLICATION of the Synovitis / Effusion blend on the 516 Codex Baker's
studies.
- Studies: the 247 with a shipped OOF (f0-2) and a Codex Synovitis read, not in the pilot. They are a selected cell (text v1 Baker's below
  / unsettled), not random.
- VLM = mean of z(logit) of the two pre-chosen configs: bak_q9b_A (= pass 1 config) and bak_q9b_Ag (= pass 2 config).
- Read: blend_check.py as on the pilot (w picked on 2 OOF folds, applied to the 3rd, 200 permuted nulls).
- Replicates if: Synovitis vs Codex cross-fitted gain > its null p95, and the fixed-w 0.25 gain > 0.
- Effusion and Baker's (vs the 516 Codex Baker's reads) are reported as INFO.

## REPLICATION (pre-registered 17:3x) on 247 new studies: FAILED for Synovitis / Effusion

Set: the 516 Codex Baker's studies minus the pilot, with a shipped OOF (f0-2) and a Codex Synovitis read. It is a selected cell (text
Baker's below / unsettled).

| VLM read | Synovitis vs Codex: cv gain (null p95, p); fixed w 0.25 [90 %] | Effusion vs Codex | Baker's vs Codex Baker's |
|---|---|---|---|
| mean of the pass-1 + pass-2 configs (the pre-registered read) | **-1.9 (1.6, 0.72); -3.4 [-14.3, +6.8]** | 0.0 (2.7); -7.4 | +0.1 (0.0, 0.03); +0.4 |
| pass-2 config alone | 0.0; -8.2 | -1.2 | +0.7 (0.0, 0.025); +1.9 [-4.3, +8.5] |
| pass-1 config alone | -0.1; -2.0 | 0.0; -9.9 | 0.0 |

Reader quality on this set:
- Synovitis 0.67-0.72 vs Codex (OOF 0.888);
- Effusion 0.77-0.80 (OOF 0.934);
- Baker's 0.79-0.83 vs Codex Baker's (OOF 0.964).

Verdict:
- The pilot's Synovitis blend gain (+3.5, p 0.025) does NOT replicate. It was most likely chance plus the choice among configs on the
  same 299 studies.
- Zero-shot Qwen3.5-9B reads, alone or averaged, add no out-of-sample signal to the 0.961 package for Synovitis / Effusion.
- Baker's: a tiny gain above its null (+0.1..+0.7e-3), inside a wide interval. Not usable.
- Strength: 2 sets (299 pilot + 247 replication), 3 configs of one 9B model + Lingshu-7B, CPU post-hoc, Codex / trusted / text
  references; gold not read.
- Still open:
  - a stronger reader (Qwen3.5-27B, 4-bit), few-shot and thinking, on the pilot, then a replication on the same 247 if any one reaches
    a clearly higher reader AUC;
  - fine-tuning (oct14_vlm sealed; the VLM Effusion member was closed by the user).

PRE-REGISTRATION (18:1x, at n=186 of the q27b_Ag pilot pass, before its end and before any 27B read of the replication set):
- Gate: bak_q27b_Ag runs only if q27b_Ag reaches Synovitis >= 0.82 vs Codex on the full 299-study pilot (9B best was 0.784).
- Replication read = blend_check.py on the 247 new studies, VLM = q27b_Ag config ALONE (no averaging, no config choice).
- Replicates iff Synovitis vs Codex cross-fitted gain > its null p95 AND fixed w 0.25 > 0.
- Effusion / Baker's: INFO.

## Pass 5: Qwen3.5-27B 4-bit (nf4), A + guide (17:58-18:15, ~3 s / study, 18.8 GB)

| finding | 27B | 9B best |
|---|---|---|
| Synovitis vs Codex | 0.807 | 0.784 |
| Effusion vs Codex | 0.827 | 0.861 |
| Baker's vs trusted / text | 0.965 / 0.865 | 0.948 / 0.806 |

- Pre-registered gate (Synovitis >= 0.82 vs Codex): FAILED -> bak_q27b_Ag cancelled, as pre-registered.
- Pilot blend with the OOF: Synovitis cv -1.3 (null p95 0.1), Effusion -0.1, Baker's text -0.2. No signal even in-sample on the pilot.
- A bigger model reads Baker's a bit better, not Synovitis.

PRE-REGISTRATION (18:2x, at n=53 of q9b_Ag_shot, before its end): the same gate and replication for the few-shot config.
- Gate: Synovitis >= 0.82 vs Codex on the full pilot.
- If the gate passes: bak_q9b_Ag_shot on the 516 set, read with blend_check on the 247 new studies, VLM = this config alone.
- Replicates iff Synovitis cv gain > null p95 AND fixed w 0.25 > 0.
- The few-shot example studies (out/shots_studies.csv) are checked to be disjoint from the replication set.
- Overlap check: 2 of the 8 shot example studies are in the 516 set -> they are EXCLUDED from the replication read (with the pilot studies).

## Pass 6: few-shot (2 + 2 Codex-clear example studies per finding in the prompt), 9B, A + guide (18:14-18:36, ~4.3 s / study)

| finding | few-shot | best zero-shot |
|---|---|---|
| Synovitis vs Codex | **0.847** | 0.784 |
| Effusion vs Codex | **0.896** | 0.861 |
| Effusion vs trusted | 0.892 | 0.885 |

- Gate (Synovitis >= 0.82): PASSED, so bak_q9b_Ag_shot (the replication) started 18:36.
- Pilot blend (INFO): Effusion vs Codex cv +11.3 (null p95 0.7, p 0.000), fixed w 0.25 +7.3 [+3.5, +11.6]; Synovitis vs Codex +1.2
  (0.3, 0.025), fixed +6.3 [-3.2, +15.2]. vs text: Effusion -8.5, Synovitis fixed -14.9.
- CAVEAT (label provenance): the example studies were picked by Codex reads, so agreement with Codex is partly FITTED-TO; the replication
  shares that reference. The independent check is gold-58.

PRE-REGISTRATION (18:4x): ONE gold-58 check, nothing tuned on it. gold_check.py on gold_q9b_Ag_shot (and gold_q9b_Ag for reference).
It reports:
- VLM-alone vs Codex-alone gold AUC with a bootstrap interval (is the VLM a Codex-level second reader?);
- package (0.961 expected gold; an approximation for the cnx Synovitis form) + 0.25 / 0.5 VLM;
- Codex + VLM (the two-reader label idea).
Gold-58 is small and noisy (Synovitis n+ ~20): reported with intervals, no pick rule.

PRE-REGISTRATION (19:0x, USER: "can we give gold examples as few shot examples instead (logging which were used as reference and exclude
them from gold measurement of this labelling approach (i.e. cross fit on gold))? and then if its better than codex-led-few-shot or can be
blended, etc"):
- Design (gold_shots.py, seed 19):
  - gold-58 -> 3 folds stratified by Synovitis (20 / 19 / 19);
  - per fold k and draw d (2 draws): 2 gold positives + 2 gold negatives per finding from the other folds;
  - fold k is read with those examples (passes gshot_k{k}_d{d}).
  - Every study is read only with examples that are not itself. 32 distinct gold studies serve as examples somewhere
    (out/gshot_examples.csv).
- Read (gold_check.py on the 58 studies; one file per draw = the 3 folds concatenated):
  (a) gold-shot alone: draw 0, draw 1 and the mean of both;
  (b) Codex-shot (gold_q9b_Ag_shot) and zero-shot (gold_q9b_Ag) on the same gold, plus Codex's gold reads;
  (c) blends: package + VLM, Codex + VLM, gold-shot + Codex-shot.
- Bootstrap 90 % intervals; gold-58 is small (Synovitis n+ 27), so no pick rule.
- CONSEQUENCE, stated before data:
  - if gold-shot examples are later used to label the training pool, any gold read of a model trained on those labels excludes the
    example studies;
  - the LB is unaffected.

## REPLICATION of the few-shot config (pre-registered 18:2x) on 472 new studies (245 with OOF + Codex): FAILED

- Reader on this set: Synovitis 0.805 vs Codex (OOF 0.887; zero-shot 0.67-0.72 here); Effusion 0.845 (OOF 0.933); Effusion vs trusted
  0.959.
- Blend: Synovitis vs Codex cv -2.1 (null p95 2.1, p 0.735), fixed w 0.25 +1.0 [-8.2, +9.4]; Effusion cv -1.4 (p 0.695), fixed +0.6.
- The pilot's Effusion +11.3 / Synovitis +1.2 do not replicate.
- Verdict: with or without few-shot, zero-shot open-VLM reads add no out-of-sample signal to the 0.961 package as a test-time member, on
  Synovitis / Effusion (3 zero-shot configs, few-shot, 27B, Lingshu; 2 study sets).
- Open: the LABEL route. Is the few-shot VLM a usable second reader next to Codex for the training target? The gold-58 checks (Codex-shot,
  zero-shot, gold-shot cross-fit) run now.

PRE-REGISTRATION (19:2x, USER: "maybe we need to run VLM labelling on some zoomed images, because findings are focal???"):
- Viewset Z2 = the zoom members' own localised crops (render_zoom.py: syn_data.build_bags notch40 / mclz / medc40 / latc40 / pfl40, FS
  windows, middle slice, 448 px, ~0.09 mm / px, 2 windows per bag x plane).
- Per finding, 2-3 crops: ACL notch sag x2 + cor; MCL coronal x2; menisci sag x2 + cor; OA cor x2 + sag; PF sag + axial x2.
- Passes on the 299 pilot studies: z_q9b_Z2 (zero-shot) and z_q9b_Z2_shot (few-shot, 2 + 2 report-explicit examples: v1 'meets' p >= .85
  vs 'denied' / 'structure_normal' p <= .1, non-gold, non-pilot; out/zshots.csv). Definition-only prompt (the Codex guide cites sheet tile
  ids).
- Read:
  - reader AUC vs text v1 and trusted cells, next to the contact-sheet pass 1 (q9b_A) on the same findings;
  - the gate for anything further = reader >= 0.80 vs text or trusted on a column;
  - blend_check as INFO.

## Gold-58 check (pre-registered, one read; gold_check.py; Synovitis n+ 27 / 58, Effusion 35 / 58)

| read (VLM = Qwen3.5-9B, A + guide) | Synovitis VLM | Effusion VLM | Syn pkg + 0.25 VLM | Eff pkg + 0.25 VLM | Syn Codex + VLM | Eff Codex + VLM |
|---|---|---|---|---|---|---|
| reference: Codex alone / package alone | Codex 0.878 / pkg 0.886 | Codex 0.891 / pkg 0.980 | | | | |
| zero-shot | 0.551 | 0.860 | 0.821 | **0.991** [-0.000, +0.031] | 0.738 | **0.974** [+0.029, +0.147] |
| Codex-picked few-shot | 0.680 | 0.820 | 0.864 [-0.057, +0.006] | 0.976 | 0.812 | 0.938 [+0.002, +0.099] |
| gold-example few-shot, cross-fit, draw 0 / 1 / mean | 0.636 / 0.597 / 0.634 | 0.860 / 0.856 / 0.857 | 0.849 | 0.979 | 0.796 | 0.947 [+0.010, +0.110] |
| all few-shot configs averaged | 0.654 | 0.852 | 0.857 | 0.979 | 0.803 | 0.952 |

Bracketed: bootstrap 90 % intervals of the difference vs the package / vs Codex alone.

Verdict:
- Synovitis: the VLM is far below Codex on gold.
  - VLM - Codex 90 % interval entirely < 0 for every config, e.g. Codex-shot [-0.33, -0.09].
  - Adding it to the package or to Codex lowers gold.
  - The LABEL route for Synovitis is CLOSED.
  - Gold examples in the prompt do not beat Codex-picked ones (0.60-0.64 vs 0.68).
- Effusion: the VLM is a complementary second reader to Codex on gold (Codex + VLM +0.05..+0.08, interval > 0).
  - The zero-shot read added to the package reaches 0.991 vs 0.980 on gold, but the non-gold replication found no gain.
  - Effusion's headroom is small, and the user closed the VLM Effusion member on 10-19 (gain < 1e-4).
  - So it is not pursued; INFO for the final assembly.
- Strength: gold-58 (small, and SELECTION for the package), 4 configs of one 9B model, one read.

## Zoom pass, zero-shot (z_q9b_Z2, 19:26-19:30): still at chance on the focal findings

vs text:
- ACL 0.50, MM 0.40, LM 0.43 (below chance: says 'yes' more on normal menisci);
- MOA 0.54, LOA 0.54, PF 0.49;
- MCL 0.68 (contact sheets: 0.49).
Trusted: ACL 0.49, menisci 0.43 / 0.52, MOA 0.52, PF 0.53. Few-shot zoom pending.

## Effusion LABEL route re-opened (USER 19:3x: "ignore what i closed earlier")

- Why: on gold, Codex + VLM (two-reader) beats Codex alone by +0.05..+0.08 (interval > 0). The non-gold 'replication' scored the blend
  against Codex ALONE, a weaker label (0.891 on gold vs 0.94-0.97 for the two-reader average), so it was biased against the VLM.
- Plan:
  1. pool_q9b_Ag: zero-shot pass-2 config, Effusion, all 4,349 non-gold studies (~35 min).
  2. Check the two-reader target vs an independent third reader: the 384 Opus Effusion reads.
  3. Experts: one crop-slot img_eff retrain on Codex + VLM (f1) vs the existing seed trained on Codex only, read in the shipped Effusion
     form.

## Effusion set-up search (USER 19:4x: "just find the best VLM labelling set up first (what to give, prompt, what ZOOM, etc), then measure on gold ...")

PRE-REGISTRATION (19:4x, before any e_* data):
- Sets:
  - 299 pilot studies (Codex / trusted / text refs);
  - 354 Opus-read studies (hard cells, text below / unsettled; Opus = an independent image reader).
- Configs:
  - e_Ag (A sheets + guide = the gold-best complement so far);
  - e_Ag_grade (same, 0-3 grade question, score = P(grade >= 2));
  - e_ZE (pouch48 zoom crops: sag x2 + axial x2, definition);
  - e_Ag_tshot (A + guide + 2 + 2 TEXT-explicit examples; not Codex-picked, so not fitted-to).
- Choice (esearch_eval.py): highest two-reader AUC (z Codex + z VLM) vs Opus; ties within 0.005 -> higher AUC vs trusted, then cheaper.
- Then ONE gold-58 read of the pick + the e_Ag baseline (gold_check.py).
- Then, if the two-reader gain on gold holds:
  - label the 4,349 non-gold studies with the pick;
  - Experts trains C217 (img_eff2r: two-reader target);
  - Brainstorm's RK3 tie-breaking can use the VLM read as a teacher for Effusion ties (their survey: Codex-teacher ties carry 5-8 %
    new order).

## Zoom pass, few-shot (z_q9b_Z2_shot, 19:30-19:46): gate FAILED -> focal VLM labelling CLOSED for this model

vs text (trusted):
- ACL 0.48 (0.50);
- MM 0.47 (0.46);
- LM 0.49 (0.49);
- MOA 0.62 (0.61);
- LOA 0.52;
- PF 0.54 (0.57);
- MCL 0.69.
No column reaches the pre-registered 0.80. Zoomed localised crops (~0.09 mm / px) and text-explicit examples do not make Qwen3.5-9B read
ligaments, menisci or cartilage. MCL (0.68-0.69) and MOA (0.62) are the only weak signals.
Strength: 2 zoom configs (zero / few-shot), 299 studies, text + trusted references. Untested: 27B or thinking on zooms, fine-tuning.

## Effusion set-up search: result (pre-registered pick) + gold check

| config | vs Codex (pilot) | vs trusted (pilot) | vs text (pilot) | VLM vs Opus | two-reader vs Opus (Codex alone 0.677) |
|---|---|---|---|---|---|
| e_Ag (sheets + guide) | 0.861 | 0.870 | 0.740 | 0.685 | 0.737 |
| e_Ag_grade (0-3 grade question) | 0.853 | 0.868 | 0.747 | 0.700 | 0.744 |
| e_ZE (pouch zoom crops) | 0.687 | 0.771 | 0.635 | 0.624 | 0.700 |
| **e_Ag_tshot (sheets + guide + 2 + 2 text-explicit examples)** | **0.930** | **0.942** | **0.785** | **0.737** | **0.756** |

Opus set: 354 hard studies, n+ 63.

- Rule pick: e_Ag_tshot. Its examples come from text labels only, so it is not fitted to Codex.
- Gold-58 (Effusion n+ 35), one read:
  - e_Ag_tshot alone 0.871 (Codex 0.891, package 0.980);
  - Codex + VLM 0.949, gain +0.058 [90 %: +0.014, +0.109];
  - package + 0.25 VLM 0.979 (no gain).
  - Reference (zero-shot e_Ag = gold_q9b_Ag): Codex + VLM 0.974 (+0.083).
  - The pick is kept as pre-registered (not re-picked on gold).
- Verdict: the two-reader LABEL route for Effusion holds on gold. The test-time member route does not.
- Next:
  - pool labelling with the pick (4,345 non-gold studies, ~2.4 h, slot after Experts' deep_sup; passes_pool.sh, flag-gated);
  - then build_ieff.py -> experiments/v3/oct16_experts/data/ieff_codex_vlm.csv;
  - C217 img_eff2r (lead robust gold; VLM and Codex fitted-to);
  - RK3 Effusion two-reader teacher variant (Brainstorm; fitted-to rule applies).

## Pool labelling done (10-20 01:06; passes_pool.sh; Qwen3.5-9B e_Ag_tshot, the pre-registered pick)
- 4,345 non-gold studies in 9,123 s (2.1 s / study, peak 20.4 GB).
- build_ieff.py -> experiments/v3/oct16_experts/data/ieff_codex_vlm.csv (T1 two-reader), ieff_vlm.csv (T3; T2's 2nd head),
  ieff_union.csv (T4; 773 VLM-only fills).
- Codex-VLM Spearman 0.732 (3,572 with both).
- vs the independent Opus reads (384 hard studies, n+ 67): Codex 0.688, VLM 0.731, two-reader 0.765.
- Next: Experts' C217a-d arms (+ EF3 with Opus), read on robust gold + trusted + text + H (PROMISE); Codex and VLM are fitted-to.
