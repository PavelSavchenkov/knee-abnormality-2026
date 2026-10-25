# Claude (Opus 5.5) labelling plan: which labels are most promising (Master, 10-19 ~23:2x)

USER (verbatim): "i still have a claude full reset, so i could label some with claude (i think about 1k-2k in total). research which ones
are the most promising labellings. also look at our older 58 gold labels and use them for analysis. since even 9B model labels have some
leads, claude opus 5.5 should be useful as well..." and "did we use all labels available in image_labels/ folder? i remember we labelled
smth like bakers... have we tried to use those labels?"

Evidence: gold_readers.py / two_reader.py -> gold_reader_table.txt (gold-58; every reader blind; tiny n, INFO). Gold numbers stay in this
folder, never in memory (labeller blindness).

## What the gold-58 table says (AUC; pkg = the shipped 0.961 fold package, all fold models)

- Opus 5.5 IMAGE-ONLY (all12 v4 prompt, 30 gold):
  - good on fluid / large structures: Effusion 0.85, Baker's 0.99, ACL 0.95, Contusion 0.83;
  - near chance on focal findings: MCL 0.45, LM 0.44, MM 0.67, MOA 0.66, LOA 0.57, PF 0.56, Fracture 0.60, Synovitis 0.65.
- Opus fluid panel (58 gold): Effusion 0.927 (the best single image reader; Codex 0.891); Synovitis 0.751 (Codex 0.878); Baker's 0.943.
- Two-reader targets:
  - Effusion: Codex + Opus 0.945; Codex + Fable 0.960 (45 gold); Codex + Qwen-9B 0.949 (tonight's pick; 0.974 zero-shot). The free 9B
    already gives what Opus gives here.
  - Synovitis: Codex + Opus 0.834 < Codex alone 0.878 (Opus hurts). Baker's: flat.
- Codex IMAGE + REPORT (39 gold; reads the report AND the images; a training label may use the report):
  - beats text v1 on Synovitis (+8.8 vs text on the same studies), Contusion (+7.0), Baker's (+6.2), MM (+3.6), Effusion (+1.4);
  - loses on PF (-5.3), MCL (-1.0); LOA = text.
  - It orders correctly many of the package's own discordant gold pairs: PF 17 / 27, MM 10 / 16, MCL 8 / 11, LM 5 / 14, Synovitis 15 / 56.
- The package beats every single reader on gold for almost every column. A reader's value is as a better TRAINING TARGET where text is
  vague or silent (the Synovitis precedent: Codex image targets moved the LB +0.048), not as a test-time member.

## Recommendation for the ~1-2k Claude budget

1. PILOT (~100 reads, 1 chat).
   - Opus 5.5 IMAGE + REPORT reads of the 58 gold studies, mixed blind with ~42 non-gold decoys.
   - Panel: Contusion, Fracture, Medial / Lateral Meniscus, PF OA (+ Synovitis / Effusion for comparison).
   - Infrastructure: the 10-16 Claude chats (work/claude_labels_oct16, fluid_views.py) plus a per-index report file, as Codex's
     gold_text_image run did (work/codex_labels_oct7/_reports_gold).
   - Read per class vs text v1, the package, and Codex image + report. "Fixes" = package-discordant gold pairs ordered correctly.
2. MAIN (~1,000-1,500 reads, 4-6 parallel chats), only for the class(es) where the pilot beats text clearly.
   - Studies: non-gold training cells where text is weakest for that class (below / unsettled / hedged + silent studies the 0.961 OOF
     scores mid-range), highest model uncertainty first.
   - -> a target CSV -> Experts trains one target arm on the class's expert (fold 1 + seed row, then folds).
   - -> a day-2 probe if it holds, and the final if the user wants it.
3. FALLBACK if the pilot fails on every class: Opus Effusion second reads on the remaining report-vague cells (eff_c1-c5 lists exist;
   384 of ~2,100 done). Low priority: tonight's Qwen pool already provides a second effusion reader of the same gold quality, and
   Effusion is saturated on gold (pkg 0.980).
Not recommended: Opus synovitis reads (they hurt the Codex target on gold); Opus image-only focal labels (near chance).

## Image-label files: what has been used (USER question)

| file | used as | result |
|---|---|---|
| codex_synovitis (3,575) | the Synovitis image-head target (img_syn) + img_eff | LB +0.048 Synovitis (0.952); in every package |
| codex_synovitis_reread (189) | calibration / attribute targets (oct10_synovitis) | no gain over the raw reads |
| codex_bakers (516 report-vague Baker's) | a reader (W2 decision read); cell rates in calfl (U68 G1: FAIL); pw cross-site pairs (Q25: LB -100e-5); RK3 tie teacher (NONE) | NEVER used as a direct per-study training target (oct9 plan step 2 never ran: ceiling ~0.001-0.002 macro) |
| opus_effusion (384) | a reader in pw / C112 / C115 / C121 reads; the Opus reference in tonight's VLM set-up search | NEVER used as a training target; C217 tests the two-reader route tonight with the Qwen reads |
| *_gold.json | evaluation only (never train on gold) | - |

Untested and cheap (GPU only, no new labels):
- a Baker's direct-target arm: the main recipe with Baker's target = the Codex read on the 516 vague cells (tgt_partial), and text
  elsewhere;
- an Effusion two-reader target with Opus on its 384 cells (folds into C217 as a variant).
Prior: low (Baker's ceiling; Q25 lost on Codex-pair Baker's).

## Update (USER: "you can use our learnings from our local VLM runs -- i think the best one was to give text as reference?")
Yes. In tonight's Effusion set-up search the winner was text as reference: e_Ag_tshot = sheets + guide + 2 + 2 TEXT-EXPLICIT example studies
(v1 'meets' p >= .85 vs 'denied' / 'structure_normal' p <= .1).
- vs Codex 0.930 (vs 0.861 without examples); vs trusted 0.942 (0.870).
- It is not fitted to any image reader.
- Zoom crops were worse (0.687); the grade question added nothing.
Port to Claude, with the text as reference in two places:
- (a) the study's OWN report beside its images (Codex image + report on gold beat text on Synovitis / Contusion / Baker's / MM);
- (b) per class, 2 + 2 text-explicit anchor studies (images + their clear report-derived label) at the start of each chat, to fix the
  host's threshold.
Pilot = (a) + (b) together on the 58 gold (blind, mixed with ~42 non-gold decoys).
