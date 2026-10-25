# Image labels

VLM reads of the knee MRI images (2026-10-07..10, 10-16 and 10-20), built by `data/build/build_image_labels.py` from the raw label folders in
`work/{codex,claude,claude_fable51}_labels_oct7/`, `work/claude_labels_oct16/`, `work/claude_labels_oct20/` and the returned ChatGPT archives in
`work/image_labeling/returned/`. This folder holds every image-based label made so far. Codex labelling has ended (quota used up); on
10-16 Claude Opus added effusion second reads. On 10-20 Claude Opus read images + report for contusion, fracture and PF OA (`opus_p2`),
after a gold pilot with 5 findings (`opus_pilot5_gold`).

Each labeller looked at the study's images itself: contact sheets of the image-labelling bundle (`image_labeling_bundle/`), and for the local
labellers (Codex, Claude) also native full-resolution slices rendered by `work/image_labeling/view.py`. Reads were blind: no report, no
labels, no knowledge of which real study an index is (the session logs were audited for this), except `codex_report_image_gold` and the
10-20 runs (`opus_pilot5*`, `opus_p2`), which also used the study's radiology report (images first, report as evidence). The 10-16 and 10-20
Claude Code chats also loaded the project's CLAUDE.md and memory index, which hold no per-study labels.

## How to use them

- **Synovitis (and effusion) as a training target: use `data/derived/synovitis_labels_codex.csv`, not the JSON files.** One row per labelled
  non-gold study (3,574): `Synovitis`, `Effusion` = mean probability over the study's reads (blank = not assessable); `*_blockrank` =
  percentile rank within blocks of 75 reads in labelling order inside each chat folder (removes each chat's wording style and drift; used by
  the image synovitis head of the 0.944 submission, `experiments/v0/oct7_baseline/IMAGE_LABELS.md`). NOT recommended since 10-11: block ranks are
  not comparable across chats with different positive rates (experiments/v1/oct10_synovitis ledger 22:25); the raw mean probability
  (`Synovitis`, `Effusion`; data/img_syn_prob.csv in the experiments) won on held-out reads (+9e-3) and on the LB (+0.012 synovitis),
  and every deployed image head uses it;
  `reads` (1 or 2); `folder`, `sampling`, `stratum`, `cell` (text v1 effusion / synovitis status); `weight` = post-stratification weight
  per (stratum, cell), because the labelled set over-represents studies whose report suggests effusion. Studies without a read have no
  image target (masked loss). Rebuild it with `work/image_labeling/export_labels.py` (usage in its docstring) after any change here.
- **Baker's cyst: `codex_bakers.json`** covers only the 516 studies whose text v1 Baker's status is below or unsettled (a report cyst
  near the size threshold, where the text labels are least reliable); elsewhere keep the text label. It has no export or weights yet
  (the pool is one selected text cell, so the synovitis weights do not apply). Expected gain is small (work/pl_research/README.md section 8).
- **Effusion second reads: `opus_effusion.json`** (Claude Opus, fluid panel) covers 384 non-gold studies whose text v1 effusion status is
  below or unsettled. All of them also have a Codex read, so they can be averaged into a 2-reader effusion target. They were picked by
  priority, not at random (see Caveats). No export yet.
- **Contusion, fracture, PF OA: `opus_p2.json`** (Claude Opus, images + report, panel p2), non-gold studies picked mostly where the
  report is vague (see Caveats for the strata). Training forms that did best on gold (`work/claude_labels_oct20/eval_target_forms*.txt`):
  Contusion and Fracture = the Opus `probability` alone, PF OA = 0.5 x Opus + 0.5 x text v1 probability; elsewhere keep the recipe's old
  label (dense target). The experiments' copies are built by `experiments/v3/oct16_experts/scripts/build_tgt_p2.py` (frozen wave 1) and
  `build_p2_v2.py` (all reads). `selection.wave` = 1 marks the 559 reads frozen for training waves 1-4.
- **Gold runs (`*_gold.json`, the 58 hand-labelled studies): evaluation and diagnostics only.** Never train on gold. They are not an
  independent test either (see Caveats).
- **Lateral OA (`codex_lateral_oa_gold.json`): a failed feasibility test.** On gold it did not beat the report text, so no training-pool
  reads were made; do not use it as a label source.
- **Other labellers (Claude Opus / Fable, ChatGPT web):** gold only, weaker than Codex; kept for comparison.
- Quality evaluations (gold AUCs, agreement, drift) are in `work/image_labeling/review/` (one script per question, docstrings say what).

## Files

| File | Studies | Labeller | Panel | Prompt |
|---|---|---|---|---|
| `codex_synovitis.json` | 3,575 non-gold | Codex (gpt-6-astra, xhigh) | synovitis | synovitis |
| `codex_synovitis_reread.json` | 189 non-gold: second blind reads of studies whose first read was synovitis possible / borderline (queue and b1-b8 chats; the 5 chats stopped at 33-47 of 100 when the Codex quota ran out) | Codex | synovitis | synovitis |
| `codex_bakers.json` | 516 non-gold with text v1 Baker's status below or unsettled (+ 12 catch copies in repeats.json) | Codex | bakers | bakers |
| `codex_bakers_gold.json` | 58 gold, mixed into the Baker's chats (14 rows read twice: second reads in repeats.json) | Codex | bakers | bakers |
| `codex_lateral_oa_gold.json` | 58 gold (+ 2 catch copies in repeats.json) | Codex | lateral_oa | lateral_oa |
| `codex_fluid_gold.json` | 58 gold | Codex | fluid | fluid |
| `codex_report_image_gold.json` | 39 of 58 gold (the chat stopped there), images + report | Codex | all12 | v4+report |
| `codex_v2_gold.json`, `codex_v4_gold.json` | 10 / 1 gold | Codex | all12 | v2 / v4 |
| `opus_v4_gold.json`, `opus_fluid_gold.json` | 30 / 58 gold | Claude Opus 5.5 | all12 / fluid | v4 / fluid |
| `fable_fluid_gold.json` | 45 gold | Claude Fable 5.1 | fluid | fluid |
| `opus_effusion.json` | 384 non-gold with text v1 effusion status below or unsettled, highest priority first (+ 6 catch copies in repeats.json) | Claude Opus 5.5 | fluid | fluid |
| `opus_effusion_gold.json` | 10 gold, mixed into the effusion second-read chats (Opus's second read of them; first read in `opus_fluid_gold.json`) | Claude Opus 5.5 | fluid | fluid |
| `opus_pilot5_fast_gold.json` | 30 gold: first pilot pass (fewer native views), superseded by the strict pass | Claude Opus 5.5 | pilot5 | pilot5+report fast |
| `opus_pilot5_gold.json` | 58 gold: strict pass (at least one native --box crop per finding, views named in notes) | Claude Opus 5.5 | pilot5 | pilot5+report |
| `opus_p2.json` | non-gold: Phase 2 lists s1-s7 (vague / silent_mid / random / frac_vague strata, `selection`) | Claude Opus 5.5 | p2 | p2+report |
| `chatgpt_v1_gold.json`, `chatgpt_v2_gold.json`, `chatgpt_v4_gold.json` | 10 / 10 / 20 gold | ChatGPT web, astra xhigh | all12 | v1 / v2 / v4 |
| `chatgpt_pro_v2_gold.json` | 10 gold | ChatGPT web, Astra Pro | all12 | v2 |
| `repeats.json` | 211 extra reads | | | |
| `reads.csv` | one row per read and finding (all files above, repeats included, column `repeat`) | | | |

Panels: `all12` = the 12 findings; `fluid` = effusion, synovitis, Baker's cyst; `synovitis` = effusion and synovitis; `bakers` = Baker's
cyst only (with `diameter_mm`); `lateral_oa` = lateral compartment osteoarthritis only (with `extent_mm`, `surfaces`; native coronal and
sagittal views of the lateral joint); `pilot5` = medial meniscus, PF OA, synovitis, contusion, fracture; `p2` = contusion, fracture, PF OA
(PF OA with `cartilage_loss`, `extent_ge_10mm`, `osteophytes`, `subchondral_marrow`, `narrowing`, `focal`; fracture with `bones`).

## Format

`<run>.json` is keyed by study_id (StudyInstanceUID), one read per study:
- `labeller`, `model`, `panel`, `prompt`; `source` (raw folder or archive) and `position` (order in which the labeller wrote it there);
  `sheets_viewed` (bundle sheets);
- `qa`: `image_quality`, `problems`, `postoperative`, `open_growth_plates`, `orientation_ok`, `pattern`, `note`;
- `findings`, per finding of the panel:
  - `status`: meets / borderline / possible / below / excluded / normal / not_assessable, judged against the official definitions;
  - `certainty`: high / moderate / low (null when not assessable);
  - `probability`: fixed value per status x certainty (below), null when not assessable;
  - the finding's attributes (for example effusion `grade` 0-3 and `depth_mm`; synovitis `effusion_synovitis` 0-3, `hoffa_signal` 0-3,
    `synovial_thickening`, `debris_or_fronds`; Baker's `grade`, `diameter_mm`, `neck_seen`, `ruptured`, `other_cysts`; meniscus
    `surface_tiles`, `morphology`, `extrusion_mm`; cartilage `cartilage_loss`, `extent_ge_10mm`; contusion `edema` per bone, `pattern`;
    fracture `line_seen`, `type`) and `note`. In `codex_report_image_gold` every note starts with "report: stated / denied / silent / unusable".
- `codex_synovitis.json` also has `selection`: `sampling` = queue (a random sample: priority and control strata, 3:1) or batch (folders
  synovitis_bK: random draws within text v1 effusion / synovitis status cells. b1-b8: the cells with the most label uncertainty,
  meets/not_mentioned, unsettled/not_mentioned, unsettled/meets, below/meets, meets/meets; b9-b18: every remaining study of those cells
  plus random draws from below/not_mentioned and not_mentioned/not_mentioned; b19-b28: the rest of those cells, the small cells and random
  denied/not_mentioned); `stratum` = priority (text effusion stated or likely) or control. The labelled set is not a random sample of all
  studies (about 770 studies, nearly all effusion-denied or general-normal, were never read); hence the export's weights.

- `opus_p2.json` also has `selection`: `stratum` (vague = text v1 contusion or PF OA status below / unsettled / hedged / conflict;
  silent_mid = not vague, report silent on contusion or PF OA and the 0.961 package's OOF probability (main / NFNet / R50 mean) for it in
  0.15-0.85; random = the rest of the pool; frac_vague = text v1 fracture status unsettled / hedged), `list` (p2_s1..p2_s7; the s7 chunks read by
  other chats keep `p2_s7`), `list_position`, `wave` (1 = saved before the 10-20 pause, the frozen 559; 2 = after) and `saved_at`
  (local time). Notes name the native series + slice range viewed and end with the source tag images / both / report.

Not included (they refer to the bundle's tile grid): boxes, evidence tile ids, structure boxes, drawn-mark positions. They stay in the raw files.

`repeats.json`: extra reads, each with `run`, `kind` and the same fields. `catch_copy` = a copy of an already labelled study shown again
under a made-up id (test-retest; 86 synovitis, 12 Baker's, 9 in gold runs, 6 Opus effusion); `gold_reread` = a gold study read again (84 at the end of
100-study synovitis chats, as a drift check; 14 Baker's gold rows read in two chats).

Probability per status x certainty (high / moderate / low): meets 0.95 / 0.85 / 0.7; borderline 0.55 / 0.5 / 0.45; possible 0.4 / 0.35 / 0.3;
below 0.1 / 0.15 / 0.2; excluded 0.05 / 0.08 / 0.12; normal 0.02 / 0.04 / 0.08 (copy of `work/image_labeling/mapping.py`).

## Prompts

- `v1`: 12 findings on bundle v1 (5 sheets: 3 fluid-sensitive overviews, non-fat-suppressed, one zoom sheet).
- `v2`: bundle v2 (9 sheets, including 5 zoom sheets of 9 consecutive slices) and its prompt.
- `v4`: per-finding sections that start with the host's definition verbatim, the host's 13 annotated figures, a fixed reading order
  (`work/image_labeling/bundle_src/prompt.txt`).
- `fluid`, `synovitis`, `bakers`, `lateral_oa`: panel prompts built from the v4 sections by `work/image_labeling/build_bundle.py`
  (templates `work/image_labeling/bundle_src/prompt_<panel>_template.txt`; built files in the bundle's prompts.zip and in the read-only
  copies the Codex chats used: `work/codex_labels_oct7/_prompts_q/` synovitis, `_prompts_b/` Baker's, `_prompts_l/` lateral OA).
- `v4+report`: v4 plus rules for using the radiology report (`work/codex_labels_oct7/session_prompts_oct8.txt`, chat 5).
- `pilot5+report fast`, `pilot5+report`, `p2+report` (10-20): the v4 prompt (`work/claude_labels_oct20/_prompts/prompt.txt`) with per-chat
  changes: panel, report rules (decide from the images first; a clearly stated report finding stays likely unless the images contradict it),
  native --box crops for every finding (strict passes), notes with views + source tag. Exact chat texts: `work/claude_labels_oct20/
  session_prompts_pilot.txt`, `session_prompts_p2s.txt`, `session_prompt_p2s7.txt`, `session_prompts_p2s7_helpers.txt`.
v1 and v2 were revised in place; only the current texts are kept.

## Where the rest is

- Raw reads (with tile boxes): `work/codex_labels_oct7/<folder>/<study_id>.json` (folder = `source` of each read); Claude in
  `work/claude_labels_oct7/`, `work/claude_labels_fable51_oct7/`; ChatGPT archives in `work/image_labeling/returned/`.
- Bundle index -> study, catch copies, queue strata: `work/image_labeling/out/order.csv`.
- How studies were chosen: `work/image_labeling/batches/make_*batches.py` (docstrings) and their selection tables `*.csv` there; the exact
  chat prompts: `work/codex_labels_oct7/session_prompts_*.txt`.
- Effusion second reads (10-16): raw reads `work/claude_labels_oct16/eff_c{1..5}/`; selection `work/image_labeling/batches/make_eff2_batches.py`
  (docstring) and its table `work/claude_labels_oct16/_batches/eff_c1-5.csv`; chat prompts `work/claude_labels_oct16/session_prompts_eff_c1-c5.txt`;
  prompt copy `work/claude_labels_oct16/_prompts/`; native views `work/claude_labels_oct16/fluid_views.py`; blindness audit
  `work/image_labeling/review/audit_claude_oct16.py` (0 flagged tool calls in all 5 chats).
- Images + report reads (10-20): raw reads `work/claude_labels_oct20/{pilot_c1,pilot_c2,pilot_c1_v1,pilot_c2_v1,p2_s*}/`; lists
  `_batches/`; selection keys `_key/pilot.csv`, `_key/p2s.csv`; list builders `build_pilot.py`, `build_phase2_6.py`, `build_phase2_frac.py`,
  `build_phase2_s7split.py` (docstrings); reports shown `_reports/`; progress / quality monitor `p2s_monitor.py` (no gold).
- Checks: `work/image_labeling/review/audit_codex.py` (blindness audit of the Codex logs), `batch_check.py`, `bakers_check.py`,
  `loa_check.py`, `reread_check.py`.

## Caveats

- Gold-58 is not an independent test of these reads: the prompts were revised after reviewing per-study disagreements on gold indexes 0-9,
  the synovitis rule "a distended joint is at least possible" was justified with summary rates of the gold-58 labels, and the Baker's pool
  was chosen from where the text labels fail on gold.
- Codex read gold 0-9 several times in one chat before its fluid-panel read, so those fluid reads are not independent.
- Late in long chats Codex shifts its wording (queue and b1-b8: "possible" instead of "meets"; b19-b28: fewer positives and fewer
  "possible", more "normal"); `position` lets you rank within blocks of a folder (the export's `*_blockrank`). The gold re-reads were
  read at the end of chats.
- Chats also differ in style: synovitis_b10 and synovitis_b27 call synovitis "possible" (instead of below / normal) and effusion
  "meets" more often than the other chats on the same kind of studies (effusion "meets" on the same text cells ranges ~0.13-0.55 per
  chat); ranks within a folder remove most of this.
- b10, b13, b17, b18 and b21 stopped on a Codex server error and were resumed in the same chat.
- Second reads (`codex_synovitis_reread`) repeat the first status in 56% of studies and move up more often than down (meets or
  borderline 83 -> 106 of 189); the export averages both reads (`reads` = 2) and ranks a second read within the block of the study's
  first read (a re-read block holds only uncertain studies).
- The bundle was rendered before the knee-side fix: about 100 studies (including gold rows 54, 55) are shown mirrored, so medial and
  lateral are swapped on their tiles (`data/derived/series_side_final.csv`, column `flip_final`). Side-specific reads (Baker's, lateral OA,
  menisci, OA) of those studies may describe the wrong side; qa.orientation_ok / qa.note flag it only sometimes.
- Some series are flagged "other knee" or lack a fluid-sensitive plane; qa.note says when a read was limited by that.
- `opus_effusion` is not a random sample of its pool (2,054 studies). Studies were ranked by |Codex effusion - image-model OOF effusion|
  + 0.5 x (1 - |2 x Codex - 1|): reader-vs-model disagreement plus an uncertain first read. The rank was dealt round-robin into 5 chats,
  and each chat stopped after its first 80 rows, so these are the ~384 highest-priority studies. Weight or stratify before using them
  as a validation reference.
- `opus_p2` is not a random sample either. Lists s1-s6 were dealt 60 % vague / 20 % silent_mid / 20 % random; after the 10-20 pause the
  remaining silent_mid / random entries were dropped at 16:09 (low value), so wave 2 of s1-s6 is mostly vague (26 silent_mid / random
  reads were saved between the 16:01 resume and the drop). s7 is report-vague fracture studies
  in random order; 9 fracture-hedged studies without a vague contusion or PF OA status were dropped from its tail. Stratify by `stratum`
  before using it as a validation reference. Report-silent / normal studies come out ~0 % positive, so silent cells hold few reads.
- `opus_p2` PF OA on report-"meets" studies: wave 2 calls fewer positives than wave 1 (12 / 37 vs 27 / 46, Fisher p 0.03; contusion and
  fracture stable), partly a different study mix (more s7 studies). Check PF results per wave.
- The 10-20 reads saw the report, so their agreement with report-derived labels is not independent; judge them against images (gold)
  and use report-derived readers as guards only.
