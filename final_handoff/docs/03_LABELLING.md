# 3. Labels: how we used ChatGPT, Codex, Claude and a local VLM

## 3.1 Why labels were the main lever

- The training set gives a radiology report per study. The hidden test (and the 58 official "gold" training studies) is labelled
  by radiologists reading the images against the host's definitions.
- Reports and image labels disagree in systematic ways:
  - findings the report never mentions ("silent cells");
  - wording that is vague about size or grade;
  - co-reported findings (bone bruise <-> MCL, ACL tear <-> lateral meniscus);
  - site habits.
- So we (1) turned reports into soft text labels with LLMs, (2) had VLM agents read the images for the classes where reports are
  weakest, and (3) picked one label source per class by checking it against gold (the "yardstick", 3.6).

Raw outputs of every labeller are summarised in `label_data/*/README.md` (copied from `data/text_labels/README.md`,
`data/image_labels/README.md`, `data/derived/README.md`); they list every file, field and caveat.

## 3.2 Text labels (reports)

| version | labeller | what it is | used for |
|---|---|---|---|
| v0 / v1 | ChatGPT (Pro, project sessions), 50 reports per session | per finding: `status` (meets / unsettled / conflict / hedged / below / general_normal / excluded / structure_normal / denied / not_mentioned), quote, attributes; v1 adds a fixed probability per status chosen by wording rules (`probability_scale.py`) | **v1prob** = the default training target of every recipe (not_mentioned -> 0.15) |
| our hand labels | us | 208 reports (gold-58 + 150 random) in the same vocabulary | scoring labellers; "trusted" / "look at" eval cells (`common/trusted_rules.csv`) |
| v2 | ChatGPT project sessions (89 ranges x 50) | every statement as a typed fact (structure, finding, assertion, scope, section, attributes), no label decisions | aux-fact experiments, v4 label designs |
| v3 | 5 Claude Code sessions (Claude Opus 5.5), index % 5 split | per finding: evidence class + forecast `p` that the image annotators marked it positive, using the clinical history and the report's thoroughness | label experiments; silent-cell context |
| v4b | code (`code/training/oct16_trainers/target_builders/v4b_build_targets.py`) | calibrated Lateral OA labels: silent cells set low, stated cells by report evidence | **silsep** (the LB-won Lateral OA member) |

How the ChatGPT sessions ran (`code/labelling/text/`):
- One ChatGPT project holds the bundle zip as a project file: `prompt.txt` (~12,400 tokens of rules + 6 worked examples),
  `definitions.md` (the host's definitions verbatim + figure captions), `schema.json`, `validate.py` and `reports.csv`.
- Each chat gets the copy-paste prompt from `session_prompts.txt`: "unzip the bundle, follow prompt.txt, deliver v1_FROM-TO.zip",
  plus `Range: FROM-TO`.
- Returned zips -> `returned_<version>/` -> `export_labels.py <version>` -> `data/text_labels/<version>.json`.
- Quality checks:
  - the first ranges are our hand-labelled reports (shuffled in);
  - repeat runs of 4 ranges gave the same (structure, finding, assertion) sets (Jaccard 0.93-0.95);
  - `validate.py` checks format, allowed values, and that quotes occur in the report.
- `reports.csv` (competition reports) is not included. Rebuild it from the competition data with `make_bundle_csv.py`.

## 3.3 Image labels (VLM agents reading the images)

Infrastructure (`code/labelling/image/bundle_tools/`):
- `render.py` / `build_bundle.py`: 9 contact sheets per study from the slice cache:
  - 4 overview sheets (sagittal / coronal / axial fluid-sensitive, non-fat-suppressed);
  - 5 zoom sheets of 9 consecutive slices (notch, medial, lateral, coronal joint line, patellofemoral);
  - burned-in tile ids, orientation letters, positions and a 10 mm bar.
- `view.py`: native-resolution views of a study by bundle index, for local agents. It hides study ids.
- `bundle_src/`: the prompts (v4 = per-finding sections that start with the host's definition verbatim + the host's 13 annotated
  figures + a fixed reading order), panel templates (fluid, synovitis, Baker's, lateral OA), schemas, `validate.py`.
- `mapping.py`: status x certainty -> probability (meets 0.95 / 0.85 / 0.7 ... normal 0.02 / 0.04 / 0.08).
- `export_labels.py`: Codex synovitis reads -> `synovitis_labels_codex.csv` (mean probability per study, block ranks,
  post-stratification weights).

| labeller | reads | panel | result |
|---|---|---|---|
| Codex (local agent, gpt-6-astra xhigh) | 3,575 non-gold, blind, images only (+189 rereads) | synovitis + effusion | **Synovitis image target** (raw mean probability, `img_syn_prob.csv`): Synovitis AUC +0.048, LB 0.948 -> 0.952; used by every image head since |
| Codex | 516 report-vague Baker's; 58 gold Lateral-OA test | Baker's / lateral OA | Baker's never beat the text target as a training target; Lateral OA failed on gold (no pool reads) |
| Claude Opus 5.5 (Claude Code chats) | 384 report-vague effusion cells | fluid | third Effusion reader in the EF3 target |
| Claude Opus 5.5 | ~1,960 non-gold (p2 811, p3 682, p3c 151, p3pf 320), images first, report as evidence; + 160 Baker's (p4) | contusion, fracture, PF OA | **bone v3c targets** (Opus where read, old target elsewhere): Contusion all-4 and Fracture v3c_fs, both LB-won; PF held (failed fold 0) |
| Qwen3.5-9B (local, zero-shot, `code/labelling/image/qwen_vlm_oct19/`) | 4,345 studies, effusion only (`label_data/image_labels_vlm/`) | P(yes) / (P(yes) + P(no)) at the first answer token, contact sheets + definition + 2+2 examples | second Effusion reader: Codex + Qwen z-aligned (+ Opus on 384) = **EF3 target** (`ieff_codex_vlm_opus.csv`), LB-won Effusion crop slots |
| ChatGPT web, Claude Fable 5.1, Opus all-12 | gold only | | comparisons; weaker than Codex |

### How the Claude Code labelling chats were run

The exact prompts are in `code/labelling/image/claude_oct16/session_prompts_eff_c1-c5.txt` and
`code/labelling/image/claude_oct20/session_prompts_*.txt`. The pattern:

1. Build priority-ordered batch lists, one per chat (`build_phase*.py` -> `_batches/<chat>.txt`, one bundle index per line), with
   catch copies (an already-read study re-shown under a new index, with a small drawn mark) and gold rereads for drift checks.
2. Start 4-6 Claude Code chats in parallel in the repo, each with its own session prompt naming:
   - its list, read-only prompt folder, output folder (one `<study_id>.json` per index) and temp folder;
   - the view tools (`fluid_views.py` / `view.py`) that render the sheets and native views into the temp folder;
   - strict blindness rules: no reports (except the images + report phases), no labels, no other chat's folders, no web, no
     attempt to identify the study;
   - "look with the Read tool at full resolution; code may only display / crop / enlarge / change contrast, never decide labels";
   - validate every file right after saving (`validate.py`); work in list order; resume from the first index without a valid file;
   - end with "N of M saved, next index" and up to 10 unsure lines.
3. Collect: `code/data_prep/build_image_labels.py` merges every raw folder into `data/image_labels/<run>.json` + `reads.csv`.
4. Evaluate on gold (`eval_*.py`, `label_audit_gold.py`) and build the training target (`target_builders/`).

Codex chats ran the same way (`code/labelling/image/codex_oct7/session_prompts_*.txt`). The Codex quota ran out during the
synovitis rereads, and the ChatGPT Pro subscription ended on 10-18; one Claude usage reset (10-19) bought ~2,100 Opus reads (phases 2-4).
Plan for quota early.

## 3.4 Lessons about labels

- A reader's value is as a better TRAINING TARGET where text is vague or silent, not as a test-time member: the package beats every
  single reader on gold.
- Image-only VLM reads are good on fluid and large structures (effusion, Baker's, ACL, contusion) and near chance on focal findings
  (MCL, menisci, cartilage grades). Images + report beat text on Synovitis, Contusion, Baker's and MM, and lost on PF OA and MCL.
- Use the raw mean probability of the reads, not within-chat block ranks (ranks are not comparable across chats with different
  positive rates; raw probability won on held-out reads and on the LB).
- **Standard recipe for a new label:** sample as before; use the new label where read and the member's old target elsewhere,
  weight 1. It beat every variation tried on the bone experts (oversampling, doses, masked heads). Masked image heads helped only
  on a new member at low coverage (sxm).
- Read every label arm against its same-recipe old-target seed twin. The twin row separates the label effect from member
  diversity.
- Calibrated text labels (v4b) helped only where the column follows gold over the reports (Lateral OA).
- Never train on gold. Gold is also not a fully independent test: some prompts were written by people who had seen gold
  statistics.

## 3.5 Pseudo-labels

`code/labelling/pseudo_labels/`, targets in `label_data/pseudo_labels/pl2_<recipe>_v1.csv`.

- History:
  - before the last day, pseudo-labelling failed ~11 times on gold and on the LB (OA, LOA, MCL experts; inner-world students
    +0.001 LB, not separable);
  - it was the last-resort lever.
- Final design (v2, 10-21):
  - teacher = honest out-of-fold predictions of the best fold pipeline, per column in its LB-best form (`teacher_v1.py`);
  - student target = (1 - a) y + a p_T, with a = 0.3 (explicit report cells / image reads), 0.7 (other stated cells),
    1.0 (silent cells), and p_T quantile-mapped onto the recipe's own OOF scale (`targets_v2.py`);
  - one pseudo-label twin per final model and seed.
- What we saw on gold-58:
  - PL seeds are ~2x more seed-stable for every recipe;
  - they help where plain members are noisy (MCL, PF OA, Effusion);
  - they hurt where a label design matters (Lateral OA: PL overwrites v4b's silent cells) and where diversity matters (bone: the
    members collapse onto each other);
  - main models: Medial OA dropped (-3.7 sd).
- On the LB, P3 (plain + PL blend, 117 models) = 0.965, the same as the plain P1 at 3 decimals (P1 ranked above P3 at hidden
  precision). P3 is one of the two finals as a hedge.

## 3.6 One label source per class (the "yardstick")

From `docs/source_notes/agent_prompts/validation.md`: each class has one label algorithm that is both the training target for new
models and the held-out referee.

| class | yardstick (= training target) |
|---|---|
| Contusion, Fracture | Opus images + report where read, old target elsewhere |
| PF OA | 0.5 Opus + 0.5 text where Opus read, old target elsewhere |
| Effusion | EF3: Codex + Qwen (z-aligned) everywhere, Opus as a third reader on its 384 vague reads |
| Synovitis | Codex synovitis reads |
| Baker's | Codex Baker's reads where read, old target elsewhere |
| ACL, MCL, menisci, Medial / Lateral OA | the text-based recipe targets (image reads were no closer to gold than the reports) |
