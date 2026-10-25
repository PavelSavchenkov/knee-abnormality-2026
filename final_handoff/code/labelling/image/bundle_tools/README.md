# Image labelling (VLM, ChatGPT Pro)

Builds `image_labeling_bundle/` (repo root, git-ignored) for image-only labelling in one ChatGPT project. Plan and rationale:
`public_knowledge/my_docs/2026-10-07_image_labelling_plan.md`.

```
PYTHONPATH=. conda run -n rsna python work/image_labeling/build_bundle.py                 # ~40 s, 11.1 GB
PYTHONPATH=. conda run -n rsna python work/image_labeling/build_bundle.py --prompts-only  # after editing bundle_src/: rewrites prompts.zip only
```

| File | Content |
|---|---|
| render.py | 9 contact sheets per study from the slice cache: 4 overview sheets (sagittal / coronal / axial fluid-sensitive, non-fat-suppressed) + 5 zoom sheets of 9 consecutive slices (notch, medial, lateral, coronal joint line, patellofemoral); burned-in tile ids, sequence, position, orientation letters, 10 mm bar |
| build_bundle.py | study order (gold, validation, priority, rest), catch items, rendering, image zips <= 480 MB, prompts.zip, README, session prompts |
| bundle_src/prompt_fluid_template.txt | fluid panel (effusion, synovitis, Baker's): build_bundle.py fills it with the shared sections of prompt.txt -> prompt_fluid.txt + schema_fluid.json in prompts.zip |
| bundle_src/prompt.txt, schema.json, validate.py | the labelling instructions, output schema and checker that go into prompts.zip (with definitions.md from the text bundle) |
| view.py | native-resolution views of a study by bundle index (same canonical orientation, letters, positions, 10 mm bar) for local agents (Codex); hides study ids |
| evaluate.py | scores returned archives (<run>_img_FROM-TO.zip): validity, minutes per study, catch items, gold status counts and AUC vs text v1 / steven v4 / R2 |
| review/ | per-cell gold disagreement listings (disagreements.py) and manual review notes |
| mapping.py | offline status x certainty -> probability (defaults; refit per finding on gold) |
| out/order.csv | index -> source study, section, priority, catch marker (internal; not uploaded) |

Returned archives go to `returned/` (and `returned_repeat/` for second runs).

Build v2 of 2026-10-07 (consecutive-slice zoom sheets after the goldA 0-9 sanity run, see review/goldA_0-9_review.md): 4,582 entries =
4,407 studies + 175 catch items; sections gold 0-59, validation 60-354, priority 355-4027, rest 4028-4581 (same order as v1);
24 image zips (480 MB each, last 113 MB) + prompts.zip. v1 (5 sheets, 11 zips) produced returned/goldA_img_0-9.zip. 31 / 56 / 48 entries lack a sagittal / coronal / axial
fluid-sensitive sheet; 439 fluid-sensitive sheets show a non-fat-suppressed substitute (said in the sheet title).

ChatGPT opened the 480 MB v1 zips from project files in its Python tool (goldA run).

Labels are categorical per finding, like text v1: `status` (meets / borderline / possible / below / excluded / normal / not_assessable)
+ `certainty` (high / moderate / low), with evidence tiles and boxes for every status from meets to excluded. No probabilities from the model.

Rebuild of 2026-10-07 23:42 (synovitis queue): indexes 0-59 are the frozen gold rows (gold_rows.csv), 60+ the synovitis queue (priority
and control strata, 3:1, shuffled; `stratum` in out/order.csv). prompts.zip adds prompt_synovitis.txt + schema_synovitis.json
(effusion + synovitis; templates in bundle_src/prompt_*_template.txt). `export_labels.py` writes the training CSV for the Experiments
session (probabilities per finding, stratum, selection weight).
