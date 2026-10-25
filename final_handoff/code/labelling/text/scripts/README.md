# Text labelling bundle

Notes for you about text_labeling_bundle/ (kept out of the bundle so the labelling model never sees them). Rebuild the root zip with `zip -r text_labeling_bundle.zip text_labeling_bundle` from the repo root.

## Files

| File | Content |
|---|---|
| prompt.txt | The full prompt (about 12,400 tokens): task, decision rules, meaning of every allowed value, probability scale, procedure, output, 6 worked examples. Expects the other bundle files alongside it and a line `Range: FROM-TO` |
| definitions.md | Official label definitions, verbatim from the host post #733343, plus the captions of its 13 annotated example figures (the figures themselves are left out on purpose: only the captions bear on reading reports) |
| schema.json | Exact output format for one study: keys, types, allowed values, allowed probability values per status (JSON Schema draft 2020-12) |
| validate.py | Checks output files: format, allowed values, index and study_id, quotes present in the report. Pure Python, no packages |
| reports.csv | All 4,407 train studies: index, study_id, knee_side, language, report |

- `knee_side` is the imaged knee from DICOM (Laterality tag, else image position; agrees with the tag in 96.8% of tagged studies). The model needs it for reports that describe both knees.
- Indexes 0-207 are the 208 reports I labelled by hand (gold-58 + 150 random non-gold), shuffled, so the first batches can be scored against my labels. Indexes 208-4406 are the rest, shuffled.
- 206 studies share their report text with another study. They are kept on purpose: their outputs should agree.

Versions: v1 adds a probability per finding (fixed values per status, chosen by wording rules; scale in `probability_scale.py`) and names output archives `v1_FROM-TO.zip`. The v0 bundle is frozen in `bundle_v0/`. Returned archives go to `returned_<version>/`; export with `export_labels.py <version>`.
