# oct16_experts: initial design

Do not edit this document unless the user explicitly asks.

## Job

Keep finding the highest-ROI expert ablations for any class or group of classes, test them, and propose Kaggle probes (see "Kaggle
probes"). Work in cycles: rank ideas by expected gain per GPU-minute, run the best one, read it, log it, re-rank.

Base: the 0.961 fold-average package (submissions/oct13_f5_win). Do not spend GPU time on any all-data models.

## Levers

These levers are general guidance, not a fixed structure. Any idea that can reasonably give an uplift is in scope if:
- it has not been properly tried before;
- its estimated ROI is high compared to other ideas;
- it could work given everything learned so far.

1. **Per-class experts** (preferred) or per-class-group experts, with hand-crafted window sampling:
   - remove certain series (for example, fat-sat only);
   - crop certain areas inside windows, with or without the full original frames. This includes zooming and higher resolution;
   - restrict each output to its own crops (per-output attention masks), so one class cannot borrow another class's evidence;
   - choose windows / crops with de-coupling in mind: train several models, each focused on one aspect of a class or class group
     (one region, one series type), then ensemble them. Each model alone misses the other aspects; this is expected.
2. **Backbones**, including ones outside the main three (effb0, NFNet-L0, R50).
3. **Inference-time settings:**
   - window / crop fusion with a top-k gate. It helps focal findings and report-trained models, and does nothing for models trained
     on image reads;
   - how many windows are evaluated.
4. **"Free" auxiliary outputs:** an expert trained for one class can supply another class's output from the same forward pass.
5. **De-coupling outputs:** add (and track) extra outputs such as "class X positive and class Y negative" (for example Lateral OA without
   medial OA, contusion without an ACL tear). They teach the model what separates often co-occurring findings; use them as auxiliary
   outputs, ensemble members or de-coupling terms.

Finalisation: re-fit the blend (e.g. the expert's share vs the generalists; any other hparams fitted to older models), and post-hoc de-coupling where outputs are over-coupled.

## How to come up with ablations

- Adapt what worked in other classes, preferably similar ones.
- Analyse the 0.961 package's systematic errors on a class: read flips and errors study by study, on gold and on Codex reads, and ask
  what causes them.
- Find over-coupled classes or cases: measure each class's AUC within strata of another class's status, and compare with how often
  the two co-occur in gold.
- Find where models fit the report instead of the image: attention on non-fat-sat series, or site and language habits in report
  wording. A narrower input diet helps there.
- Find crops and series not yet tried, and check where trained models look (attention).

## Kaggle probes

- Baseline: the 0.961 fold-average submission. If a change is inference-only and also applies to our best all-data submission (which
  may change during the session), you may propose the probe on top of that submission instead.
- One probe should generally test exactly one assumption. If only one class is tested, take every other class from the baseline
  pipeline exactly.
- Use probes for the highest-ROI improvements that look promising locally.
- Also consider arms that are within noise locally (not clearly better or worse) but whose predictions correlate less with the
  package's. They differ in something, and the LB can test whether that difference helps. Investigate, as well as you can, what that
  something is (which studies move, which cells, sites, series, co-occurring findings) before proposing the probe.
- Aim to use all daily probe slots; unused slots expire.
- Never make an official submission without the user's explicit permission for each one.

## Structure

- **prior_art.md:** everything tried before, in any experiments folder, grouped by class and lever. Per entry: the exact setting,
  folds x seeds, result, evidence strength, source path. Also the toolbox: existing trainers and package stages, their options, and
  whether the package can already deploy each form. Build it first; add a line per new result.
- **ideas.md:** one ranked table across all classes. Columns: class(es), lever, idea, why (link to prior_art.md or a finding), how
  underexplored, estimated gain (AUC and LB), GPU cost, ROI, pass bar set before running, status, result link. Re-rank before and after
  every GPU run.
- **user_ideas.md:** a ledger of what the user suggests in chat. Per entry: date and time, the user's exact words, what to try, and
  what was done about it (acted on or not, and why). The session may act on a suggestion or not. A newer suggestion can cancel or
  update an older one: mark the older entry as superseded and link to the newer one.
- Other files at the session's own judgement.
