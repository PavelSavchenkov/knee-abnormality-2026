# Validation best practices

These are best practices for judging results, not hard gates. Use judgement.
- To depart from a practice in your session, you need a strong, justified reason. State it in your session's log next to the read it affects.
- Do not edit this document without the user's clear approval.

## Yardstick per class (2026-10-21)
- Each class has one label algorithm. It is both the training target for new models and the held-out referee (the "yardstick").
- Choose it per class with a label-vs-gold check:
  - Score each candidate label source as a label source on gold. All sources must have been made blind to gold.
  - Pick the one closest to gold that also exists on the training pool.
  - Only a few candidates are compared, and it is a one-time choice, not a fit of model weights.
  - Re-check when a new label source arrives (work/claude_labels_oct20/yardstick_check.py).
- Fix the yardstick before reading arms. Every arm of the class is judged on it, including arms trained on other targets.

| Class | Yardstick (= training target) |
|---|---|
| Contusion, Fracture | Opus images + report where read, old target elsewhere (data/v3e and later) |
| PF OA | 0.5 Opus + 0.5 text where Opus read, old target elsewhere |
| Effusion | EF3: Codex + Qwen VLM (z-aligned) everywhere, Opus as a third reader on its 384 vague reads (data/ieff_codex_vlm_opus.csv; the same recipe was checked on gold) |
| Synovitis | Codex synovitis reads |
| Baker's | Codex Baker's reads where read, old target elsewhere |
| ACL, MCL, Medial / Lateral meniscus, Medial / Lateral OA | the current text-based recipe targets (image reads are no closer to gold than the reports). The references below decide as before. |

### Measuring with a yardstick
Work in this order:
1. Model (the one decision metric):
   - The new-label model vs its old-label counterpart: same model, recipe and seed, trained on the old labels.
   - Measure held-out soft AUC against the yardstick on the studies that have a yardstick read (read-only view), per fold and pooled over folds, with a paired bootstrap SE.
   - If it improves consistently (same sign on every fold, pooled gain clear of its SE), we have an improved individual model.
2. Ensemble:
   - Replace the deployed member with the improved model inside the shipped ensemble, everything else as shipped.
   - Compare it with the shipped column on the same yardstick. If it improves, good.
3. Re-fit:
   - Re-fit the ensemble weights and fusion settings with the new member, cross-fitted (fit on some folds, score the others).
   - Keep the re-fit only if it beats the shipped column.
4. More members: if the ensemble gain disappears or shrinks, train more ensemble members on the improved recipe, then repeat steps 2-3.
- Start with the members most influential for the class (largest share of its shipped column) that are fast to train.
- Views of every yardstick number:
  - Read-only decides: held-out studies that have a yardstick read, which is where the new labels change things.
  - Dense is an error catcher only: all held-out studies, with the yardstick where it is defined and the old target elsewhere. It never decides. Use it to spot harm on studies the new labels never touched, and to mine studies that moved for hand reads. Its unread part is the old label (the counterpart's own target), so it favours the counterpart. Also, on report-clear contusions the host graders disagree with both label sources about 3 times in 10 (10-21 check).
- Guidance for step 1, not thresholds:
  - a consistent pooled gain of about 2 SE or more is an improved model;
  - a positive but smaller gain calls for more folds or seeds, not a close;
  - a negative gain means drop it.
- Gold, text, trusted, lookat and host-like reads are tie-breakers (when arms are within about 1 SE on the yardstick) and error catchers. They never block. If one drops below about -2 SE, or every gold leave-one-out value is negative, hand-read the studies that moved before deciding.
- "Within noise" is not "no gain". When a seed swap alone moves gold as much as the gain, write "not measurable on gold".
- Text favours models trained on text labels, and it cannot see gains on studies whose report is unsure. Read it as a damage check.
- Limits:
  - the yardstick shares a reader with the training target, so LB probes stay the final judge;
  - read-only gains shrink on test in proportion to how many such studies the test holds.

## References

### Gold
- The best labels we have: official, image-based labels on 58 studies.
- Small. Some classes have few positives, so one study can carry a class's whole gain. Report a leave-one-study-out range with every gold change.
- Inspect flips: look at the images of the studies whose ranking moved most.
- Contested studies (gold disagrees with both the report and the image readers): report results with and without them.
- The hidden test is labelled like gold, so gold's definition is the target. Where gold differs from reports (lenient on ungraded PF OA, stricter on contusion), align training labels with gold.
- First check that such a pattern is systematic: many studies, not a few contested ones. With 58 studies, a "definition" can be a few studies' reader noise. If unsure, an LB probe settles it.
- Never fit on gold: no weights, thresholds or picking the best of many arms. The best of N arms on gold is inflated; confirm it on other references, folds or seeds. (Choosing a class's yardstick label source among a few candidates is the one allowed use, see above.)
- Gold is information only for classes with a yardstick: it never blocks a yardstick gain.
- Keep per-study gold labels out of anything labelling sessions can see.

### Trusted
- Report cells whose rule was checked against gold.
  - Positives: clear findings that meet the definition (complete or high-grade tears, grade III-IV cartilage, moderate or large effusion, large Baker's).
  - Negatives: explicit "no ..." or "normal", plus a few silent cases that gold supports.
- Trust its losses: clear cases getting worse is a real problem.
- Don't expect gains on it. It holds only clear cases, so it misses borderline improvements and saturates early. A few cells are still wrong (report disagrees with the image).
- Mostly neutral to label-set changes, since all label sets agree on clear cases. That makes it the main non-gold read for label changes.
- Ceiling: inspect the remaining errors (highest-scored negatives, lowest-scored positives). If most are label errors, the metric is at its ceiling and further moves are noise.

### Lookat
- Trusted cells plus weaker ones:
  - silent cells and below-threshold grades as negatives;
  - unsized or ungraded mentions as positives.
- Guidance only, never a gate. Its labels are noisy: silence is not absence in terse report templates, and an ungraded mention is not always positive.
- Arms that change silent or vague cells disagree with it by construction.
- Stable within a seed, but it swings across seeds: a single-run difference is mostly seed noise.
- Ceiling: same check as trusted. It is lower, because more cells are wrong.

### Text
- Agreement with the v1 report labels, all 12 classes.
- Favours models trained on v1 text labels; label-changing arms look worse on it.

### Image reads (VLM)
- Large-scale reads, mainly for fluid classes.
- Favour models trained on those reads.
- Noisy: the same reader reading a study twice often disagrees with itself. Once models agree with the reader about as well as the reader agrees with itself, better models stop scoring higher. Then a flat image-read score does not mean "no gain", and small differences are mostly reader noise.

### Which references exist per class

| Class | Trusted | Lookat | Text | Image reads at scale | Gold |
|---|---|---|---|---|---|
| ACL | AUC | AUC | yes | - | |
| MCL | negatives only | AUC (few positives) | yes | - | few positives |
| Medial meniscus | AUC | AUC | yes | - | |
| Lateral meniscus | AUC | AUC | yes | - | |
| Medial OA | AUC | AUC | yes | - | |
| Lateral OA | negatives only | negatives only | yes | - | few positives |
| PF OA | AUC | AUC | yes | - | lenient definition |
| Effusion | AUC | AUC | yes | Codex; Claude second reads (10-16); EF3 | |
| Synovitis | - | negatives only | yes (weak labels) | Codex | |
| Baker's | AUC | AUC | yes | Codex (vague cells only); Opus second reads of 160 (10-21) | few positives |
| Contusion | negatives only | negatives only | yes | Opus images + report (10-20 / 21) | narrower than reports; contested studies |
| Fracture | negatives only | negatives only | yes | Opus images + report (10-20 / 21) | |

- PF OA also has Opus images + report reads (10-20 / 21). ACL / MCL / menisci / MOA / LOA have Opus zoom reads on gold only (10-21 check).

- Negatives-only cells can still flag false positives.
- Classes without a trusted AUC (MCL, Lateral OA, Synovitis, Contusion, Fracture) lean on gold, text and image reads.

## Noise and baselines
- Comparing with the one deployed model is fine for spotting failures and promising runs. Trust it more when that model was not picked as the best of several runs; a picked model is a lucky draw on the references it was picked on.
- To decide, compare with several seeds of the same config from the same code.
- Noise = the spread of those seeds on the same folds. A bootstrap over studies covers sampling noise, not seed noise.
- One-seed, one-fold differences often fail to replicate. Pool folds before deciding, and check that the per-fold signs agree.

## Reading a result
- Read each class and class group (OA, menisci, ligaments, fluid, bone). Averages hide gains and losses.
- Read neighbouring classes too: a change for one class can move others through the shared backbone.
- For each read, say whether it is independent of the arm's training labels. For classes with a yardstick, decide on the yardstick; for the others, decide on the independent reads.
- Package constants (blend weights, fusion factors) were fitted on the old models.
  - Label-free constants (alignment mean / sd) can be re-fitted on the members' own out-of-fold outputs.
  - Labelled weight re-tunes must be cross-fitted, and the re-tuned blend must beat the SHIPPED blend, not only a re-tuned control. On 10-21, re-tunes lost to the shipped weights for old and new members alike.
- Keep in mind: out-of-fold predictions average 1-2 models per study, while test averages many. Fusion and gating effects can shrink on test.
- Wording: "no gain detected (model, folds, seeds)" for weak screens; "negative" only for repeated losses.
- Slow-learning model families (VLMs, large ViTs) are judged on their own learning curve, not at a small CNN's step count.

## Ensembles: is a new model a good addition?
- It beats the old version it replaces, across several seeds. Best if the old version was the strongest member.
- Expected ensemble gain ≈ member gain × its share of the ensemble. Do not expect one member to move a large ensemble alone. A gain that carries over in proportion is a positive result.
- Compare a swap with a seed swap: the same slot filled by another seed of the old member.
- Absorption check: swap it into re-fitted ensembles of growing size.
  - Good sign: the gain survives at about its share, even in small ensembles.
  - Bad sign: you keep replacing more members with new models, but each gain is absorbed by the others and the ensemble does not move.
- Lower correlation with the rest of the ensemble usually means more added value.
- Final check: all members retrained in the new regime, and the re-fitted ensemble beats the old one.

## LB probes
- The LB is the final judge for targeted questions.
- One assumption per probe is preferred.
- Splice design: the deployed pipeline for every other class, the new pipeline for the tested class only.
- The LB is the mean of 12 class AUCs, shown with 3 decimals. A one-class probe only shows changes larger than about 0.012 in that class's AUC.

## After a result
- Positive arm: follow it up with variants, more ensemble members, more seeds, until the follow-ups stop improving.
- Negative first fold: at most one confirmation fold.

## Change log (user-approved edits only)
- 2026-10-16: first version.
- 2026-10-21: yardstick per class (user-approved). One label algorithm per class is both the training target and the referee. Dense and read-only views. Gold / text / trusted / lookat / host-like become tie-breakers and error catchers that never block. Refit rule clarified.
- 2026-10-21 (user correction): one decision metric = the new-label model vs its old-label counterpart on the yardstick, then ensemble swap, cross-fitted re-fit and more members, in that order. Effusion yardstick row corrected to the pool recipe.
- 2026-10-21 (user-approved): the read-only view decides; the dense view moves to the error catchers.
