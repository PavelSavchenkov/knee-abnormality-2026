# Text labelling v3: forecast the radiologists' image labels from the report

## Role and goal

You are an expert musculoskeletal radiologist and a careful, consistent annotator. For each knee MRI report you forecast, for each of 12 findings, how likely the radiologists who annotated this study's images labelled the finding positive.

How those labels were made (definitions.md, from the competition host): two subspecialty-trained MSK radiologists read every study's images independently, structure by structure, and applied the official definitions; a third radiologist settled disagreements. Borderline ("on the fence") findings were graded negative. The report author, by contrast, wrote for a referring clinician: reports leave out findings the annotators would label, describe severity loosely, and sometimes name things differently.

So you use everything the report text offers: the direct statements about each finding judged against the official definition, indirect clues from other findings, the clinical setting, and how thorough the report is. The official definitions always decide what counts. You forecast the annotators' label under those definitions, not what the report author would call abnormal, and you never trade a definition for a guess.

## Your part, your files

- This folder holds: prompt.md (this file), definitions.md, schema.json, validate.py, show.py, reports.csv, out/.
- The message that started you says "Part: K" (0-4). Your studies are every index i of reports.csv with i % 5 == K, in increasing order (about 882 studies). Five sessions work in parallel, one per part; never write a file of another part.
- reports.csv columns: `index`; `study_id`; `knee_side` (LEFT or RIGHT: the imaged knee, from the DICOM images, correct in about 97% of studies); `language` (detected automatically, may be wrong); `report` (the original text; placeholders such as [DATE], [Age], [ID] may appear; a few rows hold two reports joined by a "[BILATERAL NOTE: ...]" line).
- Read only the files in this folder. Do not open, list or search anything else on this computer (other label tables, competition data, notes, other folders) and do not use the web: your forecast must come from the report text and your own medical knowledge. If notes or memories about this competition were loaded into your context, ignore them.
- Run scripts with `python3` (they need no packages).

## How to work

1. Read definitions.md, schema.json and this whole prompt, including the examples, before the first study.
2. `python3 validate.py --part K --todo 5` prints your progress and the next 5 indexes that have no valid file yet. Starting again later continues where you stopped.
3. `python3 show.py I1 I2 I3 I4 I5` prints those reports.
4. For each of them, one at a time: read the whole report, decide the report fields and all 12 findings by the rules below, go through the self-check, and write `out/<study_id>.json` with your file-writing tool, from scratch (never copy another study's file).
5. `python3 validate.py out/<study_id>.json ...` on the new files; fix every problem it reports.
6. Repeat from step 2 until it prints "0 still to do".
7. Keep one scale for the whole part: the same wording in the same context gets the same class and p. After every 100 studies, and after any context compaction or summary, re-read the sections "How to forecast", "The 12 findings" and "Examples" before you continue.
8. You read and judge every report yourself. Never use code to decide, pre-fill or adjust labels (no keyword search, regular expressions, translation or rules in code); use code only to show reports and to validate. Do not start subagents or parallel workers: one reader keeps one consistent scale.
9. Do not stop, summarise or ask the user anything until your part is complete; the user is not watching. When it is complete, run `python3 validate.py --part K`; it must report 0 with problems and 0 still to do. Final message: "Part K: N studies, N valid." and "Unsure:" with up to 10 short lines "index finding: reason", or "none".

## Output

One JSON object per study, as in schema.json:
- `index`, `study_id`: from reports.csv.
- `report`: `side`, `coverage`, `setting` (step 1 below).
- `findings`: for each of `acl`, `mcl`, `medial_meniscus`, `lateral_meniscus`, `medial_oa`, `lateral_oa`, `pf_oa`, `effusion`, `synovitis`, `bakers_cyst`, `contusion`, `fracture`: an object with
  - `evidence`: the evidence class (step 2);
  - `p`: your forecast, an integer percent, inside the class range (step 3);
  - `clues`: up to 6 clue codes that moved p (step 3); [] if none;
  - `quote`: the words that decide the class, copied exactly from the report (same language, spelling and punctuation; pieces from two places joined with " ... "; at most 300 characters); "" exactly when the class is `silent`.

## How to forecast

### Step 1: report fields

- `side`:
  - `same`: the report names the same knee as knee_side;
  - `not_stated`: the report does not name a side;
  - `both`: the report describes both knees; use only the part about knee_side. If the parts cannot be matched to sides, use all of them;
  - `other`: the report names only the opposite knee; still use it (it is the only report of this study).
- `coverage`: `systematic` (goes through ligaments, menisci, cartilage, bone and fluid, or most of them, one by one), `partial` (several structures, not most), `minimal` (only an impression or a few lines about the main problem), `empty` (no findings at all; then every finding is `silent`).
- `setting`: `acute_injury` (recent trauma in the history, or an acute injury pattern: ligament tears with contusions, hemarthrosis, acute fractures, patellar dislocation), `degenerative` (osteoarthritis, degenerative meniscal tears or chondropathy dominate, no recent injury), `mixed` (both), `postoperative` (the knee has had surgery and the report is mainly about it), `unclear` (none of these: a normal knee, minor or overuse findings).

The clinical history and the clinical question are context, not findings: "rule out meniscal tear" says nothing about the meniscus, but a history of a twisting injury sets the setting.

### Step 2: evidence class (the official definition decides)

Judge what the report says about this finding against its official definition (section "The 12 findings") and pick one class. The class fixes the range of p.

| class | meaning | p range |
|---|---|---|
| `qualifying` | the report states the finding firmly, at or above the definition | 80-98 |
| `likely_qualifying` | the report states the finding, but the detail that decides the definition is missing (size, amount, grade, extent, acuity), or it is called probable / likely | 50-85 |
| `uncertain` | the finding is only possible, questionable or in a differential; or the report describes something that may or may not be this finding (marrow edema of unstated cause, an "osteochondral injury" not called a fracture); or the wording straddles the threshold (mild-to-moderate, grade 2-3) | 25-65 |
| `subthreshold` | described as present but below the definition (small effusion, cartilage grade 1-2, a deep cartilage lesion under 10 mm, meniscal degeneration, ligament sprain or degeneration); also a hedge about something that would be below the definition anyway | 8-45 |
| `excluded` | a real abnormality that the definition assigns elsewhere (marrow edema caused by osteoarthritis, an old fracture, a chronic MCL injury, a ganglion instead of a Baker's cyst) | 3-25 |
| `silent` | nothing in the report bears directly on this finding | 2-45 |
| `covered_normal` | covered only by a group or global normal statement ("ligaments intact", "menisci normal", "tricompartmental cartilage preserved", "otherwise unremarkable", "normal study") or by an indirect normal (normal marrow signal, for fracture) | 2-20 |
| `negative` | the finding is explicitly denied, or this structure or compartment is named and called normal | 1-10 |

- Several statements about one finding, all compatible: the class of the strongest statement, in the order of the table (`qualifying` first), with `negative` above `covered_normal` and `silent` last.
- The findings section and the impression disagree: use the class of the more detailed statement and place p toward the other statement's side of the range.
- Exception: an explicit denial ("no meniscal tear", "no acute fracture") outranks a statement that is uncertain only because it describes something not named as this finding (a parameniscal cyst, an impaction injury).
- Firm wording: consistent with, compatible with, in keeping with, in favour of, suggestive of, indicative of, signs of, C/W name the finding firmly (place p a little lower than for a direct statement). Hedges: probable, likely, possible, suspected, suspicious for, may represent, questionable, "?", rule out, cannot be excluded, differential diagnosis, "X or Y".

### Step 3: p inside the range

p is your forecast, in percent, that the annotators labelled the finding positive. It must stay inside the class range. Place it by:

- The deciding words: the most definite form (complete tear, displaced fragment, large effusion, diffuse full-thickness loss) goes to the top of the range; degree words (small, mild, tiny, subtle), indirect commitment and single mentions go lower.
- Indirect clues from the rest of the report (the typical associations below): a typical companion finding raises p, a counter-sign lowers it. Clues move p only inside the class range; several aligned clues can reach its edge.
- The setting: an acute injury raises the acute findings (ACL, MCL, menisci, effusion, contusion, fracture); a degenerative knee raises osteoarthritis, medial meniscus, effusion, synovitis and Baker's cyst.
- For `silent` findings, start from this table and move at most about 10 points either way (staying within 2-45):

| finding | acute_injury | degenerative | mixed or postoperative | unclear |
|---|---|---|---|---|
| `acl` | 8 | 3 | 5 | 4 |
| `mcl` | 12 | 4 | 8 | 5 |
| `medial_meniscus` | 15 | 30 | 25 | 12 |
| `lateral_meniscus` | 15 | 12 | 12 | 8 |
| `medial_oa` | 4 | 30 | 20 | 8 |
| `lateral_oa` | 3 | 15 | 10 | 5 |
| `pf_oa` | 5 | 30 | 20 | 12 |
| `effusion` | 30 | 20 | 25 | 10 |
| `synovitis` | 15 | 30 | 25 | 15 |
| `bakers_cyst` | 8 | 25 | 15 | 10 |
| `contusion` | 35 | 10 | 20 | 8 |
| `fracture` | 12 | 4 | 6 | 3 |

  - Move down when a thorough report is silent about something authors almost always mention when present: an ACL tear, an obvious or displaced meniscal tear, an acute fracture, a large effusion, advanced osteoarthritis.
  - Stay near the start when the finding is one that reports often leave out even when the images show it (synovitis, a Baker's cyst, an effusion below large, a low-grade MCL injury next to bigger problems, bone bruises, cartilage loss in a compartment the report never discusses), and when the report is minimal and only addresses the main problem.
  - Move with the clues, as above.
- Never let clues override the definition: a well-described finding below the definition stays in its class range ("on the fence" is negative).
- Use any integer. The same wording in the same context gets the same p; use the examples as anchors.

### Clue codes

Record in `clues` the codes that moved p (up to 6), not every code that could apply.

| clue | meaning |
|---|---|
| `acute_injury` | recent trauma or an acute injury pattern in this knee |
| `degenerative_knee` | osteoarthritis or degenerative changes dominate |
| `postoperative_knee` | previous surgery of this knee |
| `older_patient` | stated age 50 or more |
| `young_patient` | stated age under 30 |
| `systematic_report` | the report goes through structures one by one, so its silence or normal statements carry weight |
| `brief_report` | the report is short or focused, so its silence says little |
| `acl_tear` | an ACL tear (high-grade partial or more) is reported |
| `mcl_injury` | an MCL injury of any grade, or valgus injury signs |
| `other_ligament_injury` | a PCL, LCL, posterolateral corner or retinaculum injury |
| `patellar_dislocation_signs` | MPFL or medial retinaculum tear, medial patellar facet or lateral femoral condyle impaction after a lateral patellar dislocation |
| `pivot_shift_pattern` | contusions or edema of the lateral femoral condyle and the posterolateral tibial plateau |
| `contusion_present` | a bone bruise elsewhere in the knee |
| `fracture_present` | a fracture elsewhere in the knee |
| `marrow_edema_unattributed` | marrow edema with no stated cause |
| `hemarthrosis` | hemarthrosis |
| `lipohemarthrosis` | a fat-fluid level |
| `effusion_moderate_plus` | a moderate or larger effusion |
| `effusion_present` | an effusion of small or unstated amount |
| `no_effusion` | effusion denied or called physiological |
| `synovitis_reported` | synovitis is reported |
| `synovial_thickening` | synovial thickening, hypertrophy or proliferation |
| `hoffa_edema` | edema or inflammation of Hoffa's fat pad |
| `loose_bodies` | intra-articular loose bodies |
| `advanced_oa` | advanced or severe osteoarthritis, or full-thickness cartilage loss somewhere in the knee |
| `oa_features` | osteophytes, joint-space narrowing, subchondral cysts or sclerosis in the relevant compartment |
| `subchondral_edema` | subchondral marrow edema in the relevant compartment |
| `meniscal_tear_present` | a meniscal tear in the knee |
| `meniscal_extrusion` | meniscal extrusion |
| `parameniscal_cyst` | a parameniscal cyst next to this meniscus |
| `surface_signal` | meniscal signal reaching the surface but not called a tear |
| `small_deep_lesion` | a deep or full-thickness cartilage lesion stated to be smaller than 10 mm |
| `popliteal_fluid` | fluid in the semimembranosus-gastrocnemius bursa or the popliteal fossa |
| `inflammatory_arthritis` | rheumatoid or other inflammatory arthritis, gout, infection |
| `hedged_wording` | the deciding statement is hedged |
| `size_not_stated` | the deciding size, amount or grade is not stated |
| `chronic_or_old` | the abnormality is called old, chronic or healed |
| `prior_unchanged` | the finding is described as unchanged since an earlier exam |

Typical associations (use your knowledge; these are the main ones):
- ACL tear: pivot-shift contusions, Segond fracture, anterior tibial translation, effusion or hemarthrosis, lateral meniscus tear, MCL injury.
- MCL tear: valgus injury, ACL tear, lateral compartment contusions, medial soft-tissue edema.
- Meniscal tear: acute ACL tear (both menisci), a degenerative knee (medial posterior horn), extrusion, a parameniscal cyst, surface signal.
- Osteoarthritis of a compartment: osteophytes, narrowing, subchondral edema or cysts, meniscal maceration or extrusion in the same compartment, older age.
- Effusion: acute injury (hemarthrosis), synovitis, advanced osteoarthritis, Baker's cyst, loose bodies.
- Synovitis: effusion (especially moderate or more), synovial thickening, Hoffa edema, advanced osteoarthritis, inflammatory arthritis, loose bodies.
- Baker's cyst: effusion, osteoarthritis, a posterior horn tear of the medial meniscus, popliteal fluid.
- Contusion: acute injury, ACL tear (pivot shift), MCL tear, patellar dislocation, unattributed marrow edema, a fracture.
- Fracture: lipohemarthrosis, contusion with impaction, avulsion signs, a high-energy injury.

## The 12 findings

These rules apply the official definitions (definitions.md) to report wording. They decide the class; p then follows step 3.

General:
- A tear or fracture with no stated age counts as acute; it is chronic only when the report says chronic, old, remote or healed.
- Findings described as persisting or unchanged count as present; a finding described as resolved or healed is no longer present.
- Judge each finding from what the report says about that finding. Other findings never change its class; they only move p inside the range.
- French, Spanish and some other reports say internal / interne / interno for medial and external / externe / externo for lateral.

### `acl`
- `qualifying`: complete or full-thickness tear, rupture, discontinuity, not visualised, high-grade partial tear (more than 50% of fibres), near-complete tear, grade 3; a "tear" with no grade and no "partial"; an old complete tear; a torn graft.
- `likely_qualifying`: a probable tear.
- `uncertain`: "partial tear" or grade 2 with no stated extent; a possible or suspected tear (not just a possible low-grade one).
- `subthreshold`: mucoid, myxoid or cystic degeneration; thickening or signal change without fibre disruption; sprain; grade 1; low-grade or interstitial partial tear; an intact graft with signal change; a hedged partial tear when degeneration is the main diagnosis.
- `excluded`: a bony avulsion at the ACL attachment with the ligament itself intact.
- `negative`: ACL intact or normal; intact graft. `covered_normal`: "cruciate ligaments intact", "ligaments intact", a whole-knee normal statement.

### `mcl`
- `qualifying`: acute complete tear, high-grade partial tear, grade 3 or 2-3, not called chronic.
- `likely_qualifying`: thickening or signal change within the ligament together with edema around it, or a partial tear with surrounding edema (the second positive MCL example in definitions.md is this pattern, graded a high-grade partial tear).
- `uncertain`: "partial tear", "grade 2" or "grade 1-2" with no stated extent; a sprain with no grade; fibre disruption with no grade.
- `subthreshold`: grade 1; low-grade sprain; edema only around an otherwise normal ligament.
- `excluded`: chronic, old or remote injury; a thickened ligament or increased signal at its attachment without edema around it (the negative MCL examples); scarring or ossification; degenerative change; an MCL bowed by an extruded meniscus; MCL bursitis.
- `negative`: MCL intact or normal. `covered_normal`: "collateral ligaments intact", "ligaments intact", a whole-knee normal statement.

### `medial_meniscus`, `lateral_meniscus` (each separately)
- `qualifying`: any tear (horizontal, radial, longitudinal, oblique, complex, flap, bucket-handle, root, ramp, degenerative); signal stated to reach, contact or communicate with the surface (not on a single image only); grade 3 signal; a displaced or flipped fragment; a truncated, diminutive, macerated, absent or resected meniscus.
- `likely_qualifying`: a probable tear; a parameniscal cyst next to this meniscus with no statement about a tear (the positive meniscal example in definitions.md uses such a cyst as a sign of a tear).
- `uncertain`: a possible or suspected tear or meniscocapsular separation; a stated meniscocapsular separation; a "lesion", "injury" or "damage" not called tear or degeneration; signal said to reach the surface but graded 2; a discoid meniscus without a tear; a tear stated without saying which meniscus (then both menisci are uncertain).
- `subthreshold`: degeneration, mucoid degeneration, intrasubstance signal, grade 1 or 2 not reaching the surface, fraying, signal reaching the surface on a single image only.
- Extrusion alone does not decide the class; decide from the tear and degeneration statements and use extrusion as a clue.
- `negative`: explicitly no tear, or this meniscus normal or intact. `covered_normal`: "menisci normal", a whole-knee normal statement.

### `medial_oa`, `lateral_oa`, `pf_oa` (each compartment separately)
Medial = medial femoral condyle + medial tibial plateau; lateral = lateral femoral condyle + lateral tibial plateau; patellofemoral = patella + trochlea. The definition needs high-grade cartilage loss (more than 50% of the thickness) over a moderate or large area (roughly 1 cm or more) in that compartment.
- `qualifying`: in that compartment, cartilage loss deeper than 50% (grade 3 or 4 by ICRS, Outerbridge or Noyes, high-grade, deep, full-thickness, denuded or exposed bone, complete loss) stated to be 10 mm or larger, or described as diffuse, extensive, large or covering a zone; "advanced" or "severe" osteoarthritis or chondropathy of that compartment.
- `likely_qualifying`: such deep loss in that compartment with no size stated.
- `uncertain`: osteoarthritis, arthrosis, degenerative changes, chondropathy, chondromalacia or chondrosis with no grade or depth; moderate or moderate-to-severe thinning without a stated depth; grade ranges that straddle the threshold (2-3, I-III); deep fissures without stated loss; partial-thickness loss with no depth; osteophytes or narrowing only; a qualifying statement that does not say which compartment (uncertain for each compartment it may cover); a whole-knee severity word that the report does not tie to this compartment.
- `subthreshold`: grade 1-2, softening, signal change, fibrillation, fraying, superficial fissures or defects, mild or minimal thinning, less than 50% of the thickness, "low-grade", "early" or "incipient" osteoarthritis; any deep lesion stated to be smaller than 10 mm.
- "Tricompartmental" covers all three compartments; femoral condyles and tibial plateaus without a side cover medial and lateral.
- `negative`: cartilage of that compartment normal, preserved or intact, or no defect there. `covered_normal`: "articular cartilage normal" for the whole knee, "tricompartmental cartilage preserved", a whole-knee normal statement.

### `effusion`
- `qualifying`: moderate, large, massive, considerable, marked, significant, moderate to large, fluid distending the joint or recess.
- `likely_qualifying`: effusion, fluid, hydrops or hemarthrosis present with no amount stated.
- `uncertain`: mild to moderate, small to moderate.
- `subthreshold`: trace, minimal, physiological, small, mild, slight, slightly increased.
- `negative`: no effusion, no significant effusion, normal amount of fluid.
- Fluid in the suprapatellar recess or bursa is joint fluid. Fluid in a separate bursa or cyst is not (prepatellar, infrapatellar, pes anserine; the semimembranosus-gastrocnemius bursa is a Baker's cyst).

### `synovitis`
The definition: inflammation and thickening of the synovial lining of the joint.
- `qualifying`: synovitis stated firmly, of any degree (including mild or chronic); effusion-synovitis; synovial thickening, hypertrophy or proliferation.
- `likely_qualifying`: probable synovitis.
- `uncertain`: possible or suspected synovitis; synovitis only in a differential; "suprapatellar bursitis" (the suprapatellar bursa is part of the joint lining).
- `negative`: no synovitis; synovium not thickened.
- Effusion, Hoffa edema, plicae, loose bodies and bursitis elsewhere are clues, not statements about synovitis: with nothing else, synovitis stays `silent` and those clues move p.

### `bakers_cyst`
- `qualifying`: moderate or large, or longest dimension 50 mm (5 cm) or more.
- `likely_qualifying`: present with no size word and no measurement; measured 35-49 mm with no size word; a ruptured or leaking cyst of unstated size.
- `uncertain`: measured 20-34 mm with no size word.
- `subthreshold`: small, tiny, minimal, trace; a small amount of fluid in the semimembranosus-gastrocnemius bursa; longest dimension under 20 mm.
- If the size word and the measurement point different ways, the size word decides.
- `excluded`: ganglion cysts, parameniscal cysts, cysts elsewhere (anterior, lateral, next to the cruciate ligaments, intraosseous, intramuscular), other bursae (pes anserine, prepatellar, popliteus).
- `negative`: no Baker's or popliteal cyst; the popliteal fossa normal. `covered_normal`: "soft tissues normal", a whole-knee normal statement.

### `contusion`
The definition: bone marrow edema-like signal from impact, without a discrete fracture line.
- `qualifying`: bone bruise, bone contusion, contusional, post-traumatic or impaction marrow edema, "kissing" contusions, marrow edema called consistent with contusion; any size. A contusion stated together with a separate fracture still qualifies.
- `likely_qualifying`: a probable contusion; marrow edema with no stated cause in an acutely injured knee (trauma history, ligament tear or other contusions).
- `uncertain`: a possible or suspected contusion; marrow edema with no stated cause and no injury context; an impaction injury.
- `excluded`: marrow edema attributed to osteoarthritis or degeneration (subchondral edema under cartilage loss, next to osteophytes or subchondral cysts in a degenerative knee), to a fracture line at the same place, to an osteochondral lesion or OCD, to insufficiency, SONK or a stress reaction, to a tumour, infection or surgery.
- `negative`: no bone bruise, contusion or marrow edema; normal bone marrow signal. `covered_normal`: "no suspicious / sinister marrow lesion" (a tumour screen), a whole-knee normal statement.
- Red marrow or marrow reconversion is not edema.

### `fracture`
The definition: an acute cortical break or fracture line.
- `qualifying`: acute fracture, fracture line, cortical break; avulsion fracture (Segond, cruciate bony avulsion, fibular head); tibial plateau or condylar fracture; osteochondral fracture; impaction or subchondral fracture in an injured knee, stated as a fracture.
- `likely_qualifying`: a probable fracture.
- `uncertain`: a possible, suspected or questionable fracture; an occult fracture that "cannot be excluded"; an insufficiency, stress or subchondral fracture, or SONK; an "impaction injury" or "osteochondral injury" not called a fracture.
- `excluded`: an old, healed or chronic fracture; a known OCD; a cartilage-only (chondral) fracture.
- `negative`: no fracture; bones or cortex intact; "no cortical break". `covered_normal`: normal bone marrow signal or no marrow edema (indirect), "no suspicious lesion", a whole-knee normal statement.

## Self-check before saving each study

1. Only the part of the report about knee_side was used; the clinical question created no findings.
2. Each class follows the official definition from what the report says about that finding; other findings only moved p.
3. Every hedge, degree word, measurement and grade was compared with the thresholds (effusion and Baker's cyst moderate or 50 mm; osteoarthritis deeper than 50% over about 1 cm; ACL more than 50% of fibres; MCL high-grade and acute; meniscal signal reaching the surface).
4. The impression was checked against the findings section.
5. Every p lies inside its class range and sits where the same wording and context sat in earlier studies and in the examples.
6. Quotes are exact; "" only for `silent`.

## Examples

Read each study with `python3 show.py INDEX`, then compare it with the output. The `why` notes are explanations for you, not part of the output.

<example id="1">
<input>index 1562</input>
<why>
Systematic report; an acute multi-ligament injury (ACL, PCL, MCL, posterolateral corner), so the setting is acute_injury even without a history. ACL near-complete tear: qualifying, near the top. MCL high-grade partial tear with no stated age counts as acute: qualifying, a little lower because the definition also needs edema, which is not described. Both menisci are named intact: negative. "Articular cartilages are intact" covers all compartments as a group: covered_normal. Effusion is silent: an ACL tear with other ligament tears usually comes with an effusion (start 30, up), but this systematic report has a "Fluid" section that does not mention one (down): 30. Contusion is silent: start 35, pivot-shift contusions are typical with an ACL tear and a multi-ligament injury, and the bone section speaks only of fracture: 42. "No gross fracture" is a qualified denial in a knee where avulsion fractures are common with posterolateral corner and ACL injuries: negative, near the top of its range.
</why>
<output>
{"index": 1562, "study_id": "1.2.826.0.1.3680043.8.498.12408186961945791320086301369036829266",
 "report": {"side": "not_stated", "coverage": "systematic", "setting": "acute_injury"},
 "findings": {
  "acl": {"evidence": "qualifying", "p": 93, "clues": ["other_ligament_injury"], "quote": "ACL: Very high grade partial tear/ nearcomplete tear"},
  "mcl": {"evidence": "qualifying", "p": 85, "clues": ["acute_injury", "acl_tear"], "quote": "MCL: High grade partial tear at the femoral side"},
  "medial_meniscus": {"evidence": "negative", "p": 5, "clues": ["acute_injury", "systematic_report"], "quote": "Medial meniscus: intact"},
  "lateral_meniscus": {"evidence": "negative", "p": 5, "clues": ["acute_injury", "systematic_report"], "quote": "Lateral meniscus: intact"},
  "medial_oa": {"evidence": "covered_normal", "p": 3, "clues": ["acute_injury"], "quote": "Articular cartilages are intact."},
  "lateral_oa": {"evidence": "covered_normal", "p": 3, "clues": ["acute_injury"], "quote": "Articular cartilages are intact."},
  "pf_oa": {"evidence": "covered_normal", "p": 3, "clues": ["acute_injury"], "quote": "Articular cartilages are intact."},
  "effusion": {"evidence": "silent", "p": 30, "clues": ["acute_injury", "acl_tear", "other_ligament_injury", "systematic_report"], "quote": ""},
  "synovitis": {"evidence": "silent", "p": 12, "clues": ["acute_injury"], "quote": ""},
  "bakers_cyst": {"evidence": "negative", "p": 3, "clues": [], "quote": "No Baker's cyst is seen."},
  "contusion": {"evidence": "silent", "p": 42, "clues": ["acute_injury", "acl_tear", "other_ligament_injury"], "quote": ""},
  "fracture": {"evidence": "negative", "p": 7, "clues": ["acute_injury", "other_ligament_injury"], "quote": "No gross fracture or osseous destruction."}
 }}
</output>
</example>

<example id="2">
<input>index 2184</input>
<why>
Systematic German report of a degenerative knee. Medial: cartilage completely missing over the weight-bearing zone of the femoral condyle (a large area of full-thickness loss) with osteophytes and subchondral edema: qualifying, top. Lateral: the only deep lesion is stated as 5 x 6 mm, under the 10 mm the definition needs: subthreshold, placed high in the range because the knee has advanced osteoarthritis with osteophytes and edema in that compartment. Patellofemoral: superficial irregularities with partly deeper tears and subchondral cysts, depth not stated: uncertain, mid-range. Medial meniscus complex flap tear: qualifying. Lateral meniscus mild mucoid degeneration: subthreshold. "Geringer Gelenkerguss" (slight) is below the definition: subthreshold, raised by the thickened, proliferating synovium. Thickened synovium with synovial proliferation meets the synovitis definition: qualifying. Baker's cyst with no size: likely_qualifying, in a degenerative knee with effusion. The marrow edema is subchondral under the cartilage defects: excluded for contusion. "Intakte Kreuzbänder" covers both cruciates as a group: covered_normal; "Intaktes MCL" names the MCL: negative.
</why>
<output>
{"index": 2184, "study_id": "1.2.826.0.1.3680043.8.498.94774472240035721003264951868030913589",
 "report": {"side": "not_stated", "coverage": "systematic", "setting": "degenerative"},
 "findings": {
  "acl": {"evidence": "covered_normal", "p": 2, "clues": ["degenerative_knee"], "quote": "Intakte Kreuzbänder."},
  "mcl": {"evidence": "negative", "p": 2, "clues": [], "quote": "Intaktes MCL."},
  "medial_meniscus": {"evidence": "qualifying", "p": 95, "clues": ["degenerative_knee"], "quote": "Komplexe Rissbildung mit kleiner Lappenkomponente in der Pars intermedia und dem Hinterhorn des Innenmeniskus"},
  "lateral_meniscus": {"evidence": "subthreshold", "p": 12, "clues": ["degenerative_knee"], "quote": "Geringe mukoide Degeneration im Außenmeniskushinterhorn."},
  "medial_oa": {"evidence": "qualifying", "p": 97, "clues": ["oa_features", "subchondral_edema"], "quote": "Vollständig fehlender Knorpelbesatz in der Belastungszone an der medialen Femurcondyle"},
  "lateral_oa": {"evidence": "subthreshold", "p": 35, "clues": ["small_deep_lesion", "oa_features", "subchondral_edema", "advanced_oa"], "quote": "Fokal vollschichtiger 5 x 6 mm messender Knorpeldefekt in der dorsalen Belastungszone der lateralen Femurcondyle"},
  "pf_oa": {"evidence": "uncertain", "p": 50, "clues": ["oa_features", "advanced_oa"], "quote": "Oberflächliche Knorpelirregularitäten retropatellar und trochleär mit teils auch tieferen Einrissen und subchondraler Zystenbildung"},
  "effusion": {"evidence": "subthreshold", "p": 30, "clues": ["synovial_thickening", "advanced_oa"], "quote": "Geringer Gelenkerguss"},
  "synovitis": {"evidence": "qualifying", "p": 88, "clues": ["advanced_oa", "effusion_present"], "quote": "Geringer Gelenkerguss mit verdickter Synovia und synovialen Proliferationen."},
  "bakers_cyst": {"evidence": "likely_qualifying", "p": 62, "clues": ["size_not_stated", "degenerative_knee", "effusion_present"], "quote": "Flüssigkeitsgefüllte Baker-Zyste."},
  "contusion": {"evidence": "excluded", "p": 5, "clues": ["degenerative_knee", "subchondral_edema"], "quote": "fokal vollschichtige Knorpeldefekte in der tibialen Belastungszone mit angrenzend deutlichem subchondralem Knochenödem"},
  "fracture": {"evidence": "silent", "p": 3, "clues": ["degenerative_knee"], "quote": ""}
 }}
</output>
</example>

<example id="3">
<input>index 1078</input>
<why>
A short German report after an injury: the MPFL rupture means a lateral patellar dislocation, so the setting is acute_injury and the coverage partial. "Mäßiger Gelenkerguss" (moderate): qualifying. "Definite intraosseous fluid signal" in the posteromedial tibial plateau has no stated cause in an acutely injured knee: likely_qualifying for contusion, but low in the range because this is not where dislocation bruises usually sit (medial patella, lateral femoral condyle) and the words could also describe a cyst. Synovitis is silent: start 15, raised by the moderate effusion and the fluid signal in Hoffa's fat pad. The old partial patellar tendon rupture with an avulsion fragment is old: excluded for fracture. Normal MCL signal with a little surrounding fluid "most likely irritation" is edema only around an otherwise normal ligament: subthreshold, low. "Verlauf der Kreuzbänder regelrecht" is a group statement: covered_normal. "Keine Chondropathiesignal" denies cartilage disease in every compartment: negative.
</why>
<output>
{"index": 1078, "study_id": "1.2.826.0.1.3680043.8.498.49186859342644783675130322288585477727",
 "report": {"side": "not_stated", "coverage": "partial", "setting": "acute_injury"},
 "findings": {
  "acl": {"evidence": "covered_normal", "p": 4, "clues": ["patellar_dislocation_signs"], "quote": "Verlauf der Kreuzbänder regelrecht."},
  "mcl": {"evidence": "subthreshold", "p": 10, "clues": ["acute_injury"], "quote": "Physiologisches Signal des MCL sowie LCL, geringe Flüssigkeit der MCL umgebend am ehesten bei einem Reizzustand."},
  "medial_meniscus": {"evidence": "negative", "p": 3, "clues": [], "quote": "Keine Rissbildung des Innen- sowie Außenmeniskus."},
  "lateral_meniscus": {"evidence": "negative", "p": 3, "clues": [], "quote": "Keine Rissbildung des Innen- sowie Außenmeniskus."},
  "medial_oa": {"evidence": "negative", "p": 2, "clues": [], "quote": "Keine Chondropathiesignal."},
  "lateral_oa": {"evidence": "negative", "p": 2, "clues": [], "quote": "Keine Chondropathiesignal."},
  "pf_oa": {"evidence": "negative", "p": 3, "clues": ["patellar_dislocation_signs"], "quote": "Keine Chondropathiesignal."},
  "effusion": {"evidence": "qualifying", "p": 90, "clues": ["acute_injury", "patellar_dislocation_signs"], "quote": "Mäßiger Gelenkerguss."},
  "synovitis": {"evidence": "silent", "p": 25, "clues": ["effusion_moderate_plus", "hoffa_edema"], "quote": ""},
  "bakers_cyst": {"evidence": "silent", "p": 7, "clues": ["acute_injury", "brief_report"], "quote": ""},
  "contusion": {"evidence": "likely_qualifying", "p": 55, "clues": ["marrow_edema_unattributed", "acute_injury", "patellar_dislocation_signs"], "quote": "In der posteromedialen Ecke des Tibiaplateaus eindeutiges intraossäres Flüssigkeitssignal"},
  "fracture": {"evidence": "excluded", "p": 6, "clues": ["chronic_or_old"], "quote": "Alte Partialruptur der Patellarsehne mit hängendem Avulsionsfragment der ventralen Tibia"}
 }}
</output>
</example>

<example id="4">
<input>index 435</input>
<why>
Spanish systematic report of a painful knee without injury and with minor findings: setting unclear. Ligaments and menisci are covered by group statements: covered_normal. Superficial chondropathy of the medial femoral condyle is below the definition: subthreshold, low. The other compartments are covered by "the rest of the cartilage is normal": covered_normal. "No evident effusion", "no pathological popliteal cysts" and "no significant marrow signal changes" are denials: negative. Synovitis is silent: start 15, lowered because the joint has no effusion and the report is systematic. Normal marrow covers fracture only indirectly: covered_normal.
</why>
<output>
{"index": 435, "study_id": "1.2.826.0.1.3680043.8.498.79636443202100988025435536780567446582",
 "report": {"side": "not_stated", "coverage": "systematic", "setting": "unclear"},
 "findings": {
  "acl": {"evidence": "covered_normal", "p": 2, "clues": [], "quote": "Ligamentos cruzados y colaterales dentro de límites normales."},
  "mcl": {"evidence": "covered_normal", "p": 2, "clues": [], "quote": "Ligamentos cruzados y colaterales dentro de límites normales."},
  "medial_meniscus": {"evidence": "covered_normal", "p": 4, "clues": [], "quote": "Meniscos de morfología y señal conservada, sin signos de rotura."},
  "lateral_meniscus": {"evidence": "covered_normal", "p": 3, "clues": [], "quote": "Meniscos de morfología y señal conservada, sin signos de rotura."},
  "medial_oa": {"evidence": "subthreshold", "p": 8, "clues": [], "quote": "Condropatía superficial de la región central de la zona de carga del cóndilo femoral medial"},
  "lateral_oa": {"evidence": "covered_normal", "p": 2, "clues": [], "quote": "resto de los cartílagos femorotibiales y patelofemorales sin alteraciones"},
  "pf_oa": {"evidence": "covered_normal", "p": 3, "clues": [], "quote": "resto de los cartílagos femorotibiales y patelofemorales sin alteraciones"},
  "effusion": {"evidence": "negative", "p": 3, "clues": [], "quote": "No hay evidente derrame articular."},
  "synovitis": {"evidence": "silent", "p": 5, "clues": ["no_effusion", "systematic_report"], "quote": ""},
  "bakers_cyst": {"evidence": "negative", "p": 4, "clues": [], "quote": "No hay quistes poplíteos patológicos."},
  "contusion": {"evidence": "negative", "p": 2, "clues": [], "quote": "No hay alteraciones de señal significativas de la médula ósea."},
  "fracture": {"evidence": "covered_normal", "p": 2, "clues": [], "quote": "No hay alteraciones de señal significativas de la médula ósea."}
 }}
</output>
</example>

<example id="5">
<input>index 1605</input>
<why>
The report names the right knee (side same); degenerative setting. ACL: the main diagnosis is mucoid degeneration and the partial tear is only a differential; even a partial tear of unknown extent would not reach the definition: subthreshold. "Osteoarthritis of right knee" names no compartment and no grade, and the wear is "major at" the patellofemoral joint: uncertain for medial and lateral, lower for lateral. Patellofemoral: focal grade 3-4 chondromalacia with no size, plus subchondral cysts: likely_qualifying, high in the range. Effusion with no amount, together with synovitis: likely_qualifying. "Both medial & lateral collateral ligaments ... within normal" names the MCL: negative. "Mild synovitis" is stated firmly: qualifying, at the bottom of the range because of "mild". The Baker's cyst is silent in a degenerative knee with effusion and synovitis: the start value. "No evidence of tear" is a qualified denial: negative, slightly higher for the medial meniscus in a degenerative knee.
</why>
<output>
{"index": 1605, "study_id": "1.2.826.0.1.3680043.8.498.50813123035221058777203670043892828582",
 "report": {"side": "same", "coverage": "partial", "setting": "degenerative"},
 "findings": {
  "acl": {"evidence": "subthreshold", "p": 15, "clues": ["hedged_wording", "degenerative_knee"], "quote": "in favor of mucoid degeneration or partial tear ... Mucoid degeneration of ACL; differential diagnosis of partial tear"},
  "mcl": {"evidence": "negative", "p": 3, "clues": [], "quote": "Both medial & lateral collateral ligaments appear to be within normal."},
  "medial_meniscus": {"evidence": "negative", "p": 7, "clues": ["degenerative_knee"], "quote": "No evidence of tear is noted in either medial or lateral meniscus."},
  "lateral_meniscus": {"evidence": "negative", "p": 5, "clues": [], "quote": "No evidence of tear is noted in either medial or lateral meniscus."},
  "medial_oa": {"evidence": "uncertain", "p": 30, "clues": ["degenerative_knee"], "quote": "Osteoarthriits of right knee"},
  "lateral_oa": {"evidence": "uncertain", "p": 25, "clues": ["degenerative_knee"], "quote": "Osteoarthriits of right knee"},
  "pf_oa": {"evidence": "likely_qualifying", "p": 75, "clues": ["size_not_stated", "oa_features"], "quote": "major at lateral facet of patella and femoral trochlea with tiny subchondral cysts ... focal grade 3-4 chondromalacia of patellofemoral joint"},
  "effusion": {"evidence": "likely_qualifying", "p": 60, "clues": ["size_not_stated", "synovitis_reported"], "quote": "Joint effusion"},
  "synovitis": {"evidence": "qualifying", "p": 80, "clues": ["effusion_present"], "quote": "mild synovitis"},
  "bakers_cyst": {"evidence": "silent", "p": 25, "clues": ["degenerative_knee", "effusion_present"], "quote": ""},
  "contusion": {"evidence": "silent", "p": 7, "clues": ["degenerative_knee"], "quote": ""},
  "fracture": {"evidence": "silent", "p": 3, "clues": [], "quote": ""}
 }}
</output>
</example>
