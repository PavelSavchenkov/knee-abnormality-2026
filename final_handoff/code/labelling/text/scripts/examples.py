"""The worked examples embedded in text_labeling_bundle/prompt.txt (my labels under the bundle rules).
Checks them against schema.json and the report text, and prints them in the compact form used in the prompt."""
import json
import re
import unicodedata
from pathlib import Path

import jsonschema
import pandas as pd

B = Path("/home/pavel/Programming/knee-abnormality-2026/text_labeling_bundle")
NM = {"status": "not_mentioned", "missing": [], "quote": "", "probability": None}
OA0 = {"depth": None, "size_mm": None, "extent": None, "severity_word": None}
M0 = {"location": [], "displaced_fragment": None}


def f(status, quote, p, missing=(), **attrs):
    return {"status": status, "missing": list(missing), "quote": quote, "probability": p, **attrs}


EX = [
    {"index": 120, "study": {"report_side": "same", "report_type": "narrative", "refers_to_prior": False, "hoffa_edema": None}, "findings": {
        "acl": f("meets", "in favor of tear ... C/W ACL tear", 0.8, grade=None, acuity=None, graft=False),
        "mcl": f("below", "Interstitial increseased signal intensity of MCL on F/S T2WI, in favor of grade 1 injury", 0.2, grade="low_grade", acuity=None),
        "medial_meniscus": f("hedged", "R/O meniscocapsular separation at the interface of posterior horn of medisl meniscus", 0.3, location=["posterior_horn"], displaced_fragment=None),
        "lateral_meniscus": f("denied", "No evidence of tear is noted in either medial or lateral meniscus", 0.03, **M0),
        "medial_oa": {**NM, **OA0}, "lateral_oa": {**NM, **OA0}, "pf_oa": {**NM, **OA0},
        "effusion": f("unsettled", "Joint effusion and suprapatellar bursitis", 0.6, ["size"], size=None),
        "synovitis": f("meets", "hypertrophy of the synovium, indicative of synovitis ... Synovitis of right knee", 0.97),
        "bakers_cyst": f("unsettled", "mild infrapatellar bursitis and Baker's cyst", 0.6, ["size"], size=None, longest_mm=None),
        "contusion": f("meets", "in favor of bone contusion ... Bone contusion of femoral condyle and tibial plateau", 0.97, cause="trauma", bones=["femur_other", "tibia_other"]),
        "fracture": f("hedged", "Suspicious of subchondral fracture of lateral femoral condyle ... R/O subchondral fracture", 0.4, acuity=None)}},
    {"index": 198, "study": {"report_side": "same", "report_type": "narrative", "refers_to_prior": False, "hoffa_edema": None}, "findings": {
        "acl": f("structure_normal", "Ligamentos cruzados y colateral lateral sin alteraciones", 0.05, grade=None, acuity=None, graft=False),
        "mcl": f("unsettled", "discreta pérdida de la continuidad de su tercio proximal ... Signos de un esguince grado I - II", 0.45, ["grade"], grade="low_to_partial", acuity=None),
        "medial_meniscus": f("below", "Aumento de señal central del cuerno posterior y cuerpo del menisco medial que no impresiona contactar con la superficie articular, sin caracteres propios de rotura", 0.2, location=["posterior_horn", "body"], displaced_fragment=None),
        "lateral_meniscus": f("structure_normal", "Menisco lateral de morfología y señal conservada, sin signos de rotura", 0.02, **M0),
        "medial_oa": f("below", "Úlcera condral espesor total de la región central de carga del cóndilo femoral medial de 2 x 3 mm", 0.35, depth="full", size_mm=3, extent="focal", severity_word=None),
        "lateral_oa": f("structure_normal", "El resto de los cartílagos femorotibiales sin alteraciones", 0.05, **OA0),
        "pf_oa": f("meets", "Úlcera condral de espesor total con cambios óseos secundarios a la región central de la tróclea femoral y del vértice de la rótula ... Condropatía lesiones focales grado 4 de la articulación patelofemoral", 0.8, depth="full", size_mm=None, extent="multifocal", severity_word=None),
        "effusion": f("denied", "No hay significativo derrame articular", 0.03, size="none"),
        "synovitis": dict(NM),
        "bakers_cyst": f("denied", "No hay significativo derrame articular ni quistes poplíteos", 0.01, size=None, longest_mm=None),
        "contusion": f("unsettled", "Discreto edema óseo periférico del platillo tibial medial ... que podría representar una lesión avulsiva no desplazada", 0.6, ["cause"], cause=None, bones=["medial_tibial_plateau"]),
        "fracture": f("hedged", "podría representar una lesión avulsiva no desplazada", 0.4, acuity=None)}},
    {"index": 6, "study": {"report_side": "both", "report_type": "narrative", "refers_to_prior": False, "hoffa_edema": None}, "findings": {
        "acl": f("below", "Ο πρόσθιος χιαστός σύνδεσµος παρουσιάζει µυξοειδή εκφύλιση", 0.2, grade="degeneration", acuity=None, graft=False),
        "mcl": f("structure_normal", "Τα λοιπά συνδεσµικά στοιχεία είναι ακέραια", 0.05, grade=None, acuity=None),
        "medial_meniscus": f("below", "Στο έσω διαµέρισµα αναγνωρίζεται εκφύλιση και ήπιο υπεξάρθρηµα του έσω µηνίσκου", 0.2, **M0),
        "lateral_meniscus": f("meets", "παρατηρείται εκφυλιστική ρήξη και υπεξάρθρηµα του µηνίσκου", 0.9, **M0),
        "medial_oa": f("structure_normal", "Ο αρθρικός χόνδρος και ο οστικός µυελός έχουν φυσιολογική απεικόνιση", 0.02, **OA0),
        "lateral_oa": f("meets", "χόνδρινο έλλειµµα διαµέτρου 2,5mm ολικού βάθους ... λέπτυνση-διάβρωση του αρθρικού χόνδρου του έξω κνηµιαίου πλατώ (ICRS grade III)", 0.8, depth="full", size_mm=2.5, extent="multifocal", severity_word=None),
        "pf_oa": f("meets", "εκφύλιση ICRS grade III στον αρθρικό χόνδρο της επιγονατίδας κατά την έσω µοίρα", 0.8, depth="deep", size_mm=None, extent=None, severity_word=None),
        "effusion": f("meets", "Ενδοαρθρικά παρατηρείται µεγάλη συλλογή υγρού", 0.97, size="large"),
        "synovitis": dict(NM),
        "bakers_cyst": f("unsettled", "συνοβιακή κύστη Baker µε παχύρευστο κατά τόπους περιεχόµενο, διαφραγµάτια, κεφαλουραία διάµετρο 4cm", 0.75, ["size"], size=None, longest_mm=40),
        "contusion": f("excluded", "υπαρθρικό οστεοµυελικό οίδηµα ιδίως εµφανές στην επίφυση της κνήµης. Τα ευρήµατα συνιστούν εκφυλιστική οστεοαρθρίτιδα", 0.05, cause="degeneration", bones=["lateral_tibial_plateau"]),
        "fracture": f("excluded", "χόνδρινο έλλειµµα διαµέτρου 2,5mm ολικού βάθους στα πλαίσια χόνδρινου κατάγµατος", 0.05, acuity=None)}},
    {"index": 105, "study": {"report_side": "not_stated", "report_type": "narrative", "refers_to_prior": False, "hoffa_edema": None}, "findings": {
        "acl": f("structure_normal", "ACL se prati u kontinuitetu.", 0.02, grade=None, acuity=None, graft=False),
        "mcl": f("structure_normal", "Medijalni kolateralni kompleks je održan.", 0.02, grade=None, acuity=None),
        "medial_meniscus": f("denied", "Medijalni menisk je bez znakova rupture", 0.03, **M0),
        "lateral_meniscus": f("meets", "Lateralni menisk je insuficijentan, u cijelosti maceriran ... Radijalna ruptura stražnjeg korijena ... Horizontalna ruptura prednjeg roga lateralnog meniska i korpusa i kompleksna ruptura stražnjeg roga", 0.97, location=["anterior_horn", "body", "posterior_horn", "posterior_root"], displaced_fragment=None),
        "medial_oa": f("unsettled", "Degenerativne promjene u sklopu OA obostrano, izraženije lateralnog kompartmenta koljena", 0.6, ["grade"], **OA0),
        "lateral_oa": f("meets", "izraženije lateralnog kompartmenta koljena, sa uznapredovalom hondromalacijom i mjestimično potpunim gubitkom hrskavice", 0.97, depth="full", size_mm=None, extent="multifocal", severity_word="severe"),
        "pf_oa": f("meets", "Hondromalacija patele IV stupnja u području grebena ... Hrskavica je difuzno stanjena. Hondromalacija trohlearne hrskavice III-IV stupnja.", 0.97, depth="full", size_mm=None, extent="diffuse", severity_word=None),
        "effusion": f("meets", "Suprapateralno umjerena količina zglobnog izljeva", 0.9, size="moderate"),
        "synovitis": dict(NM),
        "bakers_cyst": f("meets", "Veća, septirana poplitealna cista, dimenzija oko 35x35x65 mm", 0.97, size="large", longest_mm=65),
        "contusion": f("unsettled", "Interkondilarno straga nešto koštanog edema u femuru.", 0.45, ["cause"], cause=None, bones=["femur_other"]),
        "fracture": dict(NM, acuity=None)}},
    {"index": 67, "study": {"report_side": "not_stated", "report_type": "structured", "refers_to_prior": False, "hoffa_edema": False}, "findings": {
        "acl": f("structure_normal", "Intact cruciate and collateral ligaments.", 0.05, grade=None, acuity=None, graft=False),
        "mcl": f("structure_normal", "Intact cruciate and collateral ligaments.", 0.05, grade=None, acuity=None),
        "medial_meniscus": f("structure_normal", "Intact menisci.", 0.05, **M0),
        "lateral_meniscus": f("structure_normal", "Intact menisci.", 0.05, **M0),
        "medial_oa": f("structure_normal", "Well-maintained tricompartmental hyaline articular cartilage.", 0.05, **OA0),
        "lateral_oa": f("structure_normal", "Well-maintained tricompartmental hyaline articular cartilage.", 0.05, **OA0),
        "pf_oa": f("structure_normal", "Well-maintained tricompartmental hyaline articular cartilage.", 0.05, **OA0),
        "effusion": f("denied", "No knee joint effusion.", 0.01, size="none"),
        "synovitis": f("general_normal", "No significant findings.", 0.08),
        "bakers_cyst": f("general_normal", "No significant findings.", 0.08, size=None, longest_mm=None),
        "contusion": f("general_normal", "No sinister bone marrow changes. ... No significant findings.", 0.08, cause=None, bones=[]),
        "fracture": f("general_normal", "No sinister bone marrow changes. ... No significant findings.", 0.08, acuity=None)}},
    {"index": 189, "study": {"report_side": "not_stated", "report_type": "narrative", "refers_to_prior": False, "hoffa_edema": None}, "findings": {
        "acl": f("meets", "Ön çapraz bağ femoral insersiyodan rüptüredir.", 0.97, grade="complete", acuity=None, graft=False),
        "mcl": f("structure_normal", "Arka çapraz bağ yan bağlar normaldir.", 0.05, grade=None, acuity=None),
        "medial_meniscus": f("meets", "Mediyal menisküste interkondiler çentiğe deplase kova sapı yırtığı vardır.", 0.97, location=[], displaced_fragment=True),
        "lateral_meniscus": f("meets", "Lateral menisküs korpus apeksinde milimetrik radiyal yırtık vardır.", 0.8, location=["body"], displaced_fragment=None),
        "medial_oa": f("structure_normal", "Mediyal ve lateral femorotibyal, eklem yüzleri normaldir.", 0.02, **OA0),
        "lateral_oa": f("structure_normal", "Mediyal ve lateral femorotibyal, eklem yüzleri normaldir.", 0.02, **OA0),
        "pf_oa": f("unsettled", "patellar taraf mediyal fasette kondral incelme fissür oluşumu vardır.Troklear tarafta ve interkondiler sulkusa yakın lokaliasyonda kondral kayıplar vardır.", 0.6, ["grade", "extent"], **OA0),
        "effusion": f("denied", "Diz eklemi içi sıvı miktarı normaldir.", 0.01, size="none"),
        "synovitis": dict(NM), "bakers_cyst": dict(NM, size=None, longest_mm=None),
        "contusion": dict(NM, cause=None, bones=[]), "fracture": dict(NM, acuity=None)}},
]
NOTES = {
    120: "ACL 'tear' without a grade meets; an ordinary tear is 0.90, lowered one step to 0.80 because every statement uses an indirect word ('in favor of', 'C/W'). MCL grade 1 is below, clearly below: 0.20. 'R/O meniscocapsular separation' is a hedge about the medial meniscus: 0.30. Effusion and Baker's cyst have no size: unsettled, 0.60. Synovitis 0.97: synovial hypertrophy is described and the impression states it directly. Contusion meets at 0.97 (femoral condyle and tibial plateau: a kissing pattern, stated directly in the impression) even though a fracture is suspected; the suspected fracture itself is hedged, and 'suspicious of' (0.40) is firmer than 'R/O' (0.30).",
    198: "The clinical question ('Sospecha lesión meniscal') is ignored. MCL 'grade I-II' with slight loss of continuity: unsettled (grade), leaning away from the definition: 0.45. Full-thickness 2 x 3 mm lesion is below (< 10 mm), one step under the threshold: 0.35. PF grade 4 lesions with no size meet only because the size is not stated: 0.80. Edema whose cause is only a possible avulsion: unsettled (cause), 0.60; the avulsion itself is hedged with 'could represent': 0.40. 'Ligamentos cruzados ... sin alteraciones' covers the ACL as part of a group: 0.05; 'Menisco lateral ... conservada' names the structure: 0.02.",
    6: "Two reports (right, then left); knee_side is RIGHT, so only the right-knee part is used. A cartilage-only (chondral) fracture is excluded for fracture. Edema in a knee the report calls degenerative OA is excluded for contusion (cause degeneration). Baker's cyst 4 cm with no size word: unsettled, and 35-49 mm leans toward the definition: 0.75. ICRS grade III with no size meets only because the size is not stated: 0.80. 'Large' effusion: 0.97.",
    105: "OA 'on both sides, more pronounced laterally' with advanced loss described for the lateral side: lateral meets (0.97, advanced), medial unsettled (grade, 0.60). Patellar grade IV with diffusely thinned cartilage: 0.97. Baker's cyst 65 mm: 0.97. Edema with no cause in a knee the report describes as degenerative: unsettled, leaning away: 0.45. A partial PCL tear is not one of the 12 findings and is ignored.",
    67: "Normal structured report. 'No sinister bone marrow changes' is a tumour screen and the conclusion 'No significant findings' is general: contusion, fracture, synovitis and Baker's cyst are general_normal at 0.08 (whole-knee normal conclusion). Group statements ('Intact menisci', 'tricompartmental cartilage') are structure_normal at 0.05. 'No knee joint effusion' is a categorical denial: 0.01. 'No evidence of fat pad impingement' makes hoffa_edema false.",
    189: "'Rüptür' (rupture) of the ACL meets with grade complete: 0.97. Bucket-handle tear displaced into the notch: displaced_fragment true, 0.97. A 'millimetric' radial tear is weakened by its degree word: 0.80. 'Kondral kayıplar' (cartilage loss) with no depth or size: unsettled (grade, extent), 0.60. Quadriceps (suprapatellar) fat pad edema is not Hoffa's fat pad: hoffa_edema null.",
}

reports = pd.read_csv(B / "reports.csv").set_index("index")
schema = json.loads((B / "schema.json").read_text())
val = jsonschema.Draft202012Validator(schema)
norm = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().casefold()
for e in EX:
    e["study_id"] = reports.at[e["index"], "study_id"]
    e = {"index": e["index"], "study_id": e["study_id"], "study": e["study"], "findings": e["findings"]}
    errs = [er.message for er in val.iter_errors(e)]
    rep = norm(reports.at[e["index"], "report"])
    bad_q = [k for k, v in e["findings"].items() for piece in re.split(r"\s*(?:\.\.\.|…)\s*", v["quote"]) if piece.strip() and norm(piece) not in rep]
    assert not errs and not bad_q, (e["index"], errs, bad_q)
print("all examples valid; quotes found in reports")


def compact(e):
    e = {"index": e["index"], "study_id": e["study_id"], "study": e["study"], "findings": e["findings"]}
    lines = ['{"index": %d, "study_id": "%s",' % (e["index"], e["study_id"]), ' "study": %s,' % json.dumps(e["study"], ensure_ascii=False), ' "findings": {']
    items = list(e["findings"].items())
    for i, (k, v) in enumerate(items):
        lines.append('  "%s": %s%s' % (k, json.dumps(v, ensure_ascii=False), "," if i < len(items) - 1 else ""))
    lines.append(" }}")
    return "\n".join(lines)


blocks = []
for n, e in enumerate(EX, 1):
    blocks.append(f'<example id="{n}">\n<input>reports.csv index {e["index"]}</input>\n<why>\n{NOTES[e["index"]]}\n</why>\n<output>\n{compact(e)}\n</output>\n</example>')
(Path(__file__).parent / "examples_block.txt").write_text("\n\n".join(blocks) + "\n")
print("examples_block.txt written", sum(len(b) for b in blocks), "chars")
