"""First facts -> targets mapping for the v2 labels (data/text_labels/v2.json). WIP, not exported to data/.

Applies the official definitions to the facts with the v1 status ladder and v1-like probability values, so that v1 and v2
targets differ only where the two labellers read the reports differently. Per finding, every relevant fact gives a candidate
(status, probability); the highest status in v1's order wins (meets > hedged > unsettled > below > excluded > denied >
structure_normal > general_normal), the highest probability within it. Not mentioned -> NM (0.15, the main pipeline's default).

    from v2_targets import v2_targets; probs, status = v2_targets()"""
import json
import re
from pathlib import Path

import pandas as pd

V2 = Path(__file__).resolve().parents[3] / "data/text_labels/v2.json"
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa", "effusion", "synovitis",
        "bakers_cyst", "contusion", "fracture"]
ORDER = ["meets", "hedged", "unsettled", "below", "excluded", "denied", "structure_normal", "general_normal"]
NM = 0.15
HEDGE = {"probable": 0.5, "possible": 0.4, "questionable": 0.3}
FIRM = {"present", "present_indirect"}
SITES = {"medial_oa": {"medial_femoral_condyle", "medial_tibial_plateau", "medial_compartment"},
         "lateral_oa": {"lateral_femoral_condyle", "lateral_tibial_plateau", "lateral_compartment"},
         "pf_oa": {"patella", "trochlea", "patellofemoral"}}
SPREAD = {"whole_knee": {"medial_oa", "lateral_oa", "pf_oa"}, "site_unspecified": {"medial_oa", "lateral_oa", "pf_oa"},
          "femorotibial": {"medial_oa", "lateral_oa"}}
BIG = {"moderate_to_large", "large"}
CARTILAGE = {"chondromalacia", "cartilage_fissuring", "cartilage_thinning", "cartilage_defect", "full_thickness_loss",
             "osteochondral_lesion", "cartilage_signal_change"}


def max_grade(g):
    return max(int(c) for c in re.findall(r"[0-4]", g)) if g else None


def denial(f):
    return ("denied", 0.01 if f["assertion"] == "absent" else 0.03)


def normal(f):
    return ("structure_normal", 0.02 if f["scope"] == "specific" and f["structure"] not in SPREAD else 0.05)


def firm_or_hedged(f, status, p):
    """A qualifying statement: meets if firm (indirect one step lower), hedged by its hedge word otherwise."""
    if f["assertion"] in FIRM:
        return (status, p if f["assertion"] == "present" else max(0.8, p - 0.07))
    return ("hedged", HEDGE[f["assertion"]])


def acl(f, ctx):
    fd, a = f["finding"], f["assertion"]
    if a in ("absent", "absent_qualified"):
        return denial(f) if fd in ("tear", "partial_tear", "complete_tear", "sprain") else None
    if fd == "normal":
        return normal(f)
    sev, grade = f.get("severity_word"), max_grade(f.get("grade"))
    if fd in ("complete_tear", "not_visualized") or (fd == "partial_tear" and sev == "high_grade") or (fd == "sprain" and grade == 3):
        return firm_or_hedged(f, "meets", 0.97 if fd != "partial_tear" else 0.9)
    if fd == "tear":
        return firm_or_hedged(f, "meets", 0.9)
    if fd in ("partial_tear", "sprain") and a in FIRM:
        return ("below", 0.35) if sev == "low_grade" or grade in (0, 1) else ("unsettled", 0.6)
    if fd in ("partial_tear", "sprain"):
        return ("hedged", 0.3)
    if fd in ("degeneration", "signal_change", "laxity", "chronic_change", "periligamentous_edema"):
        return ("below", 0.2)
    if fd == "avulsion":
        return ("excluded", 0.05)


def mcl(f, ctx):
    fd, a = f["finding"], f["assertion"]
    if a in ("absent", "absent_qualified"):
        return denial(f) if fd in ("tear", "partial_tear", "complete_tear", "sprain") else None
    if fd == "normal":
        return normal(f)
    if f.get("acuity") == "chronic" or fd in ("chronic_change", "degeneration"):
        return ("excluded", 0.05)
    sev, grade = f.get("severity_word"), max_grade(f.get("grade"))
    if fd == "complete_tear" or (fd == "partial_tear" and sev == "high_grade") or (fd == "sprain" and grade == 3):
        return firm_or_hedged(f, "meets", 0.97 if fd == "complete_tear" else 0.9)
    if a not in FIRM:
        return ("hedged", 0.3)
    if fd in ("tear", "partial_tear", "sprain"):
        return ("below", 0.2) if sev == "low_grade" or grade == 1 else ("unsettled", 0.6)
    if fd == "periligamentous_edema":
        return ("unsettled", 0.75) if ctx["mcl_signal"] else ("below", 0.1)
    if fd == "signal_change":
        return ("unsettled", 0.75) if ctx["mcl_edema"] else ("excluded", 0.05)


def meniscus(f, ctx):
    fd, a = f["finding"], f["assertion"]
    if a in ("absent", "absent_qualified"):
        return denial(f) if fd in ("tear", "displaced_fragment", "surface_signal") else None
    if fd == "normal":
        return normal(f)
    if fd in ("displaced_fragment", "maceration") or "bucket_handle" in (f.get("tear_type") or []):
        return firm_or_hedged(f, "meets", 0.97)
    if fd in ("tear", "truncation", "resection"):
        return firm_or_hedged(f, "meets", 0.8 if f.get("severity_word") in ("trace", "small") else 0.9)
    if fd == "surface_signal":
        if f.get("images") == "single":
            return ("below", 0.35)
        return ("unsettled", 0.75) if max_grade(f.get("grade")) == 2 else firm_or_hedged(f, "meets", 0.9)
    if fd in ("capsular_separation", "discoid"):
        return ("unsettled", 0.6) if a in FIRM else ("hedged", HEDGE[a])
    if fd == "parameniscal_cyst":
        return ("unsettled", 0.75)
    if fd in ("degeneration", "fraying"):
        return ("below", 0.2)


def oa(f, ctx):
    """Cartilage / OA facts for one compartment; facts of whole_knee / femorotibial / site_unspecified are 'spread'."""
    fd, a, spread = f["finding"], f["assertion"], f["structure"] in SPREAD
    if a in ("absent", "absent_qualified"):
        return denial(f) if fd in CARTILAGE | {"osteoarthritis"} else None
    if fd == "cartilage_normal":
        return normal(f)
    if a not in FIRM:
        return ("hedged", HEDGE[a]) if fd in CARTILAGE | {"osteoarthritis"} else None
    sev, grade, size = f.get("severity_word"), max_grade(f.get("grade")), f.get("size_mm")
    deep = fd == "full_thickness_loss" or f.get("depth") in ("deep", "full") or (grade is not None and grade >= 3 and fd in CARTILAGE)
    if fd == "osteoarthritis":
        if sev in BIG:
            return ("unsettled", 0.6) if spread else ("meets", 0.97)
        return ("unsettled", 0.45 if sev == "small" else 0.6)
    if deep:
        if spread:
            return ("unsettled", 0.6)
        if size is not None and size < 10:
            return ("below", 0.35)
        return ("meets", 0.97 if (size or 0) >= 10 or f.get("extent") == "diffuse" else 0.8)
    if fd in CARTILAGE:
        superficial = f.get("depth") == "superficial" or (grade is not None and grade <= 2 and f.get("grade", "") not in ("2-3",))
        if superficial or fd == "cartilage_signal_change" or (fd in ("cartilage_thinning", "cartilage_fissuring") and sev in ("trace", "small")):
            return ("below", 0.2)
        return ("unsettled", 0.75 if sev in BIG or f.get("grade") == "2-3" else 0.6)
    if fd in ("osteophytes", "joint_space_narrowing"):
        return ("unsettled", 0.6)


def effusion(f, ctx):
    fd, a = f["finding"], f["assertion"]
    if a in ("absent", "absent_qualified"):
        return denial(f) if fd in ("effusion", "hemarthrosis") else None
    if a not in FIRM:
        return ("hedged", HEDGE[a])
    sev = f.get("severity_word")
    if sev in BIG or sev == "moderate":
        return ("meets", 0.97 if sev in BIG else 0.9)
    if sev == "small_to_moderate":
        return ("unsettled", 0.75)
    if sev in ("small", "trace"):
        return ("below", 0.35 if sev == "small" else 0.2)
    return ("unsettled", 0.75 if fd in ("hemarthrosis", "lipohemarthrosis") else 0.6)


def synovitis(f, ctx):
    fd, a = f["finding"], f["assertion"]
    if fd == "normal":
        return normal(f)
    if fd not in ("synovitis", "synovial_thickening", "synovial_proliferation"):
        return None
    if a in ("absent", "absent_qualified"):
        return denial(f)
    sev = f.get("severity_word")
    return firm_or_hedged(f, "meets", 0.97 if sev in BIG else 0.8 if sev in ("trace", "small") else 0.9)


def bakers(f, ctx):
    fd, a = f["finding"], f["assertion"]
    if fd == "normal":
        return normal(f)
    if a in ("absent", "absent_qualified"):
        return denial(f) if fd in ("bakers_cyst", "ruptured_bakers_cyst") else None
    if fd == "bursal_fluid":
        return ("below", 0.2)
    if a not in FIRM:
        return ("hedged", HEDGE[a])
    sev, size = f.get("severity_word"), f.get("size_mm")
    if sev in BIG or sev == "moderate" or (sev is None and (size or 0) >= 50):
        return ("meets", 0.97 if sev in BIG or (size or 0) >= 50 else 0.9)
    if sev in ("small", "trace"):
        return ("below", 0.35 if sev == "small" else 0.2)
    if size is not None:
        return ("unsettled", 0.75) if size >= 35 else ("unsettled", 0.45) if size >= 20 else ("below", 0.35)
    return ("unsettled", 0.6)


def contusion(f, ctx):
    fd, a = f["finding"], f["assertion"]
    if fd == "bone_normal":
        return normal(f)
    if a in ("absent", "absent_qualified"):
        return denial(f) if fd in ("contusion", "marrow_edema") else None
    if fd in ("contusion", "impaction_injury") or (fd == "marrow_edema" and f.get("cause") == "trauma"):
        return firm_or_hedged(f, "meets", 0.8 if fd == "impaction_injury" else 0.9)
    if fd == "marrow_edema":
        if f.get("cause"):
            return ("excluded", 0.05)
        if a not in FIRM:
            return ("hedged", HEDGE[a])
        return ("unsettled", 0.75 if ctx["acute"] else 0.45 if ctx["degenerative"] else 0.6)


def fracture(f, ctx):
    fd, a = f["finding"], f["assertion"]
    if fd == "bone_normal":
        return normal(f)
    if a in ("absent", "absent_qualified"):
        return denial(f) if fd in ("fracture", "avulsion_fracture", "osteochondral_fracture") else None
    if fd in ("fracture", "avulsion_fracture", "osteochondral_fracture"):
        if f.get("acuity") == "chronic":
            return ("excluded", 0.05)
        return firm_or_hedged(f, "meets", 0.97 if f.get("displaced") else 0.9)
    if fd in ("insufficiency_fracture", "impaction_injury", "osteochondral_lesion"):
        return ("unsettled", 0.6) if a in FIRM else ("hedged", 0.3)
    if fd == "ocd":
        return ("excluded", 0.05)


RULES = {"acl": ({"acl"}, acl), "mcl": ({"mcl"}, mcl), "medial_meniscus": ({"medial_meniscus"}, meniscus),
         "lateral_meniscus": ({"lateral_meniscus"}, meniscus), "effusion": ({"joint"}, effusion), "synovitis": ({"synovium"}, synovitis),
         "bakers_cyst": ({"popliteal_fossa"}, bakers)}


def study_targets(x):
    facts = x["facts"]
    ctx = {"acute": x["context"]["trauma"] == "recent" or any(f["finding"] in ("contusion", "complete_tear") or
                                                              (f["structure"] in ("acl", "mcl") and "tear" in f["finding"] and f["assertion"] in FIRM)
                                                              for f in facts),
           "degenerative": any(f["finding"] in ("osteoarthritis", "full_thickness_loss", "osteophytes") and f["assertion"] in FIRM for f in facts),
           "mcl_signal": any(f["structure"] == "mcl" and f["finding"] == "signal_change" for f in facts),
           "mcl_edema": any(f["structure"] == "mcl" and f["finding"] == "periligamentous_edema" for f in facts)}
    out = {}
    for k in KEYS:
        cands = []
        for f in facts:
            if k in SITES:
                if f["structure"] in SITES[k] or k in SPREAD.get(f["structure"], ()):
                    cands.append(oa(f, ctx))
            elif k in ("contusion", "fracture"):
                cands.append((contusion if k == "contusion" else fracture)(f, ctx))
            elif f["structure"] in RULES[k][0] or (k.endswith("meniscus") and f["structure"] == "meniscus_unspecified"):
                c = RULES[k][1](f, ctx)
                cands.append(("unsettled", 0.6) if c and c[0] == "meets" and f["structure"] == "meniscus_unspecified" else c)
            if f["structure"] == "global":
                cands.append(("general_normal", 0.08 if f["finding"] == "normal_study" else 0.15))
        cands = [c for c in cands if c]
        if not cands:
            out[k] = ("not_mentioned", NM)
            continue
        best = min(ORDER.index(s) for s, _ in cands)
        out[k] = (ORDER[best], max(p for s, p in cands if ORDER.index(s) == best))
    return out


def v2_targets(path=V2):
    t = json.load(open(path))
    rows = {s: study_targets(x) for s, x in t.items()}
    probs = pd.DataFrame({s: {k: v[1] for k, v in r.items()} for s, r in rows.items()}).T[KEYS]
    status = pd.DataFrame({s: {k: v[0] for k, v in r.items()} for s, r in rows.items()}).T[KEYS]
    return probs, status
