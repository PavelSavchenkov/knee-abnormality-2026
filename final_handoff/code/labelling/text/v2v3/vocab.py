"""Controlled vocabularies of the v2 (fact extraction) and v3 (image-read forecast) text-labelling bundles.
build.py turns them into each bundle's schema.json and checks that every value is explained in the bundle's prompt."""

FINDINGS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa", "effusion", "synovitis",
            "bakers_cyst", "contusion", "fracture"]

# ---------------------------------------------------------------- v2: facts
V2_FAMILIES = {   # family: (structures, findings, optional attributes)
    "ligament": (["acl", "pcl", "mcl", "lcl", "posterolateral_corner", "mpfl"],
                 ["normal", "signal_change", "degeneration", "sprain", "partial_tear", "complete_tear", "tear", "not_visualized",
                  "periligamentous_edema", "laxity", "avulsion", "chronic_change", "graft"],
                 ["part_ligament", "grade", "grade_system", "percent", "severity_word", "acuity", "graft", "change"]),
    "meniscus": (["medial_meniscus", "lateral_meniscus", "meniscus_unspecified"],
                 ["normal", "degeneration", "surface_signal", "tear", "fraying", "maceration", "truncation", "resection", "repair",
                  "extrusion", "capsular_separation", "discoid", "parameniscal_cyst", "displaced_fragment"],
                 ["part_meniscus", "tear_type", "images", "grade", "grade_system", "size_mm", "severity_word", "acuity", "change"]),
    "site": (["medial_femoral_condyle", "medial_tibial_plateau", "lateral_femoral_condyle", "lateral_tibial_plateau", "patella",
              "trochlea", "femur_other", "tibia_other", "fibula", "medial_compartment", "lateral_compartment", "patellofemoral",
              "femorotibial", "whole_knee", "site_unspecified"],
             ["cartilage_normal", "cartilage_signal_change", "chondromalacia", "cartilage_fissuring", "cartilage_thinning",
              "cartilage_defect", "full_thickness_loss", "osteochondral_lesion",
              "osteoarthritis", "osteophytes", "joint_space_narrowing", "subchondral_sclerosis", "subchondral_cyst",
              "bone_normal", "marrow_edema", "contusion", "fracture", "avulsion_fracture", "osteochondral_fracture",
              "impaction_injury", "insufficiency_fracture", "osteonecrosis", "ocd", "bone_lesion", "red_marrow", "postoperative_bone"],
             ["depth", "percent", "grade", "grade_system", "size_mm", "extent", "severity_word", "acuity", "cause", "displaced", "change"]),
    "joint": (["joint"], ["effusion", "hemarthrosis", "lipohemarthrosis", "loose_body", "suprapatellar_bursitis"],
              ["size_mm", "severity_word", "change"]),
    "synovium": (["synovium"], ["normal", "synovitis", "synovial_thickening", "synovial_proliferation", "plica"],
                 ["extent", "severity_word", "change"]),
    "popliteal": (["popliteal_fossa"], ["normal", "bakers_cyst", "ruptured_bakers_cyst", "bursal_fluid"],
                  ["size_mm", "severity_word", "change"]),
    "bursa": (["other_bursa"], ["bursitis", "ganglion_cyst", "cyst"], ["size_mm", "severity_word", "change"]),
    "fat_pad": (["hoffa_fat_pad"], ["normal", "edema", "impingement"], ["severity_word", "change"]),
    "tendon": (["quadriceps_tendon", "patellar_tendon"], ["normal", "tendinosis", "partial_tear", "complete_tear", "tear"],
               ["severity_word", "acuity", "change"]),
    "soft_tissue": (["soft_tissue"], ["normal", "edema"], ["severity_word", "change"]),
    "global": (["global"], ["normal_study", "otherwise_normal"], []),
}
V2_ATTRIBUTES = {   # name in V2_FAMILIES: (json key, schema)
    "part_ligament": ("part", {"type": "array", "uniqueItems": True, "minItems": 1, "items": {"enum": ["proximal", "mid", "distal"]}}),
    "part_meniscus": ("part", {"type": "array", "uniqueItems": True, "minItems": 1,
                               "items": {"enum": ["anterior_horn", "body", "posterior_horn", "anterior_root", "posterior_root"]}}),
    "tear_type": ("tear_type", {"type": "array", "uniqueItems": True, "minItems": 1,
                                "items": {"enum": ["horizontal", "longitudinal", "radial", "oblique", "complex", "flap", "bucket_handle",
                                                   "root", "ramp", "degenerative", "peripheral"]}}),
    "images": ("images", {"enum": ["single", "multiple"]}),
    "grade": ("grade", {"type": "string", "pattern": "^[0-4][abc]?(-[0-4][abc]?)?$"}),
    "grade_system": ("grade_system", {"enum": ["icrs", "outerbridge", "noyes", "kellgren_lawrence", "meniscal", "ligament", "unspecified"]}),
    "percent": ("percent", {"type": "number", "minimum": 0, "maximum": 100}),
    "depth": ("depth", {"enum": ["superficial", "partial", "deep", "full"]}),
    "size_mm": ("size_mm", {"type": "number", "exclusiveMinimum": 0}),
    "extent": ("extent", {"enum": ["focal", "multifocal", "diffuse"]}),
    "severity_word": ("severity_word", {"enum": ["trace", "small", "small_to_moderate", "moderate", "moderate_to_large", "large",
                                                 "low_grade", "high_grade"]}),
    "acuity": ("acuity", {"enum": ["acute", "subacute", "chronic"]}),
    "cause": ("cause", {"enum": ["trauma", "degeneration", "fracture", "insufficiency", "osteonecrosis", "ocd", "postoperative",
                                 "inflammatory", "tumor", "infection", "other"]}),
    "displaced": ("displaced", {"type": "boolean"}),
    "graft": ("graft", {"type": "boolean"}),
    "change": ("change", {"enum": ["new", "unchanged", "progressed", "improved", "resolved"]}),
}
V2_ASSERTIONS = ["present", "present_indirect", "probable", "possible", "questionable", "absent", "absent_qualified"]
V2_SECTIONS = ["findings", "impression", "both"]
V2_SURGERY = ["acl_reconstruction", "meniscectomy", "meniscal_repair", "cartilage_repair", "osteotomy", "arthroscopy", "other"]


def v2_schema():
    fact_props = {"structure": {"type": "string"}, "finding": {"type": "string"}, "assertion": {"enum": V2_ASSERTIONS},
                  "scope": {"enum": ["specific", "group"]}, "section": {"enum": V2_SECTIONS},
                  "quote": {"type": "string", "minLength": 1, "maxLength": 300}}
    core = list(fact_props)
    families = []
    for name, (structures, findings, attrs) in V2_FAMILIES.items():
        props = {k: fact_props[k] for k in core} | {"structure": {"enum": structures}, "finding": {"enum": findings}}
        props |= {V2_ATTRIBUTES[a][0]: V2_ATTRIBUTES[a][1] for a in attrs}
        families.append({"if": {"properties": {"structure": {"enum": structures}}},
                         "then": {"properties": props, "additionalProperties": False}})
    all_structures = [s for st, _, _ in V2_FAMILIES.values() for s in st]
    fact = {"type": "object", "required": ["structure", "finding", "assertion", "scope", "section", "quote"],
            "properties": fact_props | {"structure": {"enum": all_structures}}, "allOf": families}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Knee MRI report facts, one study (text labelling v2)",
        "type": "object", "additionalProperties": False, "required": ["index", "study_id", "report", "context", "facts"],
        "properties": {
            "index": {"type": "integer", "minimum": 0},
            "study_id": {"type": "string", "minLength": 1},
            "report": {"type": "object", "additionalProperties": False, "required": ["side", "style", "compares_prior"],
                       "properties": {"side": {"enum": ["same", "not_stated", "both", "other"]},
                                      "style": {"enum": ["structured", "narrative", "impression_only", "empty"]},
                                      "compares_prior": {"type": "boolean"}}},
            "context": {"type": "object", "additionalProperties": False, "required": ["trauma", "prior_surgery", "age_years", "quote"],
                        "properties": {"trauma": {"enum": ["recent", "remote", "none", "not_stated"]},
                                       "prior_surgery": {"type": "array", "uniqueItems": True, "items": {"enum": V2_SURGERY}},
                                       "age_years": {"type": ["integer", "null"], "minimum": 0},
                                       "quote": {"type": "string", "maxLength": 300}}},
            "facts": {"type": "array", "items": {"$ref": "#/$defs/fact"}},
        },
        "$defs": {"fact": fact},
    }


# ---------------------------------------------------------------- v3: forecast
V3_CLASSES = {   # evidence class: allowed p range (percent, inclusive)
    "qualifying": (80, 98), "likely_qualifying": (50, 85), "uncertain": (25, 65), "subthreshold": (8, 45),
    "excluded": (3, 25), "silent": (2, 45), "covered_normal": (2, 20), "negative": (1, 10),
}
V3_CLUES = [
    "acute_injury", "degenerative_knee", "postoperative_knee", "older_patient", "young_patient", "systematic_report", "brief_report",
    "acl_tear", "mcl_injury", "other_ligament_injury", "patellar_dislocation_signs", "pivot_shift_pattern", "contusion_present",
    "fracture_present", "marrow_edema_unattributed", "hemarthrosis", "lipohemarthrosis", "effusion_moderate_plus", "effusion_present",
    "no_effusion", "synovitis_reported", "synovial_thickening", "hoffa_edema", "loose_bodies", "advanced_oa", "oa_features",
    "subchondral_edema", "meniscal_tear_present", "meniscal_extrusion", "parameniscal_cyst", "surface_signal", "small_deep_lesion",
    "popliteal_fluid", "inflammatory_arthritis", "hedged_wording", "size_not_stated", "chronic_or_old", "prior_unchanged",
]


def v3_schema():
    finding = {"type": "object", "additionalProperties": False, "required": ["evidence", "p", "clues", "quote"],
               "properties": {"evidence": {"enum": list(V3_CLASSES)}, "p": {"type": "integer"},
                              "clues": {"type": "array", "uniqueItems": True, "maxItems": 6, "items": {"enum": V3_CLUES}},
                              "quote": {"type": "string", "maxLength": 300}},
               "allOf": [{"if": {"properties": {"evidence": {"const": c}}},
                          "then": {"properties": {"p": {"type": "integer", "minimum": lo, "maximum": hi},
                                                  "quote": {"const": ""} if c == "silent" else {"type": "string", "minLength": 1}}}}
                         for c, (lo, hi) in V3_CLASSES.items()]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Image-read forecast from a knee MRI report, one study (text labelling v3)",
        "type": "object", "additionalProperties": False, "required": ["index", "study_id", "report", "findings"],
        "properties": {
            "index": {"type": "integer", "minimum": 0},
            "study_id": {"type": "string", "minLength": 1},
            "report": {"type": "object", "additionalProperties": False, "required": ["side", "coverage", "setting"],
                       "properties": {"side": {"enum": ["same", "not_stated", "both", "other"]},
                                      "coverage": {"enum": ["systematic", "partial", "minimal", "empty"]},
                                      "setting": {"enum": ["acute_injury", "degenerative", "mixed", "postoperative", "unclear"]}}},
            "findings": {"type": "object", "additionalProperties": False, "required": FINDINGS,
                         "properties": {f: {"$ref": "#/$defs/finding"} for f in FINDINGS}},
        },
        "$defs": {"finding": finding},
    }
