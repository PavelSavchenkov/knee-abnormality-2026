"""Repository paths and the 12 findings, shared by every experiment."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
META = DATA / "meta"
FRAMES = DATA / "frames"
TEXT_LABELS = DATA / "text_labels"
CACHE = DATA / "cache"

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
            "Baker's", "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa", "effusion", "synovitis",
        "bakers_cyst", "contusion", "fracture"]
KEY_OF = dict(zip(FINDINGS, KEYS))
FINDING_OF = dict(zip(KEYS, FINDINGS))
