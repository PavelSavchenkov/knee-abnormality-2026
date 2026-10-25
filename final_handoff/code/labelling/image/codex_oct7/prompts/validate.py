"""Check image-label files against schema.json, studies.csv and tiles.csv. Pure Python, no extra packages.

Usage (schema.json, studies.csv and tiles.csv must sit next to this file):
  python validate.py DIR FROM TO    every index FROM..TO (inclusive) must have DIR/<study_id>.json, each must pass
  python validate.py FILE.json      check one file

Checks: the JSON schema (schema.json, or schema_fluid.json for files with "panel": "fluid"); index and study_id match studies.csv; the file is named <study_id>.json; every sheet of the study is in
sheets_viewed; every tile id used belongs to the study; boxes have x0 < x1 and y0 < y1; the consistency rules of prompt.txt.
Prints the problems and a summary; exit code 1 if any file fails."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
PANELS = {"fluid": (json.loads((HERE / "schema_fluid.json").read_text(encoding="utf-8")),
                    {"1_sag_fs", "3_ax_fs", "5_notch", "6_medial", "9_patella"})}       # panel: (schema, sheets to look at)
with open(HERE / "studies.csv", encoding="utf-8", newline="") as fh:
    ROWS = {int(r["index"]): r for r in csv.DictReader(fh)}
TILES = defaultdict(set)
with open(HERE / "tiles.csv", encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        TILES[int(r["index"])].add(r["tile"])
TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def is_type(x, t):
    if t == "integer":
        return isinstance(x, int) and not isinstance(x, bool)
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    return isinstance(x, TYPES[t])


def show(x):
    return json.dumps(x, ensure_ascii=False)[:200]


def errors(x, s, path="$"):
    """Errors of instance x against schema s (the JSON Schema subset that schema.json uses)."""
    if "$ref" in s:
        node = SCHEMA
        for part in s["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return errors(x, node, path)
    if "type" in s:
        ts = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(is_type(x, t) for t in ts):
            return [f"{path}: expected {' or '.join(ts)}, got {show(x)}"]
    out = []
    if "enum" in s and not any(x == e and type(x) is type(e) for e in s["enum"]):
        if not (is_type(x, "number") and any(is_type(e, "number") and x == e for e in s["enum"])):
            out.append(f"{path}: {show(x)} is not one of {show(s['enum'])}")
    for sub in s.get("allOf", []):
        out += errors(x, sub, path)
    if isinstance(x, dict):
        out += [f"{path}: missing key '{k}'" for k in s.get("required", []) if k not in x]
        props = s.get("properties", {})
        for k, v in x.items():
            if k in props:
                out += errors(v, props[k], f"{path}.{k}")
            elif s.get("additionalProperties") is False:
                out.append(f"{path}: unexpected key '{k}'")
    if isinstance(x, list):
        if "items" in s:
            for i, v in enumerate(x):
                out += errors(v, s["items"], f"{path}[{i}]")
        if s.get("uniqueItems") and len({json.dumps(v) for v in x}) != len(x):
            out.append(f"{path}: repeated items")
        if len(x) < s.get("minItems", 0):
            out.append(f"{path}: needs at least {s['minItems']} item(s)")
        if len(x) > s.get("maxItems", len(x)):
            out.append(f"{path}: allows at most {s['maxItems']} item(s)")
    if isinstance(x, str):
        if len(x) < s.get("minLength", 0):
            out.append(f"{path}: must not be empty")
        if len(x) > s.get("maxLength", len(x)):
            out.append(f"{path}: longer than {s['maxLength']} characters")
    if is_type(x, "number"):
        if x < s.get("minimum", x):
            out.append(f"{path}: must be >= {s['minimum']}")
        if x > s.get("maximum", x):
            out.append(f"{path}: must be <= {s['maximum']}")
    return out


PRESENT = {"meets", "borderline", "possible", "below", "excluded"}
EXCLUDED_ALLOWED = {"mcl", "bakers_cyst", "contusion", "fracture"}
HIGH_GRADE = {"partial_ge_50", "full_thickness"}
DISRUPTED = {"partly_disrupted", "discontinuous"}


def allowed(name, status, ok, why):
    if status not in ok:
        return [f"findings.{name}: status {status} is not allowed when {why} (allowed: {', '.join(sorted(ok))})"]
    return []


def consistency(f):
    """Rules of prompt.txt; a panel file holds only some findings, so each rule runs only when its findings are present."""
    out = []
    for name, x in f.items():
        s = x["status"]
        if (x["certainty"] is None) != (s == "not_assessable"):
            out.append(f"findings.{name}: certainty must be null exactly when status is not_assessable")
        if (not x["evidence"]) != (s == "not_assessable"):
            out.append(f"findings.{name}: evidence must be empty exactly when status is not_assessable")
        if s in PRESENT and not x["boxes"]:
            out.append(f"findings.{name}: boxes must not be empty when status is {s}")
        if s not in PRESENT and x["boxes"]:
            out.append(f"findings.{name}: boxes must be empty when status is {s}")
        if s == "excluded" and name not in EXCLUDED_ALLOWED:
            out.append(f"findings.{name}: status excluded is only for {', '.join(sorted(EXCLUDED_ALLOWED))}")
    if "effusion" in f:
        out += effusion_rules(f["effusion"])
    if "bakers_cyst" in f:
        out += bakers_rules(f["bakers_cyst"])
    if "synovitis" in f:
        out += synovitis_rules(f["synovitis"])
    if "acl" in f:
        out += full_rules(f)
    return out


def effusion_rules(e):
    out = []
    effusion_ok = {0: {"normal"}, 1: {"below", "borderline"}, 2: {"meets", "borderline"}, 3: {"meets"}}
    if e["grade"] is not None:
        out += allowed("effusion", e["status"], effusion_ok[e["grade"]], f"grade is {e['grade']}")
    return out


def bakers_rules(b):
    out = []
    bakers_ok = {"none": {"normal", "excluded"}, "small": {"below", "borderline"}, "moderate": {"meets", "borderline"}, "large": {"meets"}}
    if b["grade"] is not None:
        out += allowed("bakers_cyst", b["status"], bakers_ok[b["grade"]], f"grade is {b['grade']}")
    return out


def synovitis_rules(s):
    out = []
    if s["status"] == "meets" and not (s["synovial_thickening"] is True or (s["hoffa_signal"] or 0) >= 2):
        out.append("findings.synovitis: meets needs synovial_thickening true or hoffa_signal 2 or 3")
    if (s["effusion_synovitis"] or 0) >= 2:
        out += allowed("synovitis", s["status"], {"meets", "borderline", "possible"}, f"effusion_synovitis is {s['effusion_synovitis']}")
    return out


def full_rules(f):
    out = []
    for name in ["medial_oa", "lateral_oa", "pf_oa"]:
        o = f[name]
        if o["cartilage_loss"] == "none":
            out += allowed(name, o["status"], {"normal"}, "cartilage_loss is none")
        if o["cartilage_loss"] == "partial_lt_50":
            out += allowed(name, o["status"], {"below", "borderline", "possible"}, "cartilage_loss is partial_lt_50")
        if o["cartilage_loss"] in HIGH_GRADE and o["extent_ge_10mm"] is True:
            out += allowed(name, o["status"], {"meets", "borderline"}, "high-grade loss extends 10 mm or more")
        if o["status"] == "meets" and o["cartilage_loss"] not in HIGH_GRADE:
            out.append(f"findings.{name}: meets needs cartilage_loss partial_ge_50 or full_thickness")
    a = f["acl"]
    if a["fibres"] == "continuous":
        out += allowed("acl", a["status"], {"normal"}, "fibres are continuous")
    if a["fibres"] == "discontinuous":
        out += allowed("acl", a["status"], {"meets"}, "fibres are discontinuous")
    if a["fibres"] == "not_seen":
        out += allowed("acl", a["status"], {"possible", "not_assessable"}, "fibres are not_seen")
    if a["status"] == "meets" and a["fibres"] not in DISRUPTED:
        out.append("findings.acl: meets needs fibres partly_disrupted or discontinuous")
    m = f["mcl"]
    if m["status"] == "meets" and not (m["fibres"] in DISRUPTED | {"thickened_or_bright"} and m["adjacent_edema"] is True):
        out.append("findings.mcl: meets needs adjacent_edema true and fibres thickened_or_bright, partly_disrupted or discontinuous")
    for name in ["medial_meniscus", "lateral_meniscus"]:
        x = f[name]
        if x["status"] == "meets" and not ((x["surface_tiles"] or 0) >= 2 or x["morphology"] in ("truncated_or_diminutive", "displaced_fragment")):
            out.append(f"findings.{name}: meets needs surface_tiles 2 or more, or morphology truncated_or_diminutive or displaced_fragment")
    c = f["contusion"]
    if all(v == 0 for v in c["edema"].values()):
        out += allowed("contusion", c["status"], {"normal"}, "every edema region is 0")
    if c["status"] == "meets" and not any((v or 0) > 0 for v in c["edema"].values()):
        out.append("findings.contusion: meets needs at least one edema region above 0")
    if c["status"] == "excluded" and not set(c["pattern"]) & {"around_fracture", "degenerative_subchondral", "insufficiency_or_sonk"}:
        out.append("findings.contusion: excluded needs pattern around_fracture, degenerative_subchondral or insufficiency_or_sonk")
    fr = f["fracture"]
    if fr["status"] == "meets" and not (fr["line_seen"] is True and fr["type"] not in (None, "old_or_healed")):
        out.append("findings.fracture: meets needs line_seen true and a type other than old_or_healed")
    if fr["type"] == "old_or_healed":
        out += allowed("fracture", fr["status"], {"excluded"}, "type is old_or_healed")
    return out


def tile_refs(x):
    """(path, tile id) of every tile reference in a label object."""
    refs = [(f"qa.marks[{i}]", m["tile"]) for i, m in enumerate(x["qa"]["marks"])]
    refs += [(f"structures[{i}]", s["tile"]) for i, s in enumerate(x.get("structures", []))]
    for name, f in x["findings"].items():
        refs += [(f"findings.{name}.evidence", t) for t in f["evidence"]]
        refs += [(f"findings.{name}.boxes[{i}]", b["tile"]) for i, b in enumerate(f["boxes"])]
    return refs


def boxes(x):
    out = [(f"structures[{i}]", s["box"]) for i, s in enumerate(x.get("structures", []))]
    for name, f in x["findings"].items():
        out += [(f"findings.{name}.boxes[{i}]", b["box"]) for i, b in enumerate(f["boxes"])]
    return out


def check_file(path, index=None):
    path = Path(path)
    try:
        x = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:                                     # report broken JSON instead of crashing
        return [f"not valid JSON: {e}"]
    panel = x.get("panel") if isinstance(x, dict) else None
    if panel is not None and panel not in PANELS:
        return [f"panel {panel!r} is not one of {sorted(PANELS)}"]
    schema, panel_sheets = PANELS.get(panel, (SCHEMA, None))
    out = errors(x, schema)
    if out:
        return out
    if x["index"] not in ROWS:
        return [f"index {x['index']} is not in studies.csv"]
    row = ROWS[x["index"]]
    if index is not None and x["index"] != index:
        out.append(f"index is {x['index']}, expected {index}")
    if x["study_id"] != row["study_id"]:
        out.append(f"study_id does not match studies.csv row {x['index']} ({row['study_id']})")
    if path.name != f"{x['study_id']}.json":
        out.append("file name must be <study_id>.json")
    wanted = set(row["sheets"].split(";")) & (panel_sheets or set(row["sheets"].split(";")))
    missing = wanted - set(x["sheets_viewed"])
    if missing:
        out.append(f"sheets_viewed lacks {sorted(missing)}: every {'fluid-panel ' if panel else ''}sheet of the study must be looked at")
    extra = set(x["sheets_viewed"]) - set(row["sheets"].split(";"))
    if extra:
        out.append(f"sheets_viewed lists sheets the study does not have: {sorted(extra)}")
    tiles = TILES[x["index"]]
    out += [f"{p}: tile {t} is not a tile of this study" for p, t in tile_refs(x) if t not in tiles]
    out += [f"{p}: box needs x0 < x1 and y0 < y1" for p, b in boxes(x) if not (b[0] < b[2] and b[1] < b[3])]
    out += consistency(x["findings"])
    return out


def main(args):
    if len(args) == 1:
        problems = {args[0]: check_file(args[0])}
    elif len(args) == 3:
        folder, lo, hi = Path(args[0]), int(args[1]), int(args[2])
        if not (0 <= lo <= hi and hi in ROWS):
            sys.exit(f"range {lo}-{hi} is outside studies.csv (0-{max(ROWS)})")
        wanted = {f"{ROWS[i]['study_id']}.json": i for i in range(lo, hi + 1)}
        problems = {name: check_file(folder / name, i) if (folder / name).exists() else ["file missing"] for name, i in wanted.items()}
        problems.update({p.name: ["not in the requested range"] for p in folder.glob("*.json") if p.name not in wanted})
    else:
        sys.exit(__doc__)
    bad = {k: v for k, v in problems.items() if v}
    for name, msgs in bad.items():
        print(name)
        for m in msgs:
            print("  -", m)
    print(f"{len(problems) - len(bad)} file(s) OK, {len(bad)} with problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
