"""Check fact files against schema.json and reports.csv. Pure Python, no extra packages.

Usage (schema.json and reports.csv must sit next to this file):
  python validate.py DIR FROM TO    every index FROM..TO (inclusive) must have DIR/<study_id>.json, each must pass
  python validate.py FILE.json      check one file

Checks: the JSON schema (including which findings and attributes each structure family allows); index and study_id match
reports.csv; the file is named <study_id>.json; every quote piece (pieces are split on '...') appears in the report, ignoring
case and whitespace; a study whose report has any text has at least one fact. Prints the problems and a summary; exit code 1
if any file fails."""
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
csv.field_size_limit(10**8)
with open(HERE / "reports.csv", encoding="utf-8", newline="") as fh:
    ROWS = {int(r["index"]): r for r in csv.DictReader(fh)}
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
    if "const" in s and not (x == s["const"] and type(x) is type(s["const"])):
        out.append(f"{path}: must be {show(s['const'])}")
    if "enum" in s and not any(x == e and type(x) is type(e) for e in s["enum"]):
        out.append(f"{path}: {show(x)} is not one of {show(s['enum'])}")
    for sub in s.get("allOf", []):
        out += errors(x, sub, path)
    if "if" in s and not errors(x, s["if"], path):
        out += errors(x, s["then"], path)
    if isinstance(x, dict):
        out += [f"{path}: missing key '{k}'" for k in s.get("required", []) if k not in x]
        props = s.get("properties", {})
        for k, v in x.items():
            if k in props:
                out += errors(v, props[k], f"{path}.{k}")
            elif s.get("additionalProperties") is False:
                out.append(f"{path}: key '{k}' is not allowed here")
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
        if "pattern" in s and not re.search(s["pattern"], x):
            out.append(f"{path}: {show(x)} does not match {s['pattern']}")
    if is_type(x, "number"):
        if x < s.get("minimum", x):
            out.append(f"{path}: must be >= {s['minimum']}")
        if x > s.get("maximum", x):
            out.append(f"{path}: must be <= {s['maximum']}")
        if "exclusiveMinimum" in s and x <= s["exclusiveMinimum"]:
            out.append(f"{path}: must be > {s['exclusiveMinimum']}")
    return out


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def quote_errors(quote, report, path):
    return [f"{path}: not found in the report: {show(piece)}" for piece in re.split(r"\s*(?:\.\.\.|…)\s*", quote)
            if piece.strip() and norm(piece) not in report]


def check_file(path, index=None):
    path = Path(path)
    try:
        x = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:                                     # report broken JSON instead of crashing
        return [f"not valid JSON: {e}"]
    out = errors(x, SCHEMA)
    if out:
        return out
    if x["index"] not in ROWS:
        return [f"index {x['index']} is not in reports.csv"]
    row = ROWS[x["index"]]
    if index is not None and x["index"] != index:
        out.append(f"index is {x['index']}, expected {index}")
    if x["study_id"] != row["study_id"]:
        out.append(f"study_id does not match reports.csv row {x['index']} ({row['study_id']})")
    if path.name != f"{x['study_id']}.json":
        out.append("file name must be <study_id>.json")
    report = norm(row["report"])
    out += quote_errors(x["context"]["quote"], report, "$.context.quote")
    for i, f in enumerate(x["facts"]):
        out += quote_errors(f["quote"], report, f"$.facts[{i}].quote")
    if not x["facts"] and x["report"]["style"] != "empty":
        out.append("facts is empty: every report with findings or an impression has at least one fact")
    return out


def main(args):
    if len(args) == 1:
        problems = {args[0]: check_file(args[0])}
    elif len(args) == 3:
        folder, lo, hi = Path(args[0]), int(args[1]), int(args[2])
        if not (0 <= lo <= hi and hi in ROWS):
            sys.exit(f"range {lo}-{hi} is outside reports.csv (0-{max(ROWS)})")
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
