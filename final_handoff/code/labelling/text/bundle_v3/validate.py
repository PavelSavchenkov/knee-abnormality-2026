"""Check forecast files against schema.json and reports.csv. Pure Python, no extra packages.

Usage (schema.json and reports.csv must sit next to this file):
  python3 validate.py --part K           every index of part K (index % 5 == K) that has a file in out/ must pass; prints how many
                                         of the part are done, valid and still to do
  python3 validate.py --part K --todo N  also print the next N indexes of part K that have no valid file yet
  python3 validate.py FILE.json ...      check the given files

Checks: the JSON schema (including the p range of each evidence class and an empty quote exactly for silent findings); index and
study_id match reports.csv; the file is named <study_id>.json; every quote piece (pieces are split on '...') appears in the report,
ignoring case and whitespace. Exit code 1 if any checked file fails."""
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
    for name, f in x["findings"].items():
        out += quote_errors(f["quote"], report, f"$.findings.{name}.quote")
    return out


def main(args):
    if args[:1] == ["--part"] and len(args) in (2, 4):
        k = int(args[1])
        todo_n = int(args[3]) if len(args) == 4 and args[2] == "--todo" else 0
        part = [i for i in sorted(ROWS) if i % 5 == k]
        out_dir = HERE / "out"
        problems, todo = {}, []
        for i in part:
            f = out_dir / f"{ROWS[i]['study_id']}.json"
            if f.exists():
                problems[f.name] = check_file(f, i)
            if not f.exists() or problems[f.name]:
                todo.append(i)
        bad = {k: v for k, v in problems.items() if v}
        for name, msgs in bad.items():
            print(name)
            for m in msgs:
                print("  -", m)
        print(f"part {k}: {len(part)} studies, {len(problems)} files, {len(problems) - len(bad)} valid, {len(bad)} with problems, "
              f"{len(todo)} still to do")
        if todo_n:
            print("next:", " ".join(map(str, todo[:todo_n])))
        sys.exit(1 if bad else 0)
    if not args or args[0].startswith("--"):
        sys.exit(__doc__)
    problems = {a: check_file(a) for a in args}
    bad = {k: v for k, v in problems.items() if v}
    for name, msgs in bad.items():
        print(name)
        for m in msgs:
            print("  -", m)
    print(f"{len(problems) - len(bad)} file(s) OK, {len(bad)} with problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
