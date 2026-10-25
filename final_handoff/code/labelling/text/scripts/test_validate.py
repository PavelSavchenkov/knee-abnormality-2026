"""Test text_labeling_bundle/validate.py: the worked examples pass; on random mutations its schema check agrees with jsonschema;
range mode reports missing and extra files."""
import copy
import importlib.util
import json
import random
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
from pathlib import Path

import jsonschema

B = Path("/home/pavel/Programming/knee-abnormality-2026/text_labeling_bundle")
sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location("validate", B / "validate.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
import examples as E                                               # noqa: E402  (builds EX with study_id)

docs = [{"index": e["index"], "study_id": e["study_id"], "study": e["study"], "findings": e["findings"]} for e in E.EX]
tmp = Path(tempfile.mkdtemp())
for d in docs:
    (tmp / f"{d['study_id']}.json").write_text(json.dumps(d, ensure_ascii=False))
assert all(V.check_file(tmp / f"{d['study_id']}.json") == [] for d in docs), "examples should pass"

js = jsonschema.Draft202012Validator(V.SCHEMA)
values = ["meets", "unsettled", "nonsense", None, True, False, 0, -1, 3.5, "", "x" * 400, [], ["size"], ["size", "size"], ["bogus"],
          ["posterior_horn"], {"a": 1}, "low_grade", "full", "trauma", "chronic", 0.9, 0.97, 0.33, 0.05, 0.6, 0.01]
rng = random.Random(0)
agree = n = 0
for _ in range(3000):
    d = copy.deepcopy(rng.choice(docs))
    for _ in range(rng.randint(1, 3)):
        node, parents = d, []
        while isinstance(node, dict) and node and rng.random() < 0.8:
            k = rng.choice(list(node))
            parents.append((node, k))
            node = node[k]
        if not parents:
            continue
        parent, k = parents[-1]
        r = rng.random()
        if r < 0.7:
            parent[k] = copy.deepcopy(rng.choice(values))
        elif r < 0.85:
            del parent[k]
        else:
            parent["extra_key"] = 1
    a = bool(V.errors(d, V.SCHEMA))
    b = bool(list(js.iter_errors(d)))
    agree += a == b
    n += 1
print(f"schema check agrees with jsonschema on {agree}/{n} mutated documents")
assert agree == n

# semantic checks
d = copy.deepcopy(docs[0]); d["findings"]["acl"]["quote"] = "this sentence is not in the report"
p = tmp / f"{d['study_id']}.json"; p.write_text(json.dumps(d))
assert any("not found in the report" in m for m in V.check_file(p))
d = copy.deepcopy(docs[0]); d["findings"]["medial_oa"]["depth"] = "full"; p.write_text(json.dumps(d))
assert any("when status is not_mentioned" in m for m in V.check_file(p))
d = copy.deepcopy(docs[0]); d["study_id"] = docs[1]["study_id"]; p.write_text(json.dumps(d))
assert any("does not match" in m for m in V.check_file(p))
p.write_text("{not json"); assert V.check_file(p)[0].startswith("not valid JSON")
p.write_text(json.dumps(docs[0], ensure_ascii=False))

# range mode: 6..6 present, 5..7 has two missing plus out-of-range extras
r = subprocess.run([sys.executable, str(B / "validate.py"), str(tmp), "5", "7"], capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1], "| exit", r.returncode)
assert r.returncode == 1 and r.stdout.count("file missing") == 2 and "not in the requested range" in r.stdout
one = tmp / "one"; one.mkdir(); (one / f"{docs[2]['study_id']}.json").write_text(json.dumps(docs[2], ensure_ascii=False))
r = subprocess.run([sys.executable, str(B / "validate.py"), str(one), "6", "6"], capture_output=True, text=True)
print(r.stdout.strip(), "| exit", r.returncode)
assert r.returncode == 0
print("all validate.py tests passed")
