"""Build text_labeling_bundle_v2/ (ChatGPT, fact extraction) and text_labeling_bundle_v3/ (local Claude, image-read forecast).

Writes each bundle's schema.json from vocab.py, copies reports.csv and definitions.md from the v1 bundle, checks that every
vocabulary value is explained in the bundle's prompt, validates every worked example of the prompt with the bundle's own
validate.py, writes the one-line session prompts (session_prompts_v2.txt, session_prompts_v3.txt next to this file) and the
upload archive text_labeling_bundle_v2.zip at the repo root.

    python3 work/text_labeling/v2v3/build.py"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from vocab import FINDINGS, V2_ASSERTIONS, V2_ATTRIBUTES, V2_FAMILIES, V2_SURGERY, V3_CLASSES, V3_CLUES, v2_schema, v3_schema

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V1, V2, V3 = REPO / "text_labeling_bundle", REPO / "text_labeling_bundle_v2", REPO / "text_labeling_bundle_v3"
N_STUDIES = 4407
BATCH = 50


def write_bundle_files(bundle, schema):
    (bundle / "schema.json").write_text(json.dumps(schema, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in ("reports.csv", "definitions.md"):
        shutil.copyfile(V1 / name, bundle / name)


def check_vocab_in_prompt(prompt, words, label):
    missing = sorted({w for w in words if f"`{w}`" not in prompt})
    if missing:
        sys.exit(f"{label}: not explained in the prompt (as `value`): {missing}")


def check_examples(bundle, prompt, label):
    """Every <output> block of the prompt must pass the bundle's validate.py."""
    outputs = re.findall(r"<output>\s*(.*?)\s*</output>", prompt, re.S)
    if not outputs:
        sys.exit(f"{label}: no examples found")
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for text in outputs:
            x = json.loads(text)
            f = Path(tmp) / f"{x['study_id']}.json"
            f.write_text(json.dumps(x, ensure_ascii=False), encoding="utf-8")
            files.append(str(f))
        for f in files:
            r = subprocess.run([sys.executable, str(bundle / "validate.py"), f], capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"{label}: example {Path(f).name} fails validation:\n{r.stdout}")
    print(f"{label}: {len(outputs)} examples valid")


def build_v2():
    write_bundle_files(V2, v2_schema())
    prompt = (V2 / "prompt.txt").read_text(encoding="utf-8")
    words = [s for fam in V2_FAMILIES.values() for s in fam[0] + fam[1]] + V2_ASSERTIONS + V2_SURGERY
    words += [v for k, s in V2_ATTRIBUTES.values() for v in s.get("enum", s.get("items", {}).get("enum", []))]
    words += [k for k, _ in V2_ATTRIBUTES.values()]
    check_vocab_in_prompt(prompt, words, "v2")
    check_examples(V2, prompt, "v2")
    lines = []
    for lo in range(0, N_STUDIES, BATCH):
        r = f"{lo}-{min(lo + BATCH - 1, N_STUDIES - 1)}"
        lines.append(f"Unzip text_labeling_bundle_v2.zip from the project files, follow its prompt.txt exactly, and deliver v2_{r}.zip "
                     f"as it specifies. Range: {r}")
    (HERE / "session_prompts_v2.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    zpath = REPO / "text_labeling_bundle_v2.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("prompt.txt", "definitions.md", "schema.json", "validate.py", "reports.csv"):
            z.write(V2 / name, f"text_labeling_bundle_v2/{name}")
    print(f"v2: {len(lines)} session prompts, {zpath.name} {zpath.stat().st_size / 1e6:.1f} MB")


def build_v3():
    write_bundle_files(V3, v3_schema())
    (V3 / "out").mkdir(exist_ok=True)
    prompt = (V3 / "prompt.md").read_text(encoding="utf-8")
    check_vocab_in_prompt(prompt, list(V3_CLASSES) + V3_CLUES + FINDINGS, "v3")
    check_examples(V3, prompt, "v3")
    lines = [f"You are text-labelling session {k} of 5. Read prompt.md in {V3} and follow it exactly. Part: {k}" for k in range(5)]
    (HERE / "session_prompts_v3.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"v3: {len(lines)} session prompts")


if __name__ == "__main__":
    build_v2()
    build_v3()
