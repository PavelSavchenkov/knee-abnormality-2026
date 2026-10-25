"""Build text_labeling_bundle/prompt.txt from prompt_template.txt and the tested examples (examples.py writes examples_block.txt)."""
from pathlib import Path

HERE = Path(__file__).parent
B = Path("/home/pavel/Programming/knee-abnormality-2026/text_labeling_bundle")
text = (HERE / "prompt_template.txt").read_text(encoding="utf-8")
assert text.count("{{EXAMPLES}}") == 1
out = text.replace("{{EXAMPLES}}", (HERE / "examples_block.txt").read_text(encoding="utf-8").strip()) + "\n"
(B / "prompt.txt").write_text(out, encoding="utf-8")
print(f"prompt.txt: {len(out)} chars, about {len(out) // 4} tokens")
