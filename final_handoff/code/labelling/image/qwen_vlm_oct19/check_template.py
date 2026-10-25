"""CPU check of the chat template / processor for score.py: token count per read, the Yes / No ids, the tail of the prompt."""
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from score import INTRO, VIEWS, prompt  # noqa: E402

proc = AutoProcessor.from_pretrained(sys.argv[1])
tok = proc.tokenizer
print("yes / no ids", tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False))
imgs = [Image.open(Path(__file__).parent / f"views/0/{s}.png").convert("RGB") for s in VIEWS["A"]["Synovitis"]]
msg = [{"role": "user", "content": [{"type": "text", "text": INTRO}] + [{"type": "image", "image": im} for im in imgs]
        + [{"type": "text", "text": prompt("Synovitis")}]}]
x = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)
print({k: tuple(v.shape) for k, v in x.items()})
print(repr(tok.decode(x["input_ids"][0, -60:])))
