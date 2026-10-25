"""Zero-shot yes / no reads of the pilot studies by an open VLM: per (study, finding) one forward pass over the finding's view sheets
and the host definition; score = P(Yes) / (P(Yes) + P(No)) at the first answer token (thinking off).

PYTHONPATH=.:work/vlm/pylib conda run -n rsna python work/vlm_label_2026-10-19/score.py MODEL VIEWSET OUT.csv [--bits 4]
    [--findings ACL,MCL] [--limit N] [--studies CSV] [--max-edge PX] [--guide] [--shots CSV] [--think N]
VIEWSET: a key of VIEWS (sheets of views/<index>/, rendered by render_views.py)."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from common.paths import FINDINGS

HERE = Path(__file__).parent
TEXT = {"ACL": "ACL tear", "MCL": "MCL tear", "Medial Meniscus": "medial meniscus tear", "Lateral Meniscus": "lateral meniscus tear",
        "Medial OA": "medial compartment osteoarthritis", "Lateral OA": "lateral compartment osteoarthritis",
        "PF OA": "patellofemoral compartment osteoarthritis", "Effusion": "joint effusion", "Synovitis": "synovitis",
        "Baker's": "Baker (popliteal) cyst", "Contusion": "bone contusion", "Fracture": "acute fracture"}
# host definitions (text_labeling_bundle_v3/definitions.md), verbatim
DEF = {
    "ACL": "A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, "
           "or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone "
           "contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative.",
    "MCL": "A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and "
           "adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.",
    "Medial Meniscus": "Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality "
                       "such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance "
                       "degeneration that does not reach the surface is negative.",
    "Lateral Meniscus": "Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality "
                        "such as a truncated, diminutive, or displaced fragment, involving the lateral meniscus. Intrasubstance "
                        "degeneration that does not reach the surface is negative.",
    "Medial OA": "A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of "
                 "cartilage thickness, in the medial compartment, with or without underlying subchondral marrow changes.",
    "Lateral OA": "A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of "
                  "cartilage thickness, in the lateral compartment, with or without underlying subchondral marrow changes.",
    "PF OA": "A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of "
             "cartilage thickness, in the patellofemoral compartment, with or without underlying subchondral marrow changes.",
    "Effusion": "A moderate or large amount of fluid distending the joint.",
    "Synovitis": "Inflammation and thickening of the synovial lining of the joint.",
    "Baker's": "A moderate or large fluid collection in the characteristic location behind the knee.",
    "Contusion": "A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line.",
    "Fracture": "An acute cortical break or fracture line.",
}
VIEWS = {
    "A": {"ACL": ["5_notch", "1_sag_fs"], "MCL": ["2_cor_fs", "8_cor_joint"], "Medial Meniscus": ["6_medial", "8_cor_joint"],
          "Lateral Meniscus": ["7_lateral", "8_cor_joint"], "Medial OA": ["6_medial", "8_cor_joint"],
          "Lateral OA": ["7_lateral", "8_cor_joint"], "PF OA": ["9_patella", "3_ax_fs"], "Effusion": ["1_sag_fs", "3_ax_fs"],
          "Synovitis": ["1_sag_fs", "3_ax_fs"], "Baker's": ["3_ax_fs", "1_sag_fs"], "Contusion": ["1_sag_fs", "2_cor_fs"],
          "Fracture": ["2_cor_fs", "4_nonfs"]},
    # the sheets the Codex / Opus readers used (data/image_labels/README.md sheets_viewed; Brainstorm 10-19)
    "B": {"Effusion": ["1_sag_fs", "3_ax_fs", "5_notch", "9_patella"], "Synovitis": ["1_sag_fs", "3_ax_fs", "5_notch", "9_patella"],
          "Baker's": ["3_ax_fs", "1_sag_fs", "6_medial"]},
    "Z": {"Effusion": ["5_notch", "9_patella"], "Synovitis": ["5_notch", "9_patella"], "Baker's": ["6_medial"]},
    # single frames ("sheet:k,..." = tiles k of the sheet, 1-based in grid order, each its own image at 2x)
    "F": {"Effusion": ["1_sag_fs:4,5,6", "3_ax_fs:4,5,6", "9_patella:7,8,9"],
          "Synovitis": ["1_sag_fs:4,5,6", "3_ax_fs:4,5,6", "5_notch:4,5,6", "9_patella:7,8,9"],
          "Baker's": ["3_ax_fs:2,3,4", "1_sag_fs:2,3,4"]},
}
# structure ZOOM crops (render_zoom.py: the zoom members' localised 40-48 mm crops, FS, 448 px): "zoom/<bag>_<plane>_<k>"
VIEWS["Z2"] = {"ACL": ["zoom/notch40_Sagittal_0", "zoom/notch40_Sagittal_1", "zoom/notch40_Coronal_0"],
               "MCL": ["zoom/mclz_Coronal_0", "zoom/mclz_Coronal_1"],
               "Medial Meniscus": ["zoom/medc40_Sagittal_0", "zoom/medc40_Sagittal_1", "zoom/medc40_Coronal_0"],
               "Lateral Meniscus": ["zoom/latc40_Sagittal_0", "zoom/latc40_Sagittal_1", "zoom/latc40_Coronal_0"],
               "Medial OA": ["zoom/medc40_Coronal_0", "zoom/medc40_Coronal_1", "zoom/medc40_Sagittal_0"],
               "Lateral OA": ["zoom/latc40_Coronal_0", "zoom/latc40_Coronal_1", "zoom/latc40_Sagittal_0"],
               "PF OA": ["zoom/pfl40_Sagittal_0", "zoom/pfl40_Axial_0", "zoom/pfl40_Axial_1"]}
VIEWS["ZE"] = {"Effusion": ["zoom/pouch48_Sagittal_0", "zoom/pouch48_Sagittal_1", "zoom/pouch48_Axial_0", "zoom/pouch48_Axial_1"]}
REGION = {"Effusion": "the suprapatellar recess (joint fluid)", "ACL": "the intercondylar notch (anterior cruciate ligament)", "MCL": "the medial collateral ligament", "Medial Meniscus": "the medial compartment (medial meniscus)",
          "Lateral Meniscus": "the lateral compartment (lateral meniscus)", "Medial OA": "the medial compartment cartilage",
          "Lateral OA": "the lateral compartment cartilage", "PF OA": "the patellofemoral joint cartilage"}
ZOOM_INTRO = ("You are a musculoskeletal radiologist reading a knee MRI. The images are ZOOMED crops (about 40 mm wide) of one knee "
              "examination, centred on {region}, from fluid-sensitive (fat-suppressed) series; sagittal crops have anterior on the left, "
              "coronal and axial crops show the knee as a LEFT knee. Several crops are neighbouring slices or other planes of the same structure.")
COLS, PX, GAP, HEAD = 3, 400, 6, 30          # sheet grid of work/image_labeling/render.py


def images(index, items, max_edge=0):
    """PIL images for a list of view items of one study: whole sheets, or 'sheet:k,...' single tiles upscaled 2x."""
    out = []
    for it in items:
        name, _, ks = it.partition(":")
        path = HERE / "zoom" / str(index) / f"{name[5:]}.png" if name.startswith("zoom/") else HERE / "views" / str(index) / f"{name}.png"
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        if ks:
            for k in map(int, ks.split(",")):
                r, c = divmod(k - 1, COLS)
                y, x = HEAD + r * (PX + GAP), c * (PX + GAP)
                if y + PX <= im.height:
                    out.append(im.crop((x, y, x + PX, y + PX)).resize((2 * PX, 2 * PX), Image.LANCZOS))
        else:
            if max_edge:
                im.thumbnail((max_edge, max_edge))
            out.append(im)
    return out
# per-finding reading guide of the Codex bundle prompt: section header prefix in bundle_src/prompt.txt
SECTION = {"ACL": "ACL tear", "MCL": "MCL tear", "Medial Meniscus": "Meniscal tear", "Lateral Meniscus": "Meniscal tear",
           "Medial OA": "Osteoarthritis", "Lateral OA": "Osteoarthritis", "PF OA": "Osteoarthritis", "Effusion": "Joint effusion",
           "Synovitis": "Synovitis", "Baker's": "Baker's cyst", "Contusion": "Bone contusion", "Fracture": "Acute fracture"}
GUIDE_KEYS = ("- Look at", "- Normal", "- Signs that count", "- Does not count")


def guide(f):
    """The 'Look at / Normal / Signs that count / Does not count' lines of the finding's section of the Codex bundle prompt."""
    lines = (HERE.parent / "image_labeling/bundle_src/prompt.txt").read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(f"### {SECTION[f]}"))
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("#"))
    return "\n".join(ln for ln in lines[i + 1:j] if ln.startswith(GUIDE_KEYS))
INTRO = ("You are a musculoskeletal radiologist reading a knee MRI. The images are contact sheets of one knee examination. Each tile is "
         "labelled with its id, the series (plane, weighting, FS = fat-suppressed, i.e. fluid-sensitive: fluid and edema are bright), "
         "and the slice position in mm; letters on the tile edges give the orientation (S/I superior/inferior, A/P anterior/posterior, "
         "M/L medial/lateral); a 10 mm scale bar is in the corner. Every knee is displayed as a LEFT knee.")


GRADE_Q = ("Question: grade the joint effusion on these images: 0 = none or physiologic, 1 = small, 2 = moderate, 3 = large. "
           "Answer with one digit.")


def prompt(f, with_guide=False, grade=False):
    g = f"Reading guide:\n{guide(f)}\n" if with_guide else ""
    if grade:
        return f"\n\nFinding: {TEXT[f]}.\nOfficial definition: {DEF[f]}\n{g}\n{GRADE_Q}"
    return (f"\n\nFinding: {TEXT[f]}.\nOfficial definition: {DEF[f]}\nAmbiguous or borderline findings are graded negative.\n{g}\n"
            f"Question: by this definition, does this knee have a {TEXT[f]}? Answer with one word: Yes or No.")


def user_turn(imgs, text):
    return {"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": text}]}


def think(model, tok, x, n, stop):
    """Greedy reasoning up to n tokens (stops at </think>), then the inputs extended to the answer position: (inputs, thought text)."""
    out = model.generate(**x, max_new_tokens=n, do_sample=False, eos_token_id=stop)[0]
    gen = out[x["input_ids"].shape[1]:]
    ended = len(gen) and gen[-1].item() == stop[0]
    tail = torch.tensor(tok.encode("\n\n" if ended else "\n</think>\n\n", add_special_tokens=False), device=out.device)
    ids = torch.cat([out, tail])[None]
    y = dict(x)
    y["input_ids"], y["attention_mask"] = ids, torch.ones_like(ids)
    if "mm_token_type_ids" in x:
        y["mm_token_type_ids"] = torch.cat([x["mm_token_type_ids"], torch.zeros(1, ids.shape[1] - x["input_ids"].shape[1],
                                                                                  dtype=x["mm_token_type_ids"].dtype, device=ids.device)], 1)
    return y, tok.decode(gen, skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("viewset")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--findings", default=",".join(FINDINGS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--studies", default=str(HERE / "out/pilot_studies.csv"))
    ap.add_argument("--guide", action="store_true", help="add the Codex prompt's reading guide for the finding")
    ap.add_argument("--max-edge", type=int, default=0, help="downscale sheets so the longer edge is at most this (0 = native)")
    ap.add_argument("--shots", default="", help="CSV index,finding,answer: in-context examples (rendered in views/) before the query")
    ap.add_argument("--grade", action="store_true", help="ask for a 0-3 grade; score = P(grade >= 2) from the digit tokens")
    ap.add_argument("--think", type=int, default=0, help="thinking on: generate up to N reasoning tokens (greedy), then read Yes / No")
    a = ap.parse_args()
    shots = pd.read_csv(a.shots) if a.shots else pd.DataFrame(columns=["index", "finding", "answer"])
    studies = pd.read_csv(a.studies, index_col="index")
    if a.limit:
        studies = studies.iloc[:a.limit]
    findings = [f.replace("_", " ") for f in a.findings.split(",")]      # "_" for spaces (the pass runner splits on spaces)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) if a.bits == 4 else None
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", quantization_config=quant).eval()
    tok = proc.tokenizer
    yes, no = (tok.encode(w, add_special_tokens=False) for w in ("Yes", "No"))
    assert len(yes) == len(no) == 1, (yes, no)
    head = model.lm_head.weight[[yes[0], no[0]]].detach().float()       # fp32 yes / no logits: bf16 logits are quantised to ~0.1
    digits = [tok.encode(str(k), add_special_tokens=False) for k in range(4)]
    assert all(len(d) == 1 for d in digits), digits
    ghead = model.lm_head.weight[[d[0] for d in digits]].detach().float()
    stop = [tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")]
    thoughts = open(a.out + ".think.jsonl", "a") if a.think else None
    part = Path(a.out + ".part")                      # rows are appended per study (live checks: evaluate.py on the .part file)
    done = set(pd.read_csv(part)["index"]) if part.exists() else set()
    t0 = time.time()
    for n, (i, uid) in enumerate(studies.StudyInstanceUID.items()):
        if i in done:
            continue
        rows = []
        for f in findings:
            imgs = images(i, VIEWS[a.viewset][f], a.max_edge)
            if not imgs:
                rows.append((i, uid, f, float("nan"), 0))
                continue
            intro = ZOOM_INTRO.format(region=REGION[f]) if a.viewset in ("Z2", "ZE") else INTRO
            ex = shots[shots.finding == f]
            if len(ex):                                   # few-shot: INTRO as system, one user / assistant pair per example
                msg = [{"role": "system", "content": [{"type": "text", "text": intro}]}]
                for e in ex.itertuples():
                    msg += [user_turn(images(e.index, VIEWS[a.viewset][f], a.max_edge), prompt(f, a.guide, a.grade)),
                            {"role": "assistant", "content": [{"type": "text", "text": e.answer}]}]
                msg.append(user_turn(imgs, prompt(f, a.guide, a.grade)))
            else:                                         # pass-1 layout: INTRO, images, question in one user turn
                msg = [{"role": "user", "content": [{"type": "text", "text": intro}] + user_turn(imgs, prompt(f, a.guide, a.grade))["content"]}]
            x = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                         enable_thinking=bool(a.think)).to("cuda")
            with torch.inference_mode():
                if a.think:
                    x, thought = think(model, tok, x, a.think, stop)
                    thoughts.write(json.dumps({"index": int(i), "finding": f, "thought": thought}) + "\n")
                    thoughts.flush()
                h = model.model(**x).last_hidden_state[0, -1].float()
                p = torch.softmax(ghead @ h, 0)[2:].sum().item() if a.grade else torch.softmax(head @ h, 0)[0].item()
            rows.append((i, uid, f, p, x["input_ids"].shape[1]))
        pd.DataFrame(rows, columns=["index", "StudyInstanceUID", "finding", "p_yes", "tokens"]).to_csv(
            part, mode="a", header=not part.exists(), index=False)
        if n % 10 == 0:
            print(f"{n + 1}/{len(studies)} {time.time() - t0:.0f}s tokens {rows[-1][4]} peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                  flush=True)
    part.rename(a.out)
    print(f"done {len(studies)} studies {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
