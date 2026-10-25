"""Build data/image_labels/ from the raw image-label runs (VLM reads of the knee MRI bundle, 2026-10-07..09, 10-16 and 10-20).

conda run -n rsna python data/build/build_image_labels.py

Reads: work/image_labeling/out/order.csv (bundle index -> study, catch copies), the returned ChatGPT archives in
work/image_labeling/returned/ and the label folders of the local labellers in work/{codex,claude,claude_fable51}_labels_oct7/, work/claude_labels_oct16/
and work/claude_labels_oct20/ (images + report reads; selection keys in its _key/p2s.csv).
Writes, per run, data/image_labels/<run>.json keyed by study_id (one read per study), repeats.json (catch copies and gold re-reads, with
the study they repeat), reads.csv (one row per read and finding) and nothing else. Geometry that refers to the bundle's tiles (boxes,
evidence tile ids, structure boxes, drawn-mark positions) is left out; it stays in the raw files.
"""
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/image_labels"
ORDER = REPO / "work/image_labeling/out/order.csv"
RET = REPO / "work/image_labeling/returned"
CODEX, OPUS, FABLE = REPO / "work/codex_labels_oct7", REPO / "work/claude_labels_oct7", REPO / "work/claude_labels_fable51_oct7"
OPUS16 = REPO / "work/claude_labels_oct16"
OPUS20 = REPO / "work/claude_labels_oct20"
OPUS20_MODEL = "Claude Opus 5.5 (Claude Code session, with native views; images + radiology report)"
# status x certainty -> probability (copy of work/image_labeling/mapping.py DEFAULT)
PROB = {"meets": {"high": 0.95, "moderate": 0.85, "low": 0.7}, "borderline": {"high": 0.55, "moderate": 0.5, "low": 0.45},
        "possible": {"high": 0.4, "moderate": 0.35, "low": 0.3}, "below": {"high": 0.1, "moderate": 0.15, "low": 0.2},
        "excluded": {"high": 0.05, "moderate": 0.08, "low": 0.12}, "normal": {"high": 0.02, "moderate": 0.04, "low": 0.08}}
CODEX_MODEL = "Codex, gpt-6-astra, reasoning xhigh (local, with native full-resolution views)"
# run -> (labeller, model, panel, prompt, sources); sources are zip files or label folders
RUNS = {
    "chatgpt_v1_gold": ("chatgpt-web", "ChatGPT web, astra xhigh", "all12", "v1", [RET / "goldA_img_0-9.zip"]),
    "chatgpt_v2_gold": ("chatgpt-web", "ChatGPT web, astra xhigh", "all12", "v2", [RET / "v2A_img_0-9.zip"]),
    "chatgpt_pro_v2_gold": ("chatgpt-web", "ChatGPT web, Astra Pro", "all12", "v2", [RET / "v2P_img_0-9.zip"]),
    "chatgpt_v4_gold": ("chatgpt-web", "ChatGPT web, astra xhigh", "all12", "v4", [RET / "v4A_img_20-29.zip", RET / "v4A_img_50-59.zip"]),
    "codex_v2_gold": ("codex", CODEX_MODEL, "all12", "v2", [CODEX / "sanity_0-9"]),
    "codex_v4_gold": ("codex", CODEX_MODEL, "all12", "v4", [CODEX / f"gold_{a}-{a + 9}" for a in (10, 20, 30, 40, 50)]),
    "codex_fluid_gold": ("codex", CODEX_MODEL, "fluid", "fluid", [CODEX / "fluid_0-29", CODEX / "fluid_30-59"]),
    "codex_report_image_gold": ("codex", CODEX_MODEL, "all12", "v4+report", [CODEX / "gold_text_image"]),
    "codex_lateral_oa_gold": ("codex", CODEX_MODEL, "lateral_oa", "lateral_oa", [CODEX / f"lateral_oa_b{k}" for k in range(1, 5)]),
    "codex_bakers": ("codex", CODEX_MODEL, "bakers", "bakers", [CODEX / f"bakers_b{k}" for k in range(1, 13)]),
    "codex_bakers_gold": ("codex", CODEX_MODEL, "bakers", "bakers", [CODEX / f"bakers_b{k}" for k in range(1, 13)]),
    "opus_v4_gold": ("claude", "Claude Opus 5.5 (Claude Code session, with native views)", "all12", "v4",
                     [OPUS / "sanity_0-9"] + [OPUS / f"gold_{a}-{a + 9}" for a in (10, 20, 30)]),
    "opus_fluid_gold": ("claude", "Claude Opus 5.5 (Claude Code session, with native views)", "fluid", "fluid", [OPUS / "fluid_0-29", OPUS / "fluid_30-59"]),
    "fable_fluid_gold": ("claude", "Claude Fable 5.1 (Claude Code session, with native views)", "fluid", "fluid",
                         [FABLE / "fluid_0-29", FABLE / "fluid_30-59"]),
    "codex_synovitis": ("codex", CODEX_MODEL, "synovitis", "synovitis",
                        [CODEX / f"synovitis_{r}" for r in ("60-359", "360-659", "660-959")] + [CODEX / f"synovitis_b{k}" for k in range(1, 29)]
                        + [CODEX / f"synovitis_extra_b{k}" for k in range(1, 29)]),
    "codex_synovitis_reread": ("codex", CODEX_MODEL, "synovitis", "synovitis",          # second blind reads of uncertain first reads
                               [CODEX / f"synovitis_r{k}" for k in range(1, 6)] + [CODEX / f"synovitis_extra_r{k}" for k in range(1, 6)]),
    # effusion second reads of text-vague studies (2026-10-16), gold rows mixed in
    "opus_effusion": ("claude", "Claude Opus 5.5 (Claude Code session, with native views)", "fluid", "fluid", [OPUS16 / f"eff_c{k}" for k in range(1, 6)]),
    "opus_effusion_gold": ("claude", "Claude Opus 5.5 (Claude Code session, with native views)", "fluid", "fluid",
                           [OPUS16 / f"eff_c{k}" for k in range(1, 6)]),
    # images + report reads (2026-10-20): pilot on gold (fast pass, then strict native-zoom protocol), then Phase 2 on non-gold studies
    "opus_pilot5_fast_gold": ("claude", OPUS20_MODEL, "pilot5", "pilot5+report fast", [OPUS20 / "pilot_c1_v1", OPUS20 / "pilot_c2_v1"]),
    "opus_pilot5_gold": ("claude", OPUS20_MODEL, "pilot5", "pilot5+report", [OPUS20 / "pilot_c1", OPUS20 / "pilot_c2"]),
    "opus_p2": ("claude", OPUS20_MODEL, "p2", "p2+report", [OPUS20 / f"p2_s{k}" for k in range(1, 8)] + [OPUS20 / f"p2_s7_h{h}" for h in (1, 2, 4, 6, 5)]),
    # phase 3: only the listed finding(s) per study (selected for expected positives), fewer sheets; blind re-reads of p2 studies -> repeats
    "opus_p3": ("claude", OPUS20_MODEL, "p3", "p3+report", [OPUS20 / f"p3_s{k}" for k in range(1, 36)]),
    # phase 3 closing batch (10-21): the listed finding on studies whose old-label bone target is >= .3 without an Opus read of it; a study may
    # also hold an earlier opus_p3 read of OTHER findings, so the batch is its own run (merge per finding downstream)
    "opus_p3c": ("claude", OPUS20_MODEL, "p3", "p3+report", [OPUS20 / f"p3_s{k}" for k in range(36, 44)]),
    # phase 3 PF batch (10-21 11:3x): pf_oa only on a random sample of unread old-label PF positives (text p in [.6, .8) and [.8, .9)), with
    # the depth record "d~N% e~M mm;" at the start of the note (build_phase3_pf.py); its own run like opus_p3c
    "opus_p3pf": ("claude", OPUS20_MODEL, "p3", "p3+report depth", [OPUS20 / f"p3_s{k}" for k in range(44, 60)]),
    # phase 3 PF RE-READ batch (10-21 13:3x): the 249 p2-read text-'unsettled' PF studies re-read with the p3pf protocol (build_phase3_pfr.py);
    # its own run: a study keeps its opus_p2 read (all findings) and gains this pf_oa read; merge per finding downstream (p3pfr wins for PF)
    "opus_p3pfr": ("claude", OPUS20_MODEL, "p3", "p3+report depth reread", [OPUS20 / f"p3_s{k}" for k in range(60, 73)]),
    # phase 4 (10-21): Opus second reads of the Codex Baker's pool (bakers_cyst only), images + report
    "opus_bakers_p4": ("claude", OPUS20_MODEL, "p4", "p4+report", [OPUS20 / f"p4_s{k}" for k in range(1, 9)]),
    # gold feasibility (10-21): ACL / MCL / menisci / MOA / LOA on gold-58, images + report + a native zoom crop per finding
    "opus_p5_gold": ("claude", OPUS20_MODEL, "p5", "p5+report zoom", [OPUS20 / f"g5_s{k}" for k in range(1, 5)]),
    # host-alignment gold check (10-21): fracture site checklist + PF depth / extent record, images + report, panel p3 (fracture, pf_oa)
    "opus_g6_gold": ("claude", OPUS20_MODEL, "p3", "p3+report checklist", [OPUS20 / f"g6_s{k}" for k in range(1, 5)]),
}
GOLD_ONLY, POOL_ONLY = {"opus_pilot5_fast_gold", "opus_pilot5_gold", "opus_p5_gold", "opus_g6_gold"}, {"opus_p2", "opus_p3", "opus_p3c", "opus_p3pf", "opus_p3pfr", "opus_bakers_p4"}
P2_PAUSE = pd.Timestamp("2026-10-20 12:00", tz="Europe/London")   # labelling paused 11:27-16:01; wave 1 = the 559 reads frozen for training waves 1-4
DROP = {"evidence", "boxes"}
# runs that share label folders with gold rows mixed in: keep only the non-gold ("pool") or only the gold rows; a gold row read again
# in a later folder goes to repeats.json as a gold_reread
SPLIT = {"codex_bakers": "pool", "codex_bakers_gold": "gold", "opus_effusion": "pool", "opus_effusion_gold": "gold"}


def files(src):
    """[(file name, bytes, time written)] of one zip archive or label folder, in writing order."""
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            m = [(Path(i.filename).name, z.read(i), i.date_time) for i in z.infolist() if i.filename.endswith(".json")]
    else:
        m = [(f.name, f.read_bytes(), f.stat().st_mtime) for f in src.glob("*.json")]
    return sorted(m, key=lambda x: (x[2], x[0]))


def record(x, run, src, pos):
    labeller, model, panel, prompt, _ = RUNS[run]
    findings = {}
    for k, f in x["findings"].items():
        p = None if f["status"] == "not_assessable" else PROB[f["status"]][f["certainty"]]
        findings[k] = {"status": f["status"], "certainty": f["certainty"], "probability": p} | \
                      {a: v for a, v in f.items() if a not in DROP | {"status", "certainty"}}
    qa = {k: v for k, v in x["qa"].items() if k != "marks"}
    return {"labeller": labeller, "model": model, "panel": panel, "prompt": prompt, "source": src, "position": pos,
            "sheets_viewed": x["sheets_viewed"], "qa": qa, "findings": findings}


def main():
    order = pd.read_csv(ORDER, index_col="index")
    p2key = pd.read_csv(OPUS20 / "_key/p2s.csv").set_index("index")
    p3key = pd.read_csv(OPUS20 / "_key/p3.csv").set_index(["index", "session"])
    OUT.mkdir(parents=True, exist_ok=True)
    repeats, rows, summary = [], [], []
    for run, (*_, sources) in RUNS.items():
        out = {}
        for src in sources:
            if not src.exists():
                continue
            rel = str(src.relative_to(REPO))
            for pos, (name, data, t) in enumerate(files(src)):
                x = json.loads(data)
                i = x["index"]
                assert name == f"{x['study_id']}.json" and order.study_id[i] == x["study_id"], (rel, name)
                if run in SPLIT and (order.section[i] == "gold") != (SPLIT[run] == "gold"):
                    continue
                assert not (run in GOLD_ONLY and order.section[i] != "gold") and not (run in POOL_ONLY and order.section[i] == "gold"), (run, i)
                rec = {"study_id": order.source[i]} | record(x, run, rel, pos)
                if run == "codex_synovitis" and not order.catch[i] and order.section[i] != "gold":
                    rec["selection"] = {"sampling": "batch" if re.fullmatch(r"synovitis_b\d+", src.name) else "queue", "stratum": order.stratum[i]}
                if run == "opus_p2":
                    k = p2key.loc[i]
                    assert k.study_id == x["study_id"], (rel, i)
                    saved = pd.Timestamp(t, unit="s", tz="UTC").tz_convert("Europe/London")
                    rec["selection"] = {"stratum": k.stratum, "list": f"p2_s{k.session}", "list_position": int(k.pos),
                                        "wave": 1 if saved < P2_PAUSE else 2, "saved_at": saved.strftime("%Y-%m-%d %H:%M")}
                if run in ("opus_p3", "opus_p3c", "opus_p3pf", "opus_p3pfr"):
                    k = p3key.loc[(i, int(src.name.removeprefix("p3_s")))]
                    assert k.study_id == x["study_id"] and set(x["findings"]) == set(k.findings.split(",")), (rel, i)
                    saved = pd.Timestamp(t, unit="s", tz="UTC").tz_convert("Europe/London")
                    rec["selection"] = {"listed_findings": k.findings.split(","), "list": src.name, "list_position": int(k.pos),
                                        "expected_positives": None if pd.isna(k.expected_pos) else round(float(k.expected_pos), 3),
                                        "saved_at": saved.strftime("%Y-%m-%d %H:%M")}
                kind = "protocol_reread" if run in ("opus_p3", "opus_p3c", "opus_p3pf", "opus_p3pfr") and k.kind == "reread" else "catch_copy" if order.catch[i] else "gold_reread" if "extra" in src.name or (run in SPLIT and rec["study_id"] in out) else None
                if kind:
                    repeats.append({"run": run, "kind": kind} | rec)
                else:
                    assert rec["study_id"] not in out, (run, rec["study_id"])
                    out[rec["study_id"]] = rec
                for k, f in rec["findings"].items():
                    rows.append({"study_id": rec["study_id"], "run": run, "repeat": kind or "", "labeller": rec["labeller"], "panel": rec["panel"],
                                 "prompt": rec["prompt"], "finding": k, "status": f["status"], "certainty": f["certainty"],
                                 "probability": f["probability"]})
        json.dump(out, open(OUT / f"{run}.json", "w"), indent=1, ensure_ascii=False)
        summary.append((run, len(out)))
    json.dump(repeats, open(OUT / "repeats.json", "w"), indent=1, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(OUT / "reads.csv", index=False)
    for run, n in summary:
        print(f"{run:24} {n:5} studies")
    print(f"repeats {len(repeats)} | reads.csv {len(rows)} rows")


if __name__ == "__main__":
    main()
