"""Deep read of the Claude (Opus 5.5) image + report pilot on gold-58 (USER 10-20: "go deeper -- you can blend new labels with old ones, find
disagreements, etc -- anything potentially useful signal"). Panel: medial_meniscus, pf_oa, synovitis, contusion, fracture. OAI: no.
Gold numbers stay in this folder (eval_deep.txt), never in memory (labeller blindness). Per finding:
 1. AUC of every reader on the same studies: Opus strict (pilot_c1 / c2), Opus fast pass (pilot_c*_v1, ~30 studies), text v1 (prob, silent
    0.15), the shipped 0.961 package (expected gold, all fold models), Codex image + report (39 gold), Codex fluid (Synovitis), Opus
    image-only all12 (30 gold).
 2. Blends (z-mean of logits, in-sample on gold = INFO): text + Opus, package + 0.5 Opus, Codex img+rep + Opus, text + Opus + Codex img+rep.
 3. Disagreements at 0.5: Opus vs text, Opus vs package -> who gold agrees with.
 4. Decided-by tag (note prefix report / images / both): n and accuracy at 0.5 per tag.
 5. Strict vs fast on the overlap.
 6. Paired bootstrap (2,000) of AUC(Opus) - AUC(text) and AUC(pkg + 0.5 Opus) - AUC(pkg); leave-one-study-out range.
 7. Calibration: gold positive rate per Opus status.
  conda run -n rsna python work/claude_labels_oct20/deep_eval.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import gold_labels, probability_table
from common.paths import FINDINGS, KEY_OF, REPO

D = Path(__file__).parent
sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability  # noqa: E402

PANEL = ["Medial Meniscus", "PF OA", "Synovitis", "Contusion", "Fracture"]
lg = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))  # noqa: E731
z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
gold = gold_labels()
pkg = pd.read_csv(REPO / "experiments/v2/oct14_text_labels/data/expected_gold_f5_win.csv", index_col=0).reindex(gold.index)
text = probability_table("v1").reindex(gold.index)
KM = {"Baker's": "bakers_cyst"}


def score(v):
    if not v:
        return None
    if v.get("probability") is not None:
        return v["probability"]
    return None if v.get("status") == "not_assessable" else probability(v["status"], v["certainty"])


def load(dirs):
    rows, meta = {}, {}
    for d in dirs:
        for p in sorted((D / d).glob("*.json")):
            r = json.loads(p.read_text())
            rows[r["study_id"]] = {f: score(r["findings"].get(KM.get(f, KEY_OF[f]))) for f in FINDINGS}
            meta[r["study_id"]] = {f: ((r["findings"].get(KM.get(f, KEY_OF[f])) or {}).get("note") or "").split(":")[0].strip().lower()
                                   for f in FINDINGS} | {f + "|status": (r["findings"].get(KM.get(f, KEY_OF[f])) or {}).get("status") for f in FINDINGS}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=FINDINGS).astype(float), pd.DataFrame.from_dict(meta, orient="index")


def rd(name):
    d = json.load(open(REPO / f"data/image_labels/{name}.json"))
    return pd.DataFrame.from_dict({s: {f: (r["findings"].get(KM.get(f, KEY_OF[f])) or {}).get("probability") for f in FINDINGS}
                                   for s, r in d.items()}, orient="index").reindex(columns=FINDINGS).astype(float)


strict, smeta = load(["pilot_c1", "pilot_c2"])
fastp, _ = load(["pilot_c1_v1", "pilot_c2_v1"])
cxr, cxf, op12 = rd("codex_report_image_gold"), rd("codex_fluid_gold"), rd("opus_v4_gold")


def auc(y, s):
    m = s.notna() & y.notna()
    return roc_auc_score(y[m], s[m]) if m.sum() and 0 < y[m].sum() < m.sum() else np.nan


rng = np.random.default_rng(0)
out = [f"Opus strict reads on gold: {int(strict.index.isin(gold.index).sum())} / 58; fast pass {int(fastp.index.isin(gold.index).sum())}"]
for f in PANEL:
    y = gold[f]
    o = strict[f].reindex(gold.index)
    m = o.notna()
    ys, os_, ts, ps = y[m], o[m], text[KEY_OF[f]][m].fillna(0.15), pkg[f][m]
    out.append(f"\n== {f}: n {int(m.sum())}, n+ {int(ys.sum())}")
    out.append(f"   AUC: opus strict {auc(ys, os_):.3f} | text {auc(ys, ts):.3f} | pkg {auc(ys, ps):.3f}")
    for name, src in (("codex img+rep", cxr), ("codex fluid", cxf), ("opus img-only", op12), ("opus FAST pass", fastp)):
        s = src[f].reindex(ys.index)
        if s.notna().sum() >= 10:
            mm = s.notna()
            out.append(f"   on the {int(mm.sum())} with {name}: {name} {auc(ys[mm], s[mm]):.3f} vs opus strict {auc(ys[mm], os_[mm]):.3f} / text "
                       f"{auc(ys[mm], ts[mm]):.3f} / pkg {auc(ys[mm], ps[mm]):.3f}")
    bl = {"text + opus": z(lg(ts)) + z(lg(os_)), "pkg + 0.5 opus": z(lg(ps)) + 0.5 * z(lg(os_)), "pkg + 0.25 opus": z(lg(ps)) + 0.25 * z(lg(os_))}
    c = cxr[f].reindex(ys.index)
    if c.notna().sum() >= 10:
        mm = c.notna()
        bl["codex img+rep + opus (their n)"] = (z(lg(c[mm])) + z(lg(os_[mm]))).reindex(ys.index)
        bl["text + opus + codex img+rep (their n)"] = (z(lg(ts[mm])) + z(lg(os_[mm])) + z(lg(c[mm]))).reindex(ys.index)
    out.append("   blends (in-sample, INFO): " + " | ".join(f"{k} {auc(ys, v):.3f}" for k, v in bl.items()))
    for name, ref in (("text", ts), ("pkg", ps)):
        thr = 0.5
        dis = ys.index[(os_ >= .5) != (ref >= thr)]
        opus_right = int(sum((os_[s] >= .5) == bool(ys[s]) for s in dis))
        out.append(f"   disagree opus vs {name} at 0.5: {len(dis)} studies; gold sides with opus {opus_right} / {name} {len(dis) - opus_right}")
    tags = smeta[f].reindex(ys.index)
    acc = {t: (int((tags == t).sum()), float(((os_ >= .5) == (ys == 1))[tags == t].mean()) if (tags == t).any() else np.nan) for t in ("report", "images", "both")}
    out.append("   decided-by: " + " | ".join(f"{t} n {n} acc {a:.2f}" for t, (n, a) in acc.items()))
    st = smeta[f + "|status"].reindex(ys.index)
    out.append("   gold rate per Opus status: " + " | ".join(f"{s} {ys[st == s].mean():.2f} (n {int((st == s).sum())})" for s in st.dropna().unique()))
    b1, b2 = [], []
    idx = np.arange(len(ys))
    for _ in range(2000):
        i = rng.choice(idx, len(idx))
        yy = ys.iloc[i]
        if 0 < yy.sum() < len(yy):
            b1.append(roc_auc_score(yy, os_.iloc[i]) - roc_auc_score(yy, ts.iloc[i]))
            b2.append(roc_auc_score(yy, bl["pkg + 0.5 opus"].iloc[i]) - roc_auc_score(yy, ps.iloc[i]))
    lo = [auc(ys.drop(s), os_.drop(s)) - auc(ys.drop(s), ts.drop(s)) for s in ys.index]
    out.append(f"   AUC(opus) - AUC(text): {np.mean(b1):+.3f} [90 %: {np.percentile(b1, 5):+.3f}, {np.percentile(b1, 95):+.3f}], LOSO {min(lo):+.3f}..{max(lo):+.3f}; "
               f"AUC(pkg + 0.5 opus) - AUC(pkg): {np.mean(b2):+.3f} [90 %: {np.percentile(b2, 5):+.3f}, {np.percentile(b2, 95):+.3f}]")
(D / "eval_deep.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
