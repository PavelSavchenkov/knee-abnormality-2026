"""Monitor of the 6 Phase 2 Claude chats (p2_s1..s6; USER 10-20: "keep monitoring closely if these sessions label reasonably and how we
expect"). Per chat: saved / 250, list-order prefix, validate.py problems, pace. Pooled: stratum mix, per-finding positive rate, certainty,
note compliance (native view named + source tag), agreement with clear text v1 statements (meets / denied, normal) and the same metrics
on the pilot's strict reads as the reference. No gold anywhere (labeller blindness). Prints ALERT lines for anything off.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/p2s_monitor.py"""
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import status_table
from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
spec = importlib.util.spec_from_file_location("validate", D / "_prompts/validate.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)
import sys  # noqa: E402
sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability  # noqa: E402

F = ("contusion", "fracture", "pf_oa")
POS, NEG = {"meets"}, {"denied", "structure_normal", "general_normal"}
SERIES = re.compile(r"\bs\d+\b[^.;]*?#\s*\d+", re.I)   # "s1 AX PD FS #15-23" or "s4 #4-12"
TAG = re.compile(r"\b(images|both|report)\b(\s*:|\W*$)")   # pilot style "both: report ..." or a trailing tag
key = pd.read_csv(D / "_key/p2s.csv").set_index("index")
text = status_table("v1")


def load(folder):
    rows = []
    for p in folder.glob("*.json"):
        x = json.load(open(p))
        r = {"index": x["index"], "study_id": x["study_id"], "mtime": p.stat().st_mtime, "path": p}
        for f in F:
            v = x["findings"].get(f, {})
            r[f"{f}_p"] = None if v.get("status") in (None, "not_assessable") else probability(v["status"], v["certainty"])
            r[f"{f}_cert"] = v.get("certainty")
            note = v.get("note", "").strip()
            r[f"{f}_ok"] = bool(SERIES.search(note)) and bool(TAG.search(note))
        rows.append(r)
    return pd.DataFrame(rows)


def quality(df):
    out = {}
    for f in F:
        p = df[f"{f}_p"].astype(float)
        t = text.reindex(df.study_id)[f].to_numpy()
        clear = np.isin(t, list(POS | NEG)) & p.notna().to_numpy()
        y = np.isin(t[clear], list(POS)).astype(int)
        agree = ((p.to_numpy()[clear] >= .5).astype(int) == y).mean() if clear.any() else np.nan
        auc = roc_auc_score(y, p.to_numpy()[clear]) if 0 < y.sum() < len(y) else np.nan
        out[f] = dict(pos=(p >= .5).mean(), na=p.isna().mean(), high=(df[f"{f}_cert"] == "high").mean(), notes_ok=df[f"{f}_ok"].mean(),
                      n_clear=int(clear.sum()), agree=agree, auc=auc)
    return out


now = time.time()
alerts, allp = [], []
print(f"p2s monitor {time.strftime('%H:%M')}")
for s in list(range(1, 8)) + [f"7_h{h}" for h in (1, 2, 4, 6, 5)]:   # s7 = fracture-vague continuation (10-20 ~11:00); 7_hN = its chunks on finished chats (16:2x)
    lst = [int(i) for i in (D / f"_batches/p2_s{s}.txt").read_text().split()]
    folder = D / f"p2_s{s}"
    df = load(folder) if folder.exists() else pd.DataFrame()
    if df.empty:
        print(f"  s{s}: 0 / {len(lst)}")
        continue
    df = df.sort_values("mtime")
    rest = next((f for f in (D / f"_batches/p2_s{s}_part.txt", D / f"_batches/p2_s{s}_rest.txt") if f.exists()), D / "none")   # 16:2x: s7 keeps the head part; 10-20 16:09 (s1-s6) / 16:20 (s7): remaining low-value entries dropped -> target = saved + rest
    n_target = len(set(df["index"]) | {int(i) for i in rest.read_text().split()}) if rest.exists() else len(lst)
    pos = df["index"].map({i: k for k, i in enumerate(lst)})
    prefix = pos.notna().all()   # 10-20 16:09: s1-s6 switched to vague-only _rest lists -> membership, not prefix
    bad = {r.path.name: m for r in df.itertuples() if (m := validate.check_file(r.path, r.index))}
    gaps = np.diff(df.mtime.to_numpy()) / 60
    last_h = (df.mtime > now - 3600).sum()
    idle = (now - df.mtime.max()) / 60
    fr10 = df.fracture_ok.iloc[-10:].mean()   # 10-20 11:0x reminder to s1 / s2 / s4 / s6: native T1 crop + slice range for fracture
    print(f"  s{s}: {len(df)} / {n_target} | last save {idle:.0f} min ago | {last_h} in the last hour | frac notes OK (last 10) {fr10:.1f} | median {np.median(gaps):.1f} min / study"
          if len(gaps) else f"  s{s}: {len(df)} / {n_target} | last save {idle:.0f} min ago")
    if not prefix:
        alerts.append(f"s{s}: a saved study is not in its list (foreign index)")
    if bad:
        alerts.append(f"s{s}: {len(bad)} file(s) fail validate.py, e.g. {next(iter(bad))}: {next(iter(bad.values()))[:2]}")
    if len(gaps) >= 3 and np.median(gaps[-5:]) < 0.5:   # the pilot strict reads ran at 0.9 / 2.1 min median
        alerts.append(f"s{s}: last saves < 0.5 min / study (pilot strict: 0.9-2.1)")
    if idle > 20 and len(df) < n_target:
        alerts.append(f"s{s}: no save for {idle:.0f} min (stopped / stuck?)")
    allp.append(df)
ids = [json.load(open(f))["study_id"] for f in D.glob("p2_s*/*.json")]   # build_tgt_p2.py asserts one read per study
if len(ids) != len(set(ids)):
    alerts.append(f"{len(ids) - len(set(ids))} study read twice across p2_s*/ folders (build_tgt_p2.py will fail)")
if allp:
    a = pd.concat(allp)
    k6 = key.reindex(a["index"])
    mix = k6[k6.session <= 6].stratum.value_counts(normalize=True).round(2).to_dict()
    print(f"  pooled {len(a)} saved; strata {mix} (s1-s6; target vague .6 / silent_mid .2 / random .2); s7 fracture-vague: {int((k6.session == 7).sum())}")
    ref = quality(pd.concat([load(D / "pilot_c1"), load(D / "pilot_c2")]))
    q = quality(a)
    for f in F:
        r, x = ref[f], q[f]
        print(f"  {f:9s} pos {x['pos']:.2f} (pilot, other mix {r['pos']:.2f}) | n/a {x['na']:.2f} | high cert {x['high']:.2f} (pilot {r['high']:.2f}) | notes OK {x['notes_ok']:.2f} "
              f"(pilot {r['notes_ok']:.2f}) | vs clear text n {x['n_clear']}: agree {x['agree']:.2f} AUC {x['auc']:.2f} (pilot agree {r['agree']:.2f} AUC {r['auc']:.2f})")
        if len(a) >= 20:
            if x["notes_ok"] < .8:
                alerts.append(f"{f}: only {x['notes_ok']:.0%} of notes name a native view + end with a source tag")
            if x["na"] > .1:
                alerts.append(f"{f}: {x['na']:.0%} not_assessable")
            if x["n_clear"] >= 15 and x["agree"] < r["agree"] - .15:
                alerts.append(f"{f}: agreement with clear text {x['agree']:.2f} vs pilot {r['agree']:.2f}")
print("\n".join(f"ALERT {m}" for m in alerts) if alerts else "no alerts")
