"""Training targets from text labels. make_targets(studies, cfg) -> targets (n, 12) and per-cell loss weights (n, 12).

Recipes (cfg keys in brackets):
  v1prob       v1 probability; not_mentioned -> cfg["nm"]
  status_rate  pooled gold-58 positive rate of the v1 status (uses gold: gold-58 is then not a clean check)
  fs_rate      gold-58 rate per finding x v1 status, shrunk to the pooled rate with weight fs_shrink (uses gold)
  hard         1 for v1 status meets / unsettled, else 0
  steven       steven v4 blend probability
  blend        mean in logit space of v1prob and steven v4
  ext:NAME     an external label set as the target (common.labels.external: steven_v4_blend, steven_v2, pilkwang_v2, lixin_sol56)
  mix:A+B+...  mean in logit space of several sets (v1prob and / or external names), e.g. mix:v1prob+steven_v4_blend+pilkwang_v2
Modifiers (applied in this order):
  v0_mix         logit mean with the same rate table applied to the v0 status (rate recipes) / to STATUS_RATE (others)
  syn_proxy      1: synovitis not_mentioned -> SYN_PROXY[effusion status]; 2: every study's synovitis target -> max(target, that) (gold-58: effusion meets 4/5 positive, unsettled 4/11, below 5/16, denied 0/5)
  eff_below      effusion status "below" -> this value (gold-58: 10/21 positive)
  cf_cap         contusion / fracture targets capped at this value (gold-58: "meets" only ~71% positive)
  sharpen        targets -> sigmoid(logit(y) / sharpen) (< 1 sharpens, > 1 softens)
  img_syn        CSV (StudyInstanceUID, Synovitis): image-read synovitis probability replaces the synovitis target where present
                 (blind Codex reads, Image Labeling session; work/image_labeling/export_labels.py)
  oa_grade       OA cells refined by the v1 depth / extent / severity attributes (oa_grade(); text only)
  severity       meets / unsettled / hedged cells with a known grade -> SEVERITY[grade] (effusion / Baker's size, OA depth, ACL grade;
                 gold reads count borderline findings as negative; mapping from the Ideas session)
  fo_value       not_mentioned at findings-only sites (training mention rate of ACL/MCL/menisci < 0.7, >= 10 studies) -> this value
  unc_weight     loss weight of cells with status hedged / unsettled / conflict / below
Per-group overrides (b_*): findings in b_findings ("B" = MCL, Lateral OA, Synovitis, Contusion, Fracture: no trusted positives,
text and gold disagree most; "A" = the 7 trusted findings) take their targets from make_targets with b_labels / b_sharpen /
b_v0_mix / b_nm instead of the main settings, and their loss weights are multiplied by b_weight.
status_targets(studies) -> (n, 12) index into STATUSES (auxiliary status head)
attribute_targets(studies) -> (n, len(ATTRS)) 0/1 sub-labels from v1 attribute fields (NaN = unknown): 1 when the finding is
present (meets / unsettled) with that attribute, 0 when present without it or when the finding is denied / normal.
  hard_trusted   trusted cells (common.labels.eval_cells) -> exact 0/1
  w_trusted      loss weight of trusted cells (others 1)
  nm_weight      loss weight of not_mentioned cells
  site_silence   not_mentioned -> site_silence_value where the study's site (>= site_silence_min training studies) mentions
                 that finding in >= site_silence_thr of its training reports
  syn_mask_rate  synovitis not_mentioned -> loss weight 0 at sites (>= site_silence_min training studies) that mention synovitis in
                 < syn_mask_rate of their training reports: there silence follows the site's reporting habit, not the image
                 (R2 OOF synovitis tracks the site's text rate, r 0.94 over 27 sites; Ideas session, site_habits.py)"""
from pathlib import Path

import numpy as np
import pandas as pd

from common.labels import eval_cells, external, gold_labels, probability_table, status_table, text_labels
from common.meta import study_table
from common.paths import FINDINGS, KEYS

STATUS_RATE = {"denied": 0.02, "structure_normal": 0.04, "excluded": 0.09, "general_normal": 0.18, "not_mentioned": 0.14, "below": 0.25,
               "hedged": 0.60, "conflict": 0.50, "unsettled": 0.63, "meets": 0.86}
SYN_PROXY = {"meets": 0.7, "unsettled": 0.36, "below": 0.31, "denied": 0.03}
SEVERITY = {"effusion": ("size", {"trace": 0.1, "small": 0.4, "mild": 0.45, "mild_to_moderate": 0.7, "moderate": 0.9, "large": 0.97}),
            "bakers_cyst": ("size", {"small": 0.5, "moderate": 0.9, "large": 0.97}),
            "acl": ("grade", {"degeneration": 0.15, "low_grade": 0.35, "low_to_partial": 0.5, "partial": 0.7, "high_grade": 0.9, "complete": 0.98})} | \
           {k: ("depth", {"superficial": 0.4, "partial": 0.7, "deep": 0.9, "full": 0.97}) for k in ("medial_oa", "lateral_oa", "pf_oa")}
RECIPES = {"sharp05": {"sharpen": 0.5}, "soft2": {"sharpen": 2.0}, "nm05": {"nm": 0.05}, "syn": {"syn_proxy": True},
           "fo": {"fo_value": 0.05}, "labstack": {"v0_mix": True, "syn_proxy": True, "cf_cap": 0.75, "nm": 0.05},
           "steven": {"labels": "steven"}, "steven2": {"labels": "ext:steven_v2"}, "pilk": {"labels": "ext:pilkwang_v2"},
           "lixin": {"labels": "ext:lixin_sol56"}, "knn": {"labels": "ext:knn_e5_v1_k10"}, "sev": {"severity": True},
           "hardw": {"labels": "hard", "unc_weight": 0.5, "nm_weight": 0.3}}    # named label recipes (train.py head_recipes)
STATUSES = list(STATUS_RATE)
UNCERTAIN = ("hedged", "unsettled", "conflict", "below")
GROUPS = {"B": ["MCL", "Lateral OA", "Synovitis", "Contusion", "Fracture"],
          "A": ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "PF OA", "Effusion", "Baker's"]}


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def site_mention_rate(studies, min_site=20):
    """Share of the given (training) studies whose report mentions each finding, per site; sites with < min_site studies -> 0."""
    st = status_table("v1").loc[studies]
    site = study_table().site.reindex(st.index)
    g = (st != "not_mentioned").groupby(site)
    return g.mean().where(g.size() >= min_site, 0.0)


# (name, finding key, field, test on the field value); the field value is a string or a list of strings
ATTRS = [("acl_complete", "acl", "grade", lambda v: v == "complete"), ("acl_partial", "acl", "grade", lambda v: v in ("partial", "low_grade")),
         ("mm_post", "medial_meniscus", "location", lambda v: "posterior_horn" in v), ("mm_body", "medial_meniscus", "location", lambda v: "body" in v),
         ("mm_ant", "medial_meniscus", "location", lambda v: "anterior_horn" in v),
         ("lm_post", "lateral_meniscus", "location", lambda v: "posterior_horn" in v), ("lm_body", "lateral_meniscus", "location", lambda v: "body" in v),
         ("lm_ant", "lateral_meniscus", "location", lambda v: "anterior_horn" in v),
         ("mm_displaced", "medial_meniscus", "displaced_fragment", lambda v: bool(v)), ("lm_displaced", "lateral_meniscus", "displaced_fragment", lambda v: bool(v)),
         ("moa_deep", "medial_oa", "depth", lambda v: v in ("full", "deep")), ("loa_deep", "lateral_oa", "depth", lambda v: v in ("full", "deep")),
         ("pfoa_deep", "pf_oa", "depth", lambda v: v in ("full", "deep")),
         ("moa_diffuse", "medial_oa", "extent", lambda v: v in ("diffuse", "multifocal")), ("loa_diffuse", "lateral_oa", "extent", lambda v: v in ("diffuse", "multifocal")),
         ("pfoa_diffuse", "pf_oa", "extent", lambda v: v in ("diffuse", "multifocal")),
         ("eff_large", "effusion", "size", lambda v: v in ("moderate", "large", "massive")), ("baker_large", "bakers_cyst", "size", lambda v: v in ("moderate", "large")),
         ("cont_trauma", "contusion", "cause", lambda v: v == "trauma"),
         ("cont_lat", "contusion", "bones", lambda v: any(b.startswith("lateral") for b in v)), ("cont_med", "contusion", "bones", lambda v: any(b.startswith("medial") for b in v)),
         ("cont_patella", "contusion", "bones", lambda v: "patella" in v)]
NEGATIVE = ("denied", "structure_normal")


def attribute_targets(studies):
    t = text_labels("v1")
    out = np.full((len(studies), len(ATTRS)), np.nan, np.float32)
    for i, sid in enumerate(studies):
        for j, (_, key, field, test) in enumerate(ATTRS):
            f = t[sid]["findings"][key]
            if f["status"] in NEGATIVE:
                out[i, j] = 0.0
            elif f["status"] in ("meets", "unsettled") and f.get(field) not in (None, [], ""):
                out[i, j] = float(test(f[field]))
    return out


def findings_only_sites(studies, max_rate=0.7, min_site=10):
    st = status_table("v1").loc[studies]
    site = study_table().site.reindex(st.index)
    g = (st[["acl", "mcl", "medial_meniscus", "lateral_meniscus"]] != "not_mentioned").mean(axis=1).groupby(site)
    r = g.mean()[g.size() >= min_site]
    return set(r[r < max_rate].index)


STATUS_GROUPS = {
    "": {s: i for i, s in enumerate(STATUSES)},                                             # the 10 v1 statuses
    "g4": {"meets": 0, "unsettled": 0, "hedged": 1, "conflict": 1, "below": 1, "denied": 2, "structure_normal": 2,
           "general_normal": 2, "excluded": 2, "not_mentioned": 3},                       # positive / uncertain / negative / silent
    "g3m": {"meets": 0, "unsettled": 0, "hedged": 1, "conflict": 1, "below": 1, "denied": 2, "structure_normal": 2,
            "general_normal": 2, "excluded": 2, "not_mentioned": -100},                   # same without silent cells (ignored)
}


def status_targets(studies, groups=""):
    """(n, 12) class index per finding (STATUS_GROUPS[groups]); -100 = ignored by the loss."""
    st = status_table("v1").loc[studies]
    return st[KEYS].replace(STATUS_GROUPS[groups]).to_numpy(np.int64)


def n_status_classes(groups=""):
    return len({v for v in STATUS_GROUPS[groups].values() if v >= 0})


def untrusted_studies():
    """Studies whose labels cannot be trusted for their images: the report names only the opposite knee and the side check does not
    explain it (39), or all 12 findings are silent (53: findings-only reports about other problems, truncated reports, one shoulder
    report). None is gold."""
    flipped = set(pd.read_csv(Path(__file__).resolve().parent / "series_side_final.csv", index_col=0).query("flip_final").StudyInstanceUID)
    other = {k for k, v in text_labels("v1").items() if v["study"]["report_side"] == "other"} - flipped
    st = status_table("v1")
    return other | set(st.index[(st == "not_mentioned").all(axis=1)])


def finding_status_rate(shrink=5.0):
    """status x finding table: (gold positives + shrink * pooled rate) / (gold count + shrink), pooled rate = STATUS_RATE."""
    g = gold_labels()
    st = status_table("v1").loc[g.index]
    out = {}
    for f, k in zip(FINDINGS, KEYS):
        d = pd.DataFrame({"s": st[k], "y": g[f]}).groupby("s").y.agg(["sum", "size"])
        out[f] = {s: (d.loc[s, "sum"] + shrink * r) / (d.loc[s, "size"] + shrink) if s in d.index else r for s, r in STATUS_RATE.items()}
    return pd.DataFrame(out)


def rate_targets(st, table):
    return np.stack([st[k].map(table[f]).to_numpy(float) for f, k in zip(FINDINGS, KEYS)], 1)


OA_KEYS = {"Medial OA": "medial_oa", "Lateral OA": "lateral_oa", "PF OA": "pf_oa"}
OA_SEV = {"mild": 0.4, "mild_to_moderate": 0.5, "moderate": 0.6, "moderate_to_severe": 0.7, "severe": 0.75}


def oa_grade(y, studies, j):
    """Text-only OA targets from the v1 attributes against the host rule (> 50% depth over ~1 cm or more; OA analysis A5 / A21):
    meets: depth full 0.95 / deep 0.8 / partial 0.6 (else keep), extent focal -0.05, diffuse / multifocal +0.03 (max 0.97);
    unsettled: severity word mild 0.4 ... severe 0.75, partial / superficial depth 0.45 (else keep);
    below: superficial depth or mild word 0.15, deep / full depth 0.35 (else keep)."""
    t = text_labels("v1")
    y = y.copy()
    for f, key in OA_KEYS.items():
        for i, sid in enumerate(studies):
            c = t[sid]["findings"][key]
            st, depth, ext, sev = c["status"], c.get("depth"), c.get("extent"), c.get("severity_word")
            v = y[i, j[f]]
            if st == "meets":
                v = {"full": 0.95, "deep": 0.8, "partial": 0.6}.get(depth, v)
                v = v - 0.05 if ext == "focal" else min(v + 0.03, 0.97) if ext in ("diffuse", "multifocal") else v
            elif st == "unsettled":
                v = OA_SEV.get(sev, 0.45 if depth in ("partial", "superficial") else v)
            elif st == "below":
                v = 0.15 if depth == "superficial" or sev == "mild" else 0.35 if depth in ("deep", "full") else v
            y[i, j[f]] = v
    return y


def make_targets(studies, cfg):
    if cfg.get("b_findings"):
        base = {k: v for k, v in cfg.items() if not k.startswith("b_")}
        y, w = make_targets(studies, base)
        alt = base | {k[2:]: v for k, v in cfg.items() if k in ("b_labels", "b_sharpen", "b_v0_mix", "b_nm") and v}
        yb, _ = make_targets(studies, alt)
        cols = [FINDINGS.index(f) for f in GROUPS[cfg["b_findings"]]]
        y[:, cols] = yb[:, cols]
        w[:, cols] *= cfg.get("b_weight", 1.0)
        return y, w
    st = status_table("v1").loc[studies]
    v1 = probability_table("v1").loc[studies].fillna(cfg.get("nm", 0.15)).to_numpy()
    recipe = cfg["labels"]
    table = pd.DataFrame({f: STATUS_RATE for f in FINDINGS})
    if recipe == "fs_rate":
        table = finding_status_rate(cfg.get("fs_shrink", 5.0))
    if recipe == "v1prob":
        y = v1
    elif recipe in ("status_rate", "fs_rate"):
        y = rate_targets(st, table)
    elif recipe == "hard":
        y = st.isin(("meets", "unsettled")).to_numpy(float)
    elif recipe == "steven":
        y = external("steven_v4_blend").loc[studies, FINDINGS].to_numpy()
    elif recipe == "blend":
        s = external("steven_v4_blend").loc[studies, FINDINGS].to_numpy()
        y = 1 / (1 + np.exp(-(logit(v1) + logit(s)) / 2))
    elif recipe.startswith(("ext:", "mix:")):                # missing external cells fall back to v1prob
        names = recipe[4:].split("+")
        sets = [v1 if n == "v1prob" else external(n).reindex(studies)[FINDINGS].to_numpy() for n in names]
        sets = [np.where(np.isnan(x), v1, x) for x in sets]
        y = 1 / (1 + np.exp(-np.mean([logit(x) for x in sets], 0)))
    else:
        raise ValueError(recipe)
    if cfg.get("v0_mix"):
        y0 = rate_targets(status_table("v0").reindex(studies).fillna(st), table)
        y = 1 / (1 + np.exp(-(logit(y) + logit(y0)) / 2))
    if cfg.get("sharpen"):
        y = 1 / (1 + np.exp(-logit(y) / cfg["sharpen"]))
    y = y.astype(np.float32).copy()
    j = {f: i for i, f in enumerate(FINDINGS)}
    if cfg.get("img_syn"):
        im = pd.read_csv(cfg["img_syn"]).set_index("StudyInstanceUID").Synovitis.reindex(studies).to_numpy()
        ok = ~np.isnan(im)
        y[ok, j["Synovitis"]] = im[ok]
    if cfg.get("oa_grade"):
        y = oa_grade(y, studies, j)
    if cfg.get("severity"):
        t = text_labels("v1")
        for key, (field, table) in SEVERITY.items():
            col = FINDINGS[KEYS.index(key)]
            for i, sid in enumerate(studies):
                c = t[sid]["findings"][key]
                if c["status"] in ("meets", "unsettled", "hedged") and c.get(field) in table:
                    y[i, j[col]] = table[c[field]]
    if cfg.get("syn_proxy"):
        v = st.effusion.map(SYN_PROXY).to_numpy(float)
        m = ~np.isnan(v)
        if cfg["syn_proxy"] == 1:
            m &= (st.synovitis == "not_mentioned").to_numpy()
            y[m, j["Synovitis"]] = v[m]
        else:
            y[m, j["Synovitis"]] = np.maximum(y[m, j["Synovitis"]], v[m])
    if cfg.get("eff_below"):
        y[(st.effusion == "below").to_numpy(), j["Effusion"]] = cfg["eff_below"]
    if cfg.get("cf_cap"):
        for f in ("Contusion", "Fracture"):
            y[:, j[f]] = np.minimum(y[:, j[f]], cfg["cf_cap"])
    w = np.ones_like(y)
    nm = (st == "not_mentioned").to_numpy()
    if cfg.get("site_silence_thr"):
        rate = site_mention_rate(studies, cfg.get("site_silence_min", 20))
        site = study_table().site.reindex(studies)
        r = rate.reindex(site).to_numpy()
        silent_neg = nm & (r >= cfg["site_silence_thr"])
        y[silent_neg] = cfg.get("site_silence_value", 0.03)
    if cfg.get("fo_value"):
        fo = study_table().site.reindex(studies).isin(findings_only_sites(studies)).to_numpy()
        y[nm & fo[:, None]] = cfg["fo_value"]
    w[nm] = cfg.get("nm_weight", 1.0)
    w[st.isin(UNCERTAIN).to_numpy()] = cfg.get("unc_weight", 1.0)
    if cfg.get("syn_mask_rate"):
        site = study_table().site.reindex(studies).to_numpy()
        g = (st.synovitis != "not_mentioned").groupby(site)
        rate = g.mean().where(g.size() >= cfg.get("site_silence_min", 20))          # NaN for small sites: keep their cells
        low = pd.Series(site).map(rate).to_numpy() < cfg["syn_mask_rate"]
        w[low & nm[:, j["Synovitis"]], j["Synovitis"]] = 0
    if cfg.get("clean_labels"):
        w[pd.Index(studies).isin(untrusted_studies())] = 0
    if cfg.get("hard_trusted") or cfg.get("w_trusted", 1.0) != 1.0:
        c = eval_cells()
        c = c[(c.verdict == "trusted") & c.StudyInstanceUID.isin(set(studies))]
        ri = pd.Index(studies).get_indexer(c.StudyInstanceUID)
        ci = np.array([FINDINGS.index(f) for f in c.finding])
        if cfg.get("hard_trusted"):
            y[ri, ci] = c.label.to_numpy()
        w[ri, ci] = cfg.get("w_trusted", 1.0)
    return y, w
