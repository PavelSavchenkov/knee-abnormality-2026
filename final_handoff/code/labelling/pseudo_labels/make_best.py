"""Best-estimate PL packages (USER 08:5x: "i just need max estimated AUC"; not count-matched). OAI: no. CPU, read-only.
Per GROUP a choice: plain (safe3w9's seeds) | pl (their frozen P20k twins) | both (plain + twins = the group's plain + PL average, more copies).
A twin missing from the frozen store fails loudly (no silent plain fallback). Writes variants/best_<name>.json in Experts' format.
  conda run -n rsna python work/pl_alldata_2026-10-21/make_best.py [--dry DIR]"""
import argparse
import json
import sys
from pathlib import Path

D = Path(__file__).parent
REPO = D.parents[1]
FA = REPO / "submissions/oct21_final_alldata"
sys.path[:0] = [str(FA), str(REPO / "work/alldata_final_2026-10-21")]
import allocate as A  # noqa: E402  (rc(r) = Kaggle min / copy at scale, kaggle_costs.json overrides)
import build as B  # noqa: E402

SCALE = 0.695                                  # Master 09:0x: Submission Timing's rescale of the at-scale per-copy costs

BASE = "safe3w9"
GROUPS = {"bone": ["b3c_rep", "b3l_rep", "b3n_rep", "bfs_rep"], "loax": ["oasep", "oasep_nf", "silsep"], "oa": ["oa_v2cor", "oa_v2cor_nf"],
          "lig": ["z1_acl", "w5b_mcl", "b3g", "sd1b", "bsd_nf"], "eff": ["ef3", "ef3fc", "ef3nf", "ef3nffc"],
          "syn": ["cs_fs", "cs_fc125", "s_fs_nf", "s_fc125_nf", "m_prob05", "n_prob05"], "gen": ["sub_b0", "nfnet", "r50"]}
# dp4 exact attribution (Experts 07:3x) + priors; 'gen' is decided by the dp5 read
CANDIDATES = {"b1": {"bone": "plain", "loax": "plain", "oa": "pl", "lig": "pl", "eff": "pl", "syn": "pl", "gen": "plain"},
              "b2": {"bone": "plain", "loax": "plain", "oa": "pl", "lig": "pl", "eff": "pl", "syn": "pl", "gen": "pl"},
              "b3": {"bone": "plain", "loax": "plain", "oa": "both", "lig": "both", "eff": "pl", "syn": "pl", "gen": "plain"},
              "b4": {"bone": "plain", "loax": "plain", "oa": "both", "lig": "both", "eff": "both", "syn": "both", "gen": "plain"},
              # b5: gold vs stability conflict on the generalists (Baker's -0.011 vs ratio 0.33-0.78) -> hedge 'both'; w5b 'both' (plain = the
              # noisiest recipe, PL 2.4x steadier); per-recipe keys override the group
              "b5": {"bone": "plain", "loax": "plain", "oa": "pl", "lig": "pl", "w5b_mcl": "both", "eff": "pl", "syn": "pl", "gen": "both"},
              # Master 09:1x prior: 'both' hedges = expected-positive (noise cut + diversity), best gain per minute; within the +55-min staging cap.
              # b6: hedge everything cheap that is not gold-negative: OA + z1 / w5b / sd1b 'both' (b3g / bsd_nf PL: dear), gen 'both'
              "b6": {"bone": "plain", "loax": "plain", "oa": "both", "lig": "pl", "z1_acl": "both", "w5b_mcl": "both", "sd1b": "both",
                     "eff": "pl", "syn": "pl", "gen": "both"},
              # b7: if the dp5 xgen read says generalist PL is net negative: gen plain, and the freed minutes go to EF3 'both'
              "b7": {"bone": "plain", "loax": "plain", "oa": "both", "lig": "pl", "z1_acl": "both", "w5b_mcl": "both", "sd1b": "both",
                     "eff": "both", "syn": "pl", "gen": "plain"}}
# Names lowercase (Experts' builder). Master 08:3x P5 group-level fallback b6a: b6 + inv1_A's plain adds (PLAIN_N), their twins too where 'both'
CANDIDATES |= {"b6a": CANDIDATES["b6"], "b6c": CANDIDATES["b6"], "b6ca": CANDIDATES["b6"]}
PLAIN_N = {k: {"bfs_rep": 5, "oa_v2cor": 5, "sd1b": 5} for k in ("b6a", "b6ca")}   # candidate -> recipe -> plain seed count (build.plan_seeds)
# P4 / P5 (Master 08:3x): b6's model set with OA experts + generalists routed PER COLUMN (Experts' format: "seeds" = plain labels, "pl" = PL
# labels, "routes" = {recipe: {column: plain | pl | both}}, unlisted = plain). Routable (Master 08:5x, from Experts' code): oa_v2cor(_nf)
# MOA / PF per column, MM = LM one route; sub_b0 ACL / MM / LM / MOA / LOA / PF / Fracture (MCL moot: only r50's counts); m_prob05 Synovitis only (independent, but
# ROUTED with its plain seeds: a routed sub_b0's plain main set needs an image-head model, else forward() takes an untested branch, Experts
# 08:5x; +3 copies); nfnet / r50 ACL / MM / LM / MOA / LOA / PF / Fracture (+ MCL r50). Effusion / Baker's (window
# fusion) are plain-only in routed variants. Routes by prior (8b 08:27 to Master): OA PF PL, MOA plain, MM / LM both; generalists MOA plain,
# the rest PL.
GEN = {"ACL": "pl", "Medial Meniscus": "pl", "Lateral Meniscus": "pl", "Medial OA": "plain", "Lateral OA": "pl", "PF OA": "pl", "Fracture": "pl"}
OA = {"Medial OA": "plain", "PF OA": "pl", "Medial Meniscus": "both", "Lateral Meniscus": "both"}
ROUTE = {"oa_v2cor": OA, "oa_v2cor_nf": OA, "sub_b0": GEN, "m_prob05": {"Synovitis": "pl"}, "nfnet": GEN, "r50": GEN | {"MCL": "pl"}}
WINDOW = {"b6c": "both", "b6ca": "both"}   # Master 09:0x + Experts' format: window passes (Effusion / Baker's) take plain + PL files, as b6 does
ROUTES = {"b6c": ROUTE, "b6ca": ROUTE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", help="write EVERY candidate (unfrozen twins too) to this dir, for build.py --check-variant")
    a = ap.parse_args()
    base = B.VARIANTS[BASE]
    form = base["form"]
    idx = json.load(open(FA / "reg_final/store_index.json"))
    frozen = {}
    for v in idx.values():
        if "seed" in v:
            frozen.setdefault(v["recipe"], set()).add(str(v["seed"]))
    dead = {"b_v3", "b_lr3", "b_nf", "ef3m", "e3bn"} | (set() if form["sxm"] else {"sxm"})
    grp = {r: g for g, rs in GROUPS.items() for r in rs}
    live = [r for r in base["seeds"] if r not in dead]
    assert set(live) == set(grp), f"group map != {BASE} recipes: {set(live) ^ set(grp)}"
    assert all(rt["Medial Meniscus"] == rt["Lateral Meniscus"] for r, rt in ROUTE.items() if r.startswith("oa_")), "OA: MM = LM one route"
    for name, choice in CANDIDATES.items():
        assert name == name.lower(), name
        seeds, pl, routes, missing = {}, {}, {}, []
        for r in live:
            plain = list(base["seeds"][r])
            if r in PLAIN_N.get(name, {}):
                plain = B.plan_seeds(r, PLAIN_N[name][r])
                missing += [f"{r}:{s}" for s in plain if str(s) not in frozen.get(r, set())]
            twins = [f"P{200 + k}" for k in range(1, len(plain) + 1)]   # P20k = twin of the k-th plan seed; 'both' stays 1:1 (Master 08:4x)
            if r in ROUTES.get(name, {}):                                # routed: plain + PL both loaded, mixed per column
                seeds[r], pl[r], routes[r] = plain, twins, ROUTES[name][r]
                missing += [f"{r}:{t}" for t in twins if t not in frozen.get(r, set())]
                continue
            ch = choice.get(r, choice[grp[r]])
            missing += [f"{r}:{t}" for t in twins if t not in frozen.get(r, set())] if ch != "plain" else []
            seeds[r] = {"plain": plain, "pl": twins, "both": plain + twins}[ch]
        cnt = {r: len(seeds[r]) + len(pl.get(r, [])) for r in live}
        n = sum(cnt.values())
        extra = sum((cnt[r] - len(base["seeds"][r])) * A.rc(r) for r in live) * SCALE
        if missing and not a.dry:
            print(f"best_{name}: {n} models, extra ~{extra:.0f} min (x{SCALE}); NOT written, twins not frozen yet: {len(missing)}")
            continue
        out = {"form": form, "seeds": seeds} | ({"pl": pl, "routes": routes} if routes else {}) | ({"window": WINDOW[name]} if name in WINDOW else {})
        f = (Path(a.dry) if a.dry else D / "variants") / f"best_{name}.json"
        if f.exists() and json.load(open(f)) == out:             # unchanged: keep the file (and its mtime) for Experts' waiter
            continue
        json.dump(out, open(f, "w"), indent=1)
        print(f"best_{name}: {n} models ({n - sum(len(base['seeds'][r]) for r in live):+d} vs {BASE}), extra ~{extra:.0f} min (x{SCALE}) {choice}"
              + (f" routed {sorted(routes)}" if routes else ""))

if __name__ == "__main__":
    main()
