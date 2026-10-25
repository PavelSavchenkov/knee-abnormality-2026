"""PL variant JSONs for Experts' exact gold-58 package runs (local_ref.py --ids gold), from the FROZEN store only. OAI: no. CPU, read-only.
Twins: PL seed P<200+k> <-> the BASE candidate's k-th plain seed (build.VARIANTS[--base], default safe3w9; form = the base's); a pair counts only if BOTH are frozen.
APPLES TO APPLES (USER 00:0x), same seed indices per recipe:
  <tag>_plain_m : each recipe's plain seeds that have a frozen twin (recipes without one: their frozen final seeds)
  <tag>_pl_m    : those twins (recipes without one: the same plain seeds as plain_m)
  <tag>_avg_m   : both lists (= more models, by design)
  <tag>_plain   : every frozen seed of the final (context: the final as far as frozen)
  <tag>_half_m  : per recipe the first ceil(k/2) twinned pairs plain, the rest PL (count-matched to plain_m = a submittable plain + PL mix)
  <tag>_seedrow : plain_m with each twinned plain seed swapped for an UNUSED frozen plain seed of the same recipe (final seeds first, then
                  any extra frozen plain seed outside the final, e.g. null-only seeds 108+; kept where no spare exists, listed) = the seed-swap
                  null: seedrow - plain_m sizes what pl_m - plain_m would show from new seeds alone
Form = the final's (frac on, effc off, sxm on, fc_off on). Writes variants/<name>.json + prints the twin table.
  conda run -n rsna python work/pl_alldata_2026-10-21/make_variants.py --tag d1"""
import argparse
import json
import sys
from pathlib import Path

D = Path(__file__).parent
REPO = D.parents[1]
FA = REPO / "submissions/oct21_final_alldata"
sys.path.insert(0, str(FA))
import build as B  # noqa: E402

DEAD = {"b_v3", "b_lr3", "b_nf", "ef3m", "e3bn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="delta tag, e.g. d1 (names pl_<tag>_*)")
    ap.add_argument("--base", default="safe3w9", help="build.VARIANTS entry the twins pair with (Master 08:0x: the candidate is safe3w9)")
    a = ap.parse_args()
    idx = json.load(open(FA / "reg_final/store_index.json"))
    frozen = {}
    for v in idx.values():
        if "seed" in v:
            frozen.setdefault(v["recipe"], set()).add(str(v["seed"]))
    out = {"plain_m": {}, "pl_m": {}, "avg_m": {}, "plain": {}, "seedrow": {}, "half_m": {}}
    nospare = []
    rows = []
    base = B.VARIANTS[a.base]
    FORM = base["form"]
    dead = DEAD | (set() if FORM["sxm"] else {"sxm"})
    for r, labels in base["seeds"].items():
        if r in dead:
            continue
        fz = frozen.get(r, set())
        plain = [s for s in labels if str(s) in fz]
        pairs = [(s, f"P{200 + k}") for k, s in enumerate(labels, 1) if str(s) in fz and f"P{200 + k}" in fz]
        pp, pq = [s for s, _ in pairs], [q for _, q in pairs]
        out["plain"][r] = plain
        out["plain_m"][r] = pp if pairs else plain
        out["pl_m"][r] = pq if pairs else plain
        out["avg_m"][r] = pp + pq if pairs else plain
        h = (len(pairs) + 1) // 2                        # HALF (count-matched plain + PL mix): first ceil(k/2) pairs plain, the rest their PL twins
        out["half_m"][r] = (pp[:h] + pq[h:]) if pairs else plain
        extra = sorted((int(x) if x.isdigit() else x for x in fz if not x.startswith("P") and x not in {str(y) for y in labels}),
                       key=lambda x: (isinstance(x, str), str(x)))                    # frozen plain seeds OUTSIDE the final (null spares)
        spare = [s for s in plain if s not in pp] + extra
        if pairs:
            sw = [spare[i] if i < len(spare) else s for i, s in enumerate(pp)]
            if len(spare) < len(pp):
                nospare.append(f"{r} ({len(pp) - len(spare)} of {len(pp)} kept)")
            out["seedrow"][r] = sw
        else:
            out["seedrow"][r] = plain
        rows.append(f"  {r:12s} final {len(labels)}  frozen plain {len(plain)}  frozen twins {len(pairs)}  {pairs}")
        assert plain, f"{r}: no frozen plain seed"
    (D / "variants").mkdir(exist_ok=True)
    for k, seeds in out.items():
        name = f"pl_{a.tag}_{k}"
        json.dump({"form": FORM, "seeds": seeds}, open(D / "variants" / f"{name}.json", "w"), indent=1)
    n = {k: sum(len(v) for v in s.values()) for k, s in out.items()}
    print(f"variants pl_{a.tag}_{{plain_m, pl_m, avg_m, plain, seedrow, half_m}} written to {D / 'variants'}; models per variant {n}")
    print("\n".join(rows))
    print("seedrow: twinned seeds without a spare plain seed (kept as in plain_m -> the null understates their seed noise):", nospare or "none")


if __name__ == "__main__":
    main()
