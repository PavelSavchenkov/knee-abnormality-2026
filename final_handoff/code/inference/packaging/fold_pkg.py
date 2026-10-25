"""The merged FOLD-avg package in the FINAL form (USER 10-21 22:5x via Master: "the same pipeline we run inside kaggle -- best pipeline for each
class, fold average"; OAI: no; CPU build, nothing uploaded): build.assemble(FINAL_FORM) = v2 + Contusion all-4 + sxm w 0.3 + Fracture v3c_fs
+ EF3 Effusion, fc_off (eff_c OFF) with every member's FOLD models, linked from the kits' work dirs (same-name files across kits asserted
byte-identical) + pkg_code/ensemble.py + the all-4 code + main_sd/ -> _work/fold_<name>/, ready for local_ref.py --pkg-dir.
  conda run -n rsna python submissions/oct21_final_alldata/fold_pkg.py [NAME] (default final_form)"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402

KITS = ("all4", "ef3", "effc", "sxm", "frac")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "final_form"
    form = B.FINAL_FORM
    pkg4, cfg, syn, kits = B.assemble(argparse.Namespace(**form))
    out = HERE / "_work" / f"fold_{name}"
    shutil.rmtree(out, ignore_errors=True)
    (out / "main_sd").mkdir(parents=True)
    src = {}
    for k in KITS:
        for p in (HERE / "_work" / k).iterdir():
            if not p.is_symlink() or p.suffix == ".py":
                continue
            t = p.resolve()
            s = src.setdefault(p.name, t)
            assert s == t or B.md5(s) == B.md5(t), f"{p.name}: kits link different files"
    for n, t in src.items():
        os.symlink(t, out / n)
    for p in (pkg4 / "main_sd").iterdir():
        os.symlink(p.resolve(), out / "main_sd" / p.name)
    for p in pkg4.iterdir():
        if p.suffix in (".py", ".csv") and p.name != "ensemble.py" and p.is_file():
            shutil.copyfile(p.resolve(), out / p.name)
    shutil.copyfile(HERE / "pkg_code/ensemble.py", out / "ensemble.py")
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    json.dump(syn, open(out / cfg["syn"]["members"], "w"), indent=1)
    need = [f for m in syn.values() for f in m["files"]] + [f for m in cfg["oa"]["members"].values() for f in m["files"]]
    need += [f for m in cfg["fc"]["members"].values() for f in m["files"]] + cfg["weights"] + cfg["win"]["main"] + cfg["cont_sxm"]["files"]
    need += [c for cs in list(cfg["basin"].values()) + list(cfg["basin_syn"].values()) + [cfg["mcl_sd"]["nfnet"]] for c in cs]
    missing = [f for f in need if not (out / f).exists()]
    assert not missing, missing
    assert not cfg.get("eff_main") and cfg.get("cont_sxm") and cfg.get("frac_allmem") and cfg.get("fc_shipped_off"), "FINAL_FORM switches"
    print(f"fold package {out.name}: form {form}; {len(need)} referenced weights present; RUNTIME {B.stage_counts(cfg, out)}")


if __name__ == "__main__":
    main()
