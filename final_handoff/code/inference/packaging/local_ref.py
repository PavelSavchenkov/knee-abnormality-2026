"""LOCAL reference of a variant's Kaggle commit check (Master 10-21 22:2x: the classic gate for THE final; OAI: no; local GPU in Master's
window). The kernel's WORK layout rebuilt locally (pkg_<variant>/ union mirror + main_sd/ from main_sd_config.json), the package's own
ensemble.predict on the SAME first 4 edge check studies (PKG/check_studies.csv, group edge, as the kernel's --quick 4 CHK_IDS), the
kernel's DEC; then Kaggle check_pred / check_pred_base / check_parts vs local: max |dp| per column / part. PASS = every column <= 1e-3
and every part <= 1e-3 (fp16 GPU, Kaggle T4 vs local 5090; the kits' Kaggle-mode tolerance), 0 failed studies, all finite.
Waits until >= --need-gb of GPU memory is free; caps itself at --gpu-gb.
  conda run -n rsna python submissions/oct21_final_alldata/local_ref.py VARIANT KAGGLE_COMMIT_DIR [--gpu-gb 11] [--need-gb 12]"""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def free_gb():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0]) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("commit_dir")
    ap.add_argument("--gpu-gb", type=float, default=11)
    ap.add_argument("--need-gb", type=float, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ids", default="edge4", choices=["edge4", "gold", "check"], help="gold = the gold-58 check studies, check = all check studies "
                    "(edge + gold + random): predictions only (no Kaggle compare)")
    ap.add_argument("--root", default=str(REPO / "data/original_data/train_series"), help="DICOM root (e.g. a dicom_stage copy for the staging gate)")
    ap.add_argument("--out-suffix", default="", help="appended to the output csv name (e.g. _staged)")
    ap.add_argument("--no-arm-wait", action="store_true", help="start on free memory alone (Master: OK next to a running oct14 arm)")
    ap.add_argument("--readahead", action="store_true", help="wrap Basin's bp.predict with pkg_code/dicom_prefetch.py exactly as the "
                    "--readahead kernel's FAST worker does (local gate: on vs off must be bit-identical)")
    ap.add_argument("--pkg-dir", default="", help="run a ready package dir (e.g. _work/fold_final_form from fold_pkg.py; main_sd/ inside); DEC = v2's kernel")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import build as B
    if a.pkg_dir:                                        # a ready package dir (fold form); the label = its dir name
        work = Path(a.pkg_dir).resolve()
        t_ = work.name
    else:
        t_ = B.tag(a.variant)
        pk = HERE / f"pkg_{t_}"
        work = HERE / "_work" / f"localref_{t_}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        cfg = json.load(open(pk / "config.json"))
        for f in pk.iterdir():                           # the kernel's WORK: code copied, the rest symlinked, + main_sd/
            (shutil.copy if f.name.endswith(".py") else os.symlink)(os.path.realpath(f), work / f.name)
        sd = work / cfg["mcl_sd"]["main_dir"]
        sd.mkdir()
        sdc = json.load(open(pk / "main_sd_config.json"))
        json.dump(sdc, open(sd / "config.json", "w"))
        for w in sdc["weights"]:
            os.symlink(os.path.realpath(pk / w), sd / w)
        if (pk / "main_pl_config.json").exists():         # oct22 routes: the PL main set's dir, as the kernel builds it
            mp = work / cfg["main_pl"]["main_dir"]
            mp.mkdir()
            mpc = json.load(open(pk / "main_pl_config.json"))
            json.dump(mpc, open(mp / "config.json", "w"))
            for w in mpc["weights"]:
                os.symlink(os.path.realpath(pk / w), mp / w)
    t0 = time.time()                                     # 22:18 lesson: a momentary dip in the neighbours' memory read 15.6 GB free and the run OOMed
    arms = lambda: subprocess.run(["pgrep", "-fc", r"^python experiments/v2/oct14_all_data/scripts/launch[.]py"], capture_output=True,  # noqa: E731
                                  text=True).stdout.strip()
    ok_n = 0
    while ok_n < 3:                                      # no oct14 training arm AND >= need GB free on 3 checks in a row (15 s apart)
        ok_n = ok_n + 1 if ((a.no_arm_wait or arms() == "0") and free_gb() >= a.need_gb) else 0
        time.sleep(15)
    print(f"{time.strftime('%H:%M:%S')} GPU free {free_gb():.1f} GB (waited {time.time() - t0:.0f} s); running {a.pkg_dir or a.variant} on {'4 edge' if a.ids == 'edge4' else 'the gold-58'} studies", flush=True)
    sys.path[:0] = [str(work), str(REPO)]
    import torch
    import ensemble
    assert Path(ensemble.__file__).parent == work, ensemble.__file__
    torch.cuda.set_per_process_memory_fraction(a.gpu_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    chk = pd.read_csv(work / "check_studies.csv")
    ids = {"edge4": chk.StudyInstanceUID[chk.group == "edge"].tolist()[:4], "gold": chk.StudyInstanceUID[chk.group == "gold"].tolist(),
           "check": chk.StudyInstanceUID.tolist()}[a.ids]
    series = pd.read_csv(REPO / "data/original_data/train_series.csv")
    t = time.time()
    if a.readahead:                                      # the same start / hook / stop as the kernel's FAST worker basin() closure
        sys.path.insert(0, str(HERE / "pkg_code"))
        import dicom_prefetch as DP
        bp, orig = ensemble.bp, ensemble.bp.predict

        def predict_ra(series_, root_, *args, **kw):
            pf = DP.start(root_, list(dict.fromkeys(series_.StudyInstanceUID)), threads=64, ahead=24, tag="basin_local")
            unhook = DP.hook_getitem(bp.DicomStudies, pf)
            try:
                return orig(series_, root_, *args, **kw)
            finally:
                unhook()
                print(f"READAHEAD local: {DP.stop(pf)}", flush=True)
        bp.predict = predict_ra
    base, parts, failed, _ = ensemble.predict(a.root, ids, series, str(work), a.workers)
    print(f"{time.strftime('%H:%M:%S')} local predict {len(base)} studies in {time.time() - t:.0f} s, failed {len(failed)}", flush=True)
    del ensemble
    torch.cuda.empty_cache()
    kdir = (HERE.parent / "oct20_stack_all_v2/kernel") if (a.pkg_dir or a.ids in ("gold", "check")) else (HERE / f"kernel_{t_}")  # DEC: v2's = every final
    #                                                     kernel's (make_kernel asserts it); gold info runs need no kernel (PL variants have none)
    src = nbformat.read(open(next(kdir.glob("*.ipynb"))), 4).cells[0].source
    dec = ast.literal_eval(re.findall(r"^DEC = (\{.*?\})\s+#", src, re.M)[0])
    pred = base.copy()
    for f, dd in dec.items():
        p_, q_ = np.clip(base[f].to_numpy(np.float64), 1e-7, 1 - 1e-7), np.clip(base[dd["context"]].to_numpy(np.float64), 1e-7, 1 - 1e-7)
        pred[f] = 1 / (1 + np.exp(-(np.log(p_ / (1 - p_)) - dd["b"] * (dd["c0"] + dd["c"] * np.log(q_ / (1 - q_))))))
    if a.ids in ("gold", "check"):                       # info runs (Master 22:4x gold-58; 10-22 staging gate: every check study + parts)
        out = f"local_{a.ids}_{t_}{a.out_suffix}.csv"
        pred.loc[ids].to_csv(HERE / out)
        if True:                                         # parts too (routing gates compare parts)
            pd.concat({k: (v if isinstance(v, pd.DataFrame) else v.to_frame("Synovitis")) for k, v in parts.items()}, names=["part"]).to_csv(
                HERE / out.replace(".csv", "_parts.csv"))
        print(f"{a.ids} run: {len(ids)} studies, failed {len(failed)}, all finite {bool(np.isfinite(pred.to_numpy()).all())} -> {out}")
        return
    kd = Path(a.commit_dir)
    kp, kb = pd.read_csv(kd / "check_pred.csv", index_col=0), pd.read_csv(kd / "check_pred_base.csv", index_col=0)
    assert list(kp.index) == ids, "the same 4 check studies in the same order"
    dp = (kp - pred.loc[ids, kp.columns]).abs().max()
    db = (kb - base.loc[ids, kb.columns]).abs().max()
    kparts = pd.read_csv(kd / "check_parts.csv", index_col=[0, 1])
    dparts = {}
    for k in kparts.index.get_level_values(0).unique():
        if k not in parts:
            dparts[k] = float("inf")
            continue
        lp = parts[k] if isinstance(parts[k], pd.DataFrame) else parts[k].to_frame("Synovitis")
        kk = kparts.loc[k].dropna(axis=1, how="all")
        dparts[k] = float(np.nanmax(np.abs(kk.to_numpy(float) - lp.reindex(index=kk.index, columns=kk.columns).to_numpy(float))))
    worst = sorted(dparts.items(), key=lambda x: -x[1])[:5]
    pd.concat({"local": pred.loc[ids], "kaggle": kp}).to_csv(HERE / f"local_ref_{t_}.csv")
    ok = dp.max() <= 1e-3 and db.max() <= 1e-3 and max(dparts.values()) <= 1e-3 and not failed and np.isfinite(pred.to_numpy()).all()
    print(f"check_pred max |dp| per column (Kaggle vs local): {dp.map(lambda x: float(f'{x:.2e}')).to_dict()}")
    print(f"check_pred_base max |dp|: {db.max():.2e}; parts: {len(dparts)} compared, worst {[(k, float(f'{v:.2e}')) for k, v in worst]}")
    print(f"LOCAL REFERENCE GATE ({a.variant}):", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
