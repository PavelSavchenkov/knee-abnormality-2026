"""End-to-end audit of the FINAL all-data package/ (Master 10-21 14:1x; OAI: no; CPU). Each check prints PASS / FAIL:
1. every all-data run used has done.txt + its checkpoint (the LAST saved file), and its config.json == the source recipe's fold config except
   fold / seed / spe_ref / name / out, the stop / save / eval-only / worker keys, and silsep's tgt_csv (v4b_fall); new DEFAULT keys = absent in
   the source (listed, must be default-off - printed for review);
2. the seed labels are distinct within each recipe and the run configs' seeds are distinct too;
3. every weight file is byte-identical (md5) to its source checkpoint (the SD-2 members: their stripped key set == the source minus sd_*);
4. NO fold model anywhere in the package: every .pt is an all-data file listed in config all_data.sources or a localiser / detector asset;
5. dead members absent: s_fs / s_fc125 effb0, bfs_fs (+ b_v3 / b_lr3 / b_nf with fc_shipped_off);
6. CPU parity vs v2's work config, key by key: only the file lists, the members' files / run / epoch, the switched blocks (cont_allmem,
   frac_allmem, frac_fs, eff_main, cont_sxm, fc_shipped_off), the EF3 Effusion stats / pipelines, cont_fs.name and all_data may differ;
7. strict loads of every weight by the package's own classes (knee_infer.KneeNet, basin_predict.load_model, syn_stage.load_experts,
   oa_stage / fc_stage loaders).
8. the package is FLAT (kaggle datasets create skips subdirectories) and the switched blocks equal their kits' blocks (files aside).
PASS -> audit_<variant>.json {pass, config md5, date}: make_kernel.py requires it for the same config.json.
conda run -n rsna python submissions/oct21_final_alldata/audit.py --variant NAME"""
import argparse
import time
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True          # check 7 imports the package's modules from PKG: never leave __pycache__ in the (flat) dataset
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PKG = None                     # package_<variant> (main)
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402

md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()   # noqa: E731
ok = True
E = REPO / "experiments"
TOUCHED = ("weights", "models", "basin", "basin_syn", "basin_sources", "win.main", "win.basin", "mcl_sd.nfnet", "oa.members", "fc.members",
           "syn.pipelines", "syn.eff_pipelines", "syn.fuse.designs", "cont_allmem", "frac_allmem", "frac_fs", "eff_main", "cont_sxm",
           "fc_shipped_off", "eff.pipelines", "eff.stats.Effusion", "cont_fs.name", "all_data")
ROUTED = ("main_pl", "oa.loa_extra", "oa.extra_sets", "oa.extra_align", "men.experts")   # oct22 column routing: allowed only in routed variants


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(': ' + str(detail)[:600]) if detail else ''}")


def main():
    global PKG
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    PKG = HERE / f"pkg_{B.tag(ap.parse_args().variant)}"       # the union mirror build.py --variant writes (symlinks into the registry's ds/ dirs)
    cfg = json.load(open(PKG / "config.json"))
    reg = HERE / f"reg_{cfg['all_data']['registry']}"
    sindex, sdatasets = json.load(open(reg / "store_index.json")), json.load(open(reg / "datasets.json"))
    ad = cfg["all_data"]
    srcs = {n: s.split(" [")[0] for n, s in ad["sources"].items()}
    # 1: runs done + checkpoint = the last saved file (the schedule's final epoch) + config gate = build.run_config_problems (one source of
    #    truth for plain and PL runs: queue --src / A1 parity except the all-data overrides; PL runs also PL_KEYS + a PL target)
    runs = {}
    for n, r in ad["runs"].items():
        runs.setdefault(r["run"], r)
    bad1 = []
    for run, r in runs.items():
        d = REPO / run
        kind = B.RECIPES[r["recipe"]][0]
        files = [n for n, x in ad["runs"].items() if x["run"] == run]
        try:
            ck = str(B.last_ckpt(d, kind).relative_to(REPO))
        except AssertionError as e:
            bad1.append((run, str(e)[-80:]))
            continue
        if not (d / "done.txt").exists() or any(srcs[n] != ck for n in files):
            bad1.append((run, "not done / not the last checkpoint"))
        p_ = B.run_config_problems(r["recipe"], d)
        if p_:
            bad1.append((run, p_))
    npl = sum(B.is_pl(r["seed"]) for r in runs.values())
    check(f"1. {len(runs)} all-data runs ({npl} PL) done, last checkpoint = the schedule's end, config gate (build.run_config_problems)", not bad1, bad1)
    # 2: seeds distinct
    by = {}
    for run, r in runs.items():
        by.setdefault(r["recipe"], []).append((r["seed"], json.load(open(REPO / run / "config.json")).get("seed")))
    dup = {r: s for r, s in by.items() if len({x[0] for x in s}) != len(s) or len({x[1] for x in s}) != len(s)}
    check("2. seeds distinct within each recipe (labels and config seeds)", not dup, dup or {r: [x[1] for x in s] for r, s in by.items()})
    # 3: md5 / stripped key sets
    import torch
    bad3 = []
    for n, s in ad["sources"].items():
        p = PKG / n
        rec = ad["runs"][n]["recipe"]
        if "[strip_" in s:                               # stripped: tensor-equal to strip(source) (build.STRIP / build.strip)
            ref, got = B.strip(torch.load(REPO / srcs[n], map_location="cpu"), B.STRIP[rec]), torch.load(p, map_location="cpu")
            if set(ref) != set(got) or any(not torch.equal(ref[k], got[k]) for k in ref):
                bad3.append(n)
        elif md5(p) != ad["md5_src"][n]:
            bad3.append(n)
        if set(torch.load(p, map_location="cpu")) != B.ship_keys(rec):
            bad3.append(f"{n}: keys != the shipped fold file {B.SHIP_REF[rec]}")
    check(f"3. {len(ad['sources'])} weight files == their source checkpoints (md5; stripped = tensor-equal to strip(source)) AND key-for-key == "
          "the shipped fold file of the recipe", not bad3, bad3)
    # 4: no fold model
    assets = {cfg["oa"]["loc"], cfg["oa"]["detector"], cfg["fc"]["loc"], cfg["fc"]["detector"], cfg["syn"]["loc"], cfg["syn"]["detector"], cfg["basin_detector"]}
    pts = [p for p in PKG.rglob("*.pt")]
    stray = [str(p.relative_to(PKG)) for p in pts if p.name not in sindex or not (sindex[p.name]["recipe"] == "asset" or "run" in sindex[p.name])]
    unused = [p.name for p in pts if p.name not in ad["sources"] and p.name not in assets]
    check(f"4. no fold model in the union ({len(pts)} .pt = {len(ad['sources'])} used all-data + {len(assets)} assets + {len(unused)} unused "
          "all-data seeds of other variants; every .pt is a registry store file)", not stray, stray)
    # 5: dead members absent
    syn = json.load(open(PKG / cfg["syn"]["members"]))
    dead = [n for n in syn if n.startswith(("s_fs_", "s_fc125_")) and "_nf" not in n] + [n for n in cfg["fc"]["members"] if n == "bfs_fs"]
    if cfg.get("fc_shipped_off"):
        dead += [n for n in cfg["fc"]["members"] if n.startswith("runs/b_")]
    check("5. dead members absent (s_fs / s_fc125 effb0, bfs_fs" + (", b_v3 / b_lr3 / b_nf)" if cfg.get("fc_shipped_off") else ")"), not dead, dead)
    # 6: parity vs v2's work config
    v2 = json.load(open(HERE / "_work/v2/config.json"))
    f, f0 = B.flat(cfg), B.flat(v2)
    routed = bool(cfg.get("main_pl") or cfg.get("basin_pl") or any(n.endswith("@pl") for n in cfg["oa"]["members"]))
    touched = TOUCHED + (ROUTED if routed else ())
    other = sorted(k for k in set(f) | set(f0) if json.dumps(f.get(k), sort_keys=True) != json.dumps(f0.get(k), sort_keys=True) and not k.startswith(touched))
    check("6. config parity vs v2: only file lists / switched blocks / EF3 stats / cont_fs.name / all_data differ", not other, other)
    for k in ("oa.recipe", "men", "fc.recipe", "fc.fuse", "oa.fuse", "syn.fuse.spec", "mcl_member", "mcl_member2", "acl_member", "acl_member2", "cont_fs.w"):
        a_, b_ = ({x: v for x, v in f.items() if x.startswith(k) and not (routed and x.startswith(ROUTED))},
                  {x: v for x, v in f0.items() if x.startswith(k) and not (routed and x.startswith(ROUTED))})
        if a_ != b_:
            check(f"6b. constant block {k} unchanged", False, k)
    if routed:                                           # 6c: oct22 routing structure (build.route_config's invariants)
        oa_ = cfg["oa"]
        pls = [n for n in oa_["members"] if n.endswith("@pl")]
        dep = [n for n in oa_["members"] if n not in oa_.get("loa_extra", [])]
        mpl = json.load(open(PKG / "main_pl_config.json"))["weights"] if cfg.get("main_pl") else []
        bpl = sum((cfg.get("basin_pl") or {}).values(), [])
        plain_files = set(cfg["weights"]) | set(sum(cfg["basin"].values(), []))
        bad6c = ([n for n in pls if n not in oa_["loa_extra"] or n in oa_.get("loa_set", [])]
                 + [k for k, ms in (oa_.get("extra_sets") or {}).items() if not set(ms) <= set(oa_["members"]) or k not in oa_.get("extra_align", {})]
                 + [x for x in cfg["men"]["experts"] if x not in dep and x not in pls]
                 + sorted((set(mpl) | set(bpl)) & plain_files)
                 + [k for k in cfg["win"]["basin"] if cfg["win"]["basin"][k] not in (cfg["basin"][k], cfg["basin"][k] + (cfg.get("basin_pl") or {}).get(k, []))]
                 + ([] if cfg["win"]["main"] in ([w for w in cfg["weights"] if w.startswith("sub_b0_")],
                                                 [w for w in cfg["weights"] if w.startswith("sub_b0_")] + [w for w in mpl if w.startswith("sub_b0_")]) else ["win.main"]))
        check(f"6c. routes: OA PL members {pls} only in loa_extra (not loa_set), extra_sets {oa_.get('extra_sets')}, men {cfg['men']['experts']}, "
              f"main_pl {len(mpl)} / basin_pl {len(bpl)} files disjoint from plain, windows plain or plain + PL", not bad6c, bad6c)
    # 7: strict loads by the package's own loaders, ONE file / member at a time (RAM: never through an lru cache)
    sys.path.insert(0, str(PKG))
    import gc
    import basin_predict as bp
    import fc_stage
    import knee_infer as ki
    import oa_stage
    import syn_stage
    n7 = 0
    for cj in ("config.json", "main_sd_config.json") + (("main_pl_config.json",) if (PKG / "main_pl_config.json").exists() else ()):
        c = json.load(open(PKG / cj))
        for w in c["weights"]:
            sd = torch.load(PKG / w, map_location="cpu")
            ki.KneeNet(c["backbone"], n_out=sd["cls_b"].shape[0], pretrained=False, hidden=c["hidden"], drop=c["drop"], use_type=c["use_type"],
                       use_pos=c["use_pos"], head=c["head"]).load_state_dict(sd)
            n7 += 1
    em = cfg.get("eff_main") or {}
    for fs in (list(cfg["basin"].values()) + list(cfg["basin_syn"].values()) + [cfg["mcl_sd"]["nfnet"]] + list(em.get("basin", {}).values())
               + list((cfg.get("basin_pl") or {}).values())):
        for w in fs:
            getattr(bp.load_model, "__wrapped__", bp.load_model)(str(PKG / w))   # (lru-cached in some package versions: bypass the cache)
            n7 += 1
            gc.collect()
    for n, e in cfg["oa"]["members"].items():
        n7 += len(oa_stage.load_experts(str(PKG), {"members": {n: e}, "n_types": cfg["oa"]["n_types"]}, torch.device("cpu")).get(n, [[]])[0])
        gc.collect()
    for n, e in cfg["fc"]["members"].items():
        n7 += len(fc_stage.load_experts({n: e | {"files": [str(PKG / x) for x in e["files"]]}}, torch.device("cpu")).get(n, [[]])[0])
        gc.collect()
    syn_all = syn | ({"sxm": {"cfg": cfg["cont_sxm"]["cfg"], "files": cfg["cont_sxm"]["files"]}} if cfg.get("cont_sxm") else {})
    for n, e in syn_all.items():
        n7 += len(syn_stage.load_experts({n: {"cfg": e["cfg"], "files": [str(PKG / x) for x in e["files"]]}}, torch.device("cpu"))[n][0])
        gc.collect()
    n_w = len(ad["sources"])
    check(f"7. strict loads by the package loaders: {n7} of {n_w} all-data files (KneeNet main + main_sd, Basin load_model, OA / FC / syn load_experts)"
          + ("" if not em.get("main") else " + eff_main main below"), n7 + len(em.get("main", [])) == n_w, (n7, n_w))
    if em.get("main"):
        c = json.load(open(PKG / "config.json"))
        for w in em["main"]:
            sd = torch.load(PKG / w, map_location="cpu")
            ki.KneeNet(c["backbone"], n_out=sd["cls_b"].shape[0], pretrained=False, hidden=c["hidden"], drop=c["drop"], use_type=c["use_type"],
                       use_pos=c["use_pos"], head=c["head"]).load_state_dict(sd)
    # 8: flat + the switched blocks = their kits' (files / member names aside)
    check("8a. the union mirror is flat (no subdirectory)", not [x for x in PKG.iterdir() if x.is_dir()], [x.name for x in PKG.iterdir() if x.is_dir()])
    bad8 = []
    for slug in ad["datasets"]:                          # every dataset dir: flat, no junk; a weights dataset = exactly its frozen registry
        d_ = reg / "ds" / slug
        names = {x.name for x in d_.iterdir()} - {"dataset-metadata.json"}
        if [x for x in d_.iterdir() if x.is_dir()] or any(n.endswith((".pyc", "~")) or n.startswith(".") for n in names):
            bad8.append(f"{slug}: junk / subdir")
        if slug in sdatasets and (names != set(sdatasets[slug]["files"]) or any(md5(d_ / n) != m for n, m in sdatasets[slug]["files"].items())):
            bad8.append(f"{slug}: files / md5 != the frozen registry")
    check(f"8a'. the {len(ad['datasets'])} datasets are flat, junk-free, frozen sets intact ({ad['datasets']})", not bad8, bad8)
    nofile = lambda d: {k: v for k, v in d.items() if k not in ("files", "main", "basin")}   # noqa: E731
    for blk, kit in (("cont_allmem", "all4"), ("frac_allmem", "frac"), ("frac_fs", "frac"), ("eff_main", "effc"), ("cont_sxm", "sxm")):
        if cfg.get(blk):
            kc = json.load(open(HERE / "_work" / kit / "config.json"))[blk]
            check(f"8b. {blk} == the {kit} kit's block (files aside)", nofile(cfg[blk]) == nofile(kc), (nofile(cfg[blk]), nofile(kc)))
    e3 = json.load(open(HERE / "_work/ef3/config.json"))
    check("8c. eff recipe (Effusion stats, w) == the EF3 kit's", cfg["eff"]["stats"] == e3["eff"]["stats"] and cfg["eff"]["w"] == e3["eff"]["w"])
    print("AUDIT", "PASS" if ok else "FAIL")
    if ok:
        json.dump({"pass": True, "config_md5": md5(PKG / "config.json"), "datasets": ad["datasets"], "date": time.strftime("%Y-%m-%d %H:%M")},
                  open(HERE / f"audit_{PKG.name.split('pkg_', 1)[1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
