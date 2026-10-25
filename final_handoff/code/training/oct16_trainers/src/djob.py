"""Kind 'djob' (launch.py; Master 10-19 21:1x, the USER's inference-sweep question): an inference-only DUMP of deployed members at a new
eval setting, run by the box runner as an ordinary queue line (no training). One line = one member x fold; both sets (oof, gold) through the
existing dumpers as subprocesses; launch.py writes runs/NAME/foldF/done.txt afterwards. OAI: no.
  job=basin member=nfnet|rad t_eval=T: scripts/dump_basin_win.py on refs/basin961/basin_{member}_{F} (the deployed 0.961 Basin fold models)
            -> data/win_new/teval{T}_{nfnet|r50}_f{F}_{set}.npz (C137 names);
  job=bone run=RUN: scripts/dump_bone_win.py RUN F {set} (k 64 per bag) -> data/bone_win/RUN_f{F}_{set}.npz;
  job=zoom run=RUN k_eval=K: scripts/dump_zoom.py RUN F {set} --k-eval K -> data/zoom_win/RUN_k{K}_f{F}_{set}.npz.
Queue: NAME djob refs/djob F job=... [member= | run=] [t_eval= | k_eval=] gb=N"""
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
DEFAULTS = dict(job="", member="", run="", t_eval=24, k_eval=0)


def load_bags(cfg):
    return {}


def commands(cfg, fold):
    s = str(EXP / "scripts")
    if cfg["job"] == "basin":
        assert cfg["member"] in ("nfnet", "rad"), cfg["member"]
        tag = "r50" if cfg["member"] == "rad" else "nfnet"
        ref = (EXP / "refs/basin961" / f"basin_{cfg['member']}_{fold}").relative_to(REPO)
        return [[f"{s}/dump_basin_win.py", m, f"teval{cfg['t_eval']}_{tag}_f{fold}_{m}", "--config", f"{ref}.json", "--ckpt", f"{ref}.pt",
                 "--fold", str(fold), "--t-eval", str(cfg["t_eval"])] for m in ("oof", "gold")]
    if cfg["job"] == "bone":
        return [[f"{s}/dump_bone_win.py", cfg["run"], str(fold), m] for m in ("oof", "gold")]
    if cfg["job"] == "zoom":
        assert cfg["k_eval"] > 0, "zoom: k_eval required"
        return [[f"{s}/dump_zoom.py", cfg["run"], str(fold), m, "--k-eval", str(cfg["k_eval"])] for m in ("oof", "gold")]
    raise ValueError(f"unknown job {cfg['job']!r}")


def run_fold(cfg, fold, out, bags):
    for c in commands(cfg, fold):
        print("djob:", " ".join(c), flush=True)
        subprocess.run([sys.executable] + c, check=True, cwd=REPO)
