"""Config audit (Master 10-17 01:12, after the '--set' incident): every training line of queue.txt (active or done, incl. commented-out
#DROPPED / #HOLD lines that ran) is checked against what its run actually used. Done runs: runs/NAME/foldF/config.json must carry every
override of the queue line (value equality after the launcher's parsing, '{fold}' filled, tgt_csv rewritten). Pending runs: launch.py --dry
must resolve without error and carry every override. Prints one line per (name, fold) with OK / MISMATCH / PENDING-OK / ERROR.
conda run -n rsna python experiments/v3/oct16_experts/scripts/audit_configs.py"""
import json
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
sys.path.insert(0, str(EXP / "scripts"))
from launch import parse_value  # noqa: E402


def expected(kind, fold, args):
    over = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in args if not kv.startswith("gb="))}
    over = {k: v.replace("{fold}", str(fold)) if isinstance(v, str) else v for k, v in over.items()}
    if kind in ("oa", "bone", "syn") and over.get("tgt_csv"):
        over["tgt_csv"] = str((EXP / over["tgt_csv"]).relative_to(ROOT))
    return over


bad = 0
for line in (EXP / "queue.txt").read_text().splitlines():
    raw = line
    if line.startswith("#"):
        parts = line.lstrip("#").split()
        if not parts or parts[0] not in ("DROPPED_A5", "DROPPED_A1", "HOLD_A5", "HOLD_A4", "DEEPEN_A5", "HOLD_FSPROBE"):
            continue
        line = " ".join(p for p in parts[1:] if not p.startswith("(") and not p.endswith(")") and p not in ("FS", "probe", "57279368", "below", "0.961,", "23:55)", "read", "23:45", "its", "band)"))
        line = line[line.find(next((t for t in line.split() if "_" in t or t.islower()), "")):]
    toks = line.split()
    if len(toks) < 4 or toks[1] == "dump":
        continue
    name, kind, src, fold, args = toks[0], toks[1], toks[2], int(toks[3]), toks[4:]
    over = expected(kind, fold, args)
    cfg_path = EXP / "runs" / name / f"fold{fold}" / "config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        diff = {k: (v, cfg.get(k)) for k, v in over.items() if cfg.get(k) != v}
        status = "OK" if not diff else f"MISMATCH {diff}"
    elif raw.startswith("#"):
        continue                                      # held / dropped and never ran
    else:
        r = subprocess.run([sys.executable, str(EXP / "scripts/launch.py"), "--kind", kind, "--src", src, "--name", name, "--fold", str(fold),
                            "--set", *[x for x in args if not x.startswith("gb=")], "--dry"], capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
        res = next((ln for ln in r.stdout.splitlines() if ln.startswith("launch resolved config ")), None)
        if res is None:
            status = f"ERROR {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no output'}"
        else:
            cfg = json.loads(res[len("launch resolved config "):])
            diff = {k: (v, cfg.get(k)) for k, v in over.items() if cfg.get(k) != v}
            status = "PENDING-OK" if not diff else f"PENDING-MISMATCH {diff}"
    bad += not status.endswith("OK")
    print(f"{name:12s} f{fold} {kind:5s} {status}")
print(f"audit: {bad} problem(s)")
