"""Audit every run of oct14_all_data against its queue line (user 10-15 00:40: "keep checking that you are training exactly what we
need"). Per run:
- recipe: config.json == source config.json (path remaps applied) + the queue overrides; any other difference is printed;
- data: log header 'train 4349, val 58, gold 58' (fold -1 = every non-gold study trains; gold-58 is never trained on);
- steps: the schedule unit / total / stop the trainer printed == the queue's spe_ref x epochs and stop; finished runs reached it;
- checkpoints: every saved model has its optimizer state (main-like: model_ep / opt_ep; Basin: model_e / model_raw_e / opt_e) and
  the deployable final checkpoint exists (main: model.pt; Basin: the highest model_e; oa / bone / syn: the highest model_ep);
- health: finite losses; final gold-58 (main / Basin metric of the trainer) printed for a sanity look.
conda run -n rsna python experiments/v2/oct14_all_data/scripts/audit.py [NAME ...]   (default: every run with a config.json)"""
import json
import math
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
sys.path.insert(0, str(EXP / "scripts"))
from launch import DROP, REMAP, parse_value  # noqa: E402

N_ALL, N_GOLD = 4349, 58


def queue_lines():
    """name -> (kind, src, fold, overrides) from queue.txt and archived queues (latest wins)."""
    out = {}
    for f in sorted((EXP / "logs").glob("queue_*.txt")) + [EXP / "queue_step3_A.txt", EXP / "queue.txt"]:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            t = line.lstrip("#").split()
            if len(t) >= 4 and t[1] in ("main", "basin", "oa", "bone", "syn") and t[2].startswith("experiments/"):
                over = dict(kv.split("=", 1) for kv in t[4:] if "=" in kv)
                over.pop("gb", None)
                out[t[0]] = (t[1], t[2], int(t[3]), {k: parse_value(v) for k, v in over.items()})
    return out


def audit(name, q):
    kind, src, fold, over = q
    d = EXP / "runs" / name / f"fold{fold}"
    issues, info = [], []
    cfg = json.load(open(d / "config.json"))
    s = json.load(open(ROOT / src / "config.json"))
    exp = {k: REMAP.get(v, v) if isinstance(v, str) else v for k, v in s.items() if k not in DROP} | over
    for k, v in exp.items():
        got = cfg.get(k)
        if isinstance(v, (list, tuple)) or isinstance(got, (list, tuple)):
            v, got = list(v) if v is not None else v, list(got) if got is not None else got
        if got != v and not (isinstance(v, (int, float)) and isinstance(got, (int, float)) and math.isclose(v, got)):
            issues.append(f"cfg {k}: {got!r} != expected {v!r}")
    extra = {k: cfg[k] for k in cfg if k not in exp and k not in DROP and cfg[k] not in (0, 0.0, False, "", None, [], 1.0, True)}
    if extra:
        info.append(f"keys not in source (trainer defaults): {extra}")
    log = (EXP / "logs" / f"{name}_f{fold}.log").read_text(errors="ignore")
    hdr = re.search(r"train (\d+)(?:, val| eval) (\d+)", log)
    if kind == "basin":
        ok = hdr and int(hdr.group(1)) == N_ALL
    else:
        ok = hdr and int(hdr.group(1)) == N_ALL and int(hdr.group(2)) == N_GOLD
    if not ok:
        issues.append(f"data header {hdr.group(0) if hdr else 'missing'} (expected train {N_ALL}" + ("" if kind == "basin" else f", val {N_GOLD}") + ")")
    m = re.search(r"schedule unit (\d+)\), total(?: steps)? (\d+)(?:, stop at (\d+))?", log)
    spe = over.get("spe_ref") or 543
    eps = cfg.get("epochs", 20)
    want_total = spe * eps
    want_stop = over.get("stop_step") or (over.get("stop_epoch", 0) * 543 if over.get("stop_epoch") else want_total)
    if not m:
        issues.append("schedule line missing")
    else:
        unit, total = int(m.group(1)), int(m.group(2))
        if unit != spe or total != want_total:
            issues.append(f"schedule unit {unit} / total {total} != expected {spe} / {want_total}")
        info.append(f"steps: unit {unit}, total {total}, stop {want_stop}")
    done = (d / "done.txt").exists()
    if kind == "basin":
        c = json.load(open(d / "metrics.json"))["curve"] if (d / "metrics.json").exists() else []
        e = {int(x.group(1)) for f in d.iterdir() if (x := re.match(r"model_e(\d+)\.pt$", f.name))}
        miss = [k for k in sorted(e) if not (d / f"model_raw_e{k}.pt").exists() or not (d / f"opt_e{k}.pt").exists()]
        if miss:
            issues.append(f"Basin checkpoints without raw / opt: {miss}")
        if c:
            last = c[-1]
            info.append(f"evals {[r['epoch'] for r in c]}, last gold {last.get('gold', float('nan')):.4f} gold11 {last.get('gold11', float('nan')):.4f}")
            if any(not math.isfinite(r.get("loss", 0)) for r in c):
                issues.append("non-finite loss")
            if done and e and max(e) != last["epoch"]:
                issues.append(f"final model_e{max(e)} != last eval epoch {last['epoch']}")
            if done:
                want_ep = math.ceil(want_stop / 543)
                if last["epoch"] != want_ep:
                    issues.append(f"last epoch {last['epoch']} != expected stop epoch {want_ep} (stop step {want_stop})")
    else:
        e = {int(x.group(1)) for f in d.iterdir() if (x := re.match(r"model_ep(\d+)\.pt$", f.name))}
        miss = [k for k in sorted(e) if not (d / f"opt_ep{k}.pt").exists()]
        if miss:
            issues.append(f"checkpoints without opt: {miss}")
        losses = [float(x) for x in re.findall(r"^ep \d+ loss ([\w.+-]+)", log, re.M)]
        if any(not math.isfinite(x) for x in losses):
            issues.append("non-finite loss")
        g = re.findall(r"(?:gold|gold3|Frac_gold|g_syn) (0\.[0-9]+)", log)
        info.append(f"epochs logged {len(losses)}, saved {sorted(e)}, last gold {g[-1] if g else 'n/a'}")
        if done:
            if kind == "main" and not (d / "model.pt").exists():
                issues.append("model.pt missing")
            want_ep = math.ceil(want_stop / 543) - 1                 # 0-based epoch index of the last (partial) epoch
            if not e or max(e) != want_ep:
                issues.append(f"last saved epoch {max(e) if e else None} != expected {want_ep}")
    return done, issues, info


def main():
    Q = queue_lines()
    names = sys.argv[1:] or sorted(p.parent.parent.name for p in (EXP / "runs").glob("*/fold-1/config.json") if not p.parent.parent.name.startswith("SMK"))
    bad = 0
    for n in names:
        if n not in Q:
            print(f"{n}: not in any queue file")
            continue
        done, issues, info = audit(n, Q[n])
        state = "DONE" if done else "running/partial"
        bad += bool(issues)
        print(f"{n} [{state}] {'OK' if not issues else 'ISSUES'}")
        for i in issues:
            print(f"   !! {i}")
        for i in info:
            print(f"      {i}")
    print(f"AUDIT {'PASS' if not bad else f'FAIL ({bad} runs with issues)'}")


if __name__ == "__main__":
    main()
