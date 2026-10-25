"""Readiness of the FINAL variant (USER 10-21 20:0x go for the full 115-model package; OAI: no; CPU, read-only): every planned seed of the
final form (build.VARIANTS['final'] seeds; FINAL_FORM recipes only: fc_off drops b_v3 / b_lr3 / b_nf, effc off drops ef3m / e3bn, sxm per
FINAL_FORM) must pass the store gate (done.txt + last checkpoint at the schedule's final epoch + audit check 1 config). Prints the missing
seeds whenever the list changes; --wait polls every 60 s and exits 0 when none is missing.
  conda run -n rsna python submissions/oct21_final_alldata/ready_final.py [--variant=NAME] [--wait]"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402

VAR = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--variant=")), "final")
FORM = B.VARIANTS[VAR]["form"]
DEAD = {"b_v3", "b_lr3", "b_nf"} if FORM["fc_off"] else set()
DEAD |= set() if FORM["effc"] else {"ef3m", "e3bn"}
DEAD |= set() if FORM["sxm"] else {"sxm"}


def missing():
    out = []
    for r, labels in B.VARIANTS[VAR]["seeds"].items():
        if r in DEAD:
            continue
        kind, src, name = B.RECIPES[r]
        for s in labels:
            d = B.run_dir(r, s)
            why = None
            if not (d / "done.txt").exists():
                why = "not done"
            else:
                try:
                    B.last_ckpt(d, kind)
                    p = B.run_config_problems(r, d)
                    why = ("config: " + "; ".join(p)) if p else None
                except AssertionError as e:
                    why = f"ckpt: {str(e)[-60:]}"
            if why:
                out.append(f"{r}:{s} ({why})")
    return out


def main():
    total = sum(len(v) for r, v in B.VARIANTS[VAR]["seeds"].items() if r not in DEAD)
    last = None
    while True:
        m = missing()
        if m != last:
            print(f"{time.strftime('%H:%M')} {VAR} ({total} models): {total - len(m)} ready, missing {len(m)}: {m}", flush=True)
            last = m
        if not m or "--wait" not in sys.argv:
            return 0 if not m else 1
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
