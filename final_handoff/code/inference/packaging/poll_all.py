"""Read-only: ONE poller for all pending probes (replaces per-probe time_ref.py, which hit HTTP 429 at 4 x 30 s). Every 120 s, one
ListSubmissions call; prints each status change with minutes since acceptance and writes <probe dir>/submit_timing.json when final.
Exits when every probe is COMPLETE / ERROR. OAI: no.
  conda run -n rsna --no-capture-output python submissions/oct22_probes/poll_all.py"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
HERE = Path(__file__).parent
dirs = {int((d / "ref.txt").read_text()): d for d in sorted(HERE.glob("p*_*")) if (d / "ref.txt").exists() and not (d / "submit_timing.json").exists()}
api = KaggleApi()
api.authenticate()
last = {}
while dirs:
    try:
        subs = {s.ref: s for s in api.competition_submissions(COMP) if s.ref in dirs}
    except Exception as e:
        print(f"{datetime.now(timezone.utc):%H:%M:%S} poll failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(120)
        continue
    now = datetime.now(timezone.utc)
    for ref, s in subs.items():
        t0 = s.date if s.date.tzinfo else s.date.replace(tzinfo=timezone.utc)
        st = str(s.status)
        if last.get(ref) != st:
            last[ref] = st
            print(f"{dirs[ref].name} {ref} {now:%H:%M:%S} UTC {st} ({(now - t0).total_seconds() / 60:.1f} min after acceptance)", flush=True)
        if "COMPLETE" in st or "ERROR" in st:
            r = {"ref": ref, "accepted_utc": f"{t0:%Y-%m-%d %H:%M:%S}", "final_seen_utc": f"{now:%Y-%m-%d %H:%M:%S}",
                 "minutes": round((now - t0).total_seconds() / 60, 1), "status": st, "public": s.public_score,
                 "error": getattr(s, "error_description", "")}
            json.dump(r, open(dirs[ref] / "submit_timing.json", "w"), indent=1)
            print(f"{dirs[ref].name} FINAL {json.dumps(r)}", flush=True)
            del dirs[ref]
    time.sleep(120)
