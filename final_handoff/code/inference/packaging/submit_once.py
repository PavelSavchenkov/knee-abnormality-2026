"""Submit ONE gated kernel version to the competition and exit (no polling; USER 10-22: P1-P5 may be submitted once all planned checks
pass). Writes <probe dir>/ref.txt and submit.log; timing is a separate call (time_ref.py). OAI: no.
  conda run -n rsna python submissions/oct22_probes/submit_once.py PROBE_DIR KERNEL_SLUG VERSION   (description = PROBE_DIR/description.txt)"""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "rsna-knee-abnormality-detection"
d, kernel, version = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
assert not (d / "ref.txt").exists(), f"{d} was already submitted"
desc = (d / "description.txt").read_text().strip()
api = KaggleApi()
api.authenticate()
before = {s.ref for s in api.competition_submissions(COMP)}
t = datetime.now(timezone.utc)
out = subprocess.run(["kaggle", "competitions", "submit", COMP, "-k", kernel, "-v", version, "-f", "submission.csv", "-m", desc],
                     capture_output=True, text=True)
new = [s for s in api.competition_submissions(COMP) if s.ref not in before]
log = f"{t:%Y-%m-%d %H:%M:%S} UTC submit {kernel} v{version}\n{out.stdout.strip()}\n{out.stderr.strip()}\nnew refs: {[s.ref for s in new]}\n"
(d / "submit.log").write_text(log)
print(log, flush=True)
assert len(new) == 1, "expected exactly one new submission"
(d / "ref.txt").write_text(f"{new[0].ref}\n")
print(f"REF {new[0].ref} status {new[0].status}")
