"""DICOM read-ahead for the FAST workers (Submission Timing 10-22 reference, + Experts: hit-rate / resident-ahead stats; OAI: no).
EXACT: it only reads files into the page cache; every consumer still reads and parses its files itself, so outputs are bit-identical.

Why:
- Kaggle scoring VMs differ hugely in input-read latency. gold-58, same package: Basin wait_data 4-5 vs 160-212 s per worker per chunk
  (~26 s / study on slow VMs).
- Basin reads each study's files ONE BY ONE (DicomStudies.__getitem__ -> infer_prep._read = pydicom.dcmread per file, then
  POSIX_FADV_DONTNEED) in only 2 DataLoader workers: latency-bound.
- The crop stage and knee_infer's main pass already read in parallel (knee_infer.io_pool, 128 threads); the crop stage was unaffected.
Design:
- A separate PREFETCHER PROCESS (no threads inside the FAST worker, which forks DataLoader workers: fork + threads can deadlock).
  It reads the consumer's upcoming studies, file by file, with `threads` parallel readers (16 MB blocks), at most `ahead` studies
  beyond the consumer's position.
- Position: an 8-byte /dev/shm file. The consumer's Dataset.__getitem__(i) writes i (any process, fork-safe: pwrite). The prefetcher
  takes the running max, so two loader workers writing i+1 then i is fine.
- Basin's post-read eviction does not fight the prefetch: it evicts files Basin has already consumed; the prefetch window is ahead of
  them. A bounded window keeps the page-cache use at ~ahead x study size (~24 x 120 MB = ~3 GB).
- Failure mode: if the prefetcher dies, consumers read as before (slower, same outputs); stop() reports its exit code / stats.
Use (per stage, in the FAST worker):
    pf = start(root, studies_in_consumer_order, threads=64, ahead=24, tag="basin")   # the prefetcher lists studies itself
    unhook = hook_getitem(bp.DicomStudies, pf)          # must run BEFORE the stage creates its DataLoader (workers fork after)
    try: ...stage...  finally: unhook(); stats = stop(pf)
Prefetcher process: python dicom_prefetch.py PROGRESS_PATH JOBS_JSON THREADS AHEAD STATS_JSON"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BLOCK = 16 << 20


def study_groups(root, studies):
    """Every file of each study (sorted), in the consumer's study order."""
    return [sorted(os.path.join(d, f) for d, _, fs in os.walk(os.path.join(root, s)) for f in fs) for s in studies]


def set_progress(path, i):
    fd = os.open(path, os.O_WRONLY)
    try:
        os.pwrite(fd, int(i).to_bytes(8, "little", signed=True), 0)
    finally:
        os.close(fd)


def get_progress(path):
    with open(path, "rb", buffering=0) as fh:
        return int.from_bytes(fh.read(8), "little", signed=True)


def start(root, studies, threads=64, ahead=24, tag="pf", log=None):
    """Prefetch root/<sid>/** for `studies` (the consumer's order). The PREFETCHER lists each study lazily (lister threads) when it
    enters the window, so the consumer process does no up-front directory walk (slow on a high-latency mount)."""
    base = f"/dev/shm/knee_pf_{os.getpid()}_{tag}"
    with open(base + ".pos", "wb") as fh:
        fh.write((-1).to_bytes(8, "little", signed=True))
    json.dump({"root": root, "studies": list(studies)}, open(base + ".jobs.json", "w"))
    groups = studies
    out = open(log or base + ".log", "w")
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), base + ".pos", base + ".jobs.json", str(threads), str(ahead),
                          base + ".stats.json"], stdout=out, stderr=subprocess.STDOUT)
    return {"proc": p, "base": base, "out": out, "t0": time.time(), "n": len(groups)}


def hook_getitem(cls, pf):
    """Wrap cls.__getitem__ so every item request first publishes its index. Returns the unhook function."""
    orig, path = cls.__getitem__, pf["base"] + ".pos"

    def gi(self, i):
        set_progress(path, i)
        return orig(self, i)
    gi.__wrapped__ = orig
    cls.__getitem__ = gi

    def unhook():
        cls.__getitem__ = orig
    return unhook


def stop(pf, wait=5.0):
    """Ask the prefetcher to finish (it exits when the consumer passed the last study) or kill it after `wait` s. -> stats dict."""
    set_progress(pf["base"] + ".pos", 1 << 40)
    try:
        pf["proc"].wait(wait)
    except subprocess.TimeoutExpired:
        pf["proc"].kill()
        pf["proc"].wait()
    pf["out"].close()
    st = {"returncode": pf["proc"].returncode, "wall_s": time.time() - pf["t0"], "studies": pf["n"]}
    try:
        st |= json.load(open(pf["base"] + ".stats.json"))
    except (OSError, ValueError):
        pass
    for ext in (".pos", ".jobs.json", ".stats.json", ".log"):
        try:
            os.remove(pf["base"] + ext)
        except OSError:
            pass
    return st


def _read(path):
    n = 0
    try:
        with open(path, "rb", buffering=0) as fh:
            while True:
                b = fh.read(BLOCK)
                if not b:
                    return n
                n += len(b)
    except OSError:
        return n


REWIND = 16   # a published index this far below the running max = a NEW pass over the same studies (Basin: one pass per model group);
#             2 loader workers x prefetch 2 batches x bs 2 reorder by < 16


def _list(root, sid):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(os.path.join(root, sid)) for f in fs)


def _main(pos, jobs, threads, ahead, stats):
    J = json.load(open(jobs))
    root, groups = J["root"], J["studies"]
    t0, seen, nxt, nbytes, nfiles, max_ahead, passes = time.time(), -1, 0, 0, 0, 0, 1
    pending, listing = [], []
    hits = n_pos = res_sum = 0       # Experts (Master 08:3x): per new consumer position i, HIT = study i fully read when the position was
    futs = {}                        # seen (polled every 50 ms: a mild upper bound); resident_ahead = fully read studies right after i
    resident = lambda k: k in futs and all(f.done() for f in futs[k])    # noqa: E731
    with ThreadPoolExecutor(threads) as pool, ThreadPoolExecutor(8) as lister:
        while True:                                                  # until stop(): consumers may start further passes
            cur = get_progress(pos)
            if cur >= 1 << 40:
                break
            if 0 <= cur < seen - REWIND:
                seen, nxt, passes = cur, cur + 1, passes + 1
                futs = {}                                            # a new pass: residency is tracked afresh
            elif cur > seen:                                         # the consumer reached new studies (this pass)
                for k in range(max(seen + 1, 0), cur + 1):
                    hits += resident(k)
                    n_pos += 1
                r = 0
                while resident(cur + 1 + r):
                    r += 1
                res_sum += r
            seen = max(seen, cur)
            lo = max(seen, 0)
            nxt = max(nxt, lo + 1 if seen >= 0 else 0)           # never read what the consumer already has
            while nxt < len(groups) and nxt < lo + 1 + ahead:
                listing.append((nxt, lister.submit(_list, root, groups[nxt])))   # in order; reads start as soon as each listing lands
                nxt += 1
            max_ahead = max(max_ahead, nxt - lo)
            while listing and listing[0][1].done():
                k, li = listing.pop(0)
                fs = li.result()
                futs[k] = [pool.submit(_read, f) for f in fs]
                pending += futs[k]
                nfiles += len(fs)
            done = [f for f in pending if f.done()]
            nbytes += sum(f.result() for f in done)
            pending = [f for f in pending if not f.done()]
            time.sleep(0.05)
        for _, li in listing:
            li.cancel()
        nbytes += sum(f.result() for f in pending if not f.cancelled())
    json.dump({"prefetched_studies": nxt, "files": nfiles, "gb": nbytes / 2**30, "s": time.time() - t0, "max_ahead": max_ahead,
               "passes": passes, "positions": n_pos, "hit_rate": hits / max(n_pos, 1), "resident_ahead_mean": res_sum / max(n_pos, 1)},
              open(stats, "w"))


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
