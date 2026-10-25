"""CPU test of a variant's FINAL kernel setup block (OAI: no; no GPU: CUDA hidden), on the REAL datasets of that variant (the registry's
ds/<slug>/ dirs, laid out like /kaggle/input via KNEE_INPUT): render() the kernel and exec its source up to the DEC line (every dataset's
file set + md5 vs EXPECT, the WORK union mirror + main_sd rebuild, the RUNTIME print, the module-origin / code-feature asserts), in both
fingerprint modes. Negative test: a COPY of the config dataset with one byte changed must stop the kernel (the registry is never touched).
  CUDA_VISIBLE_DEVICES= conda run -n rsna python submissions/oct21_final_alldata/test_kernel_setup.py VARIANT
(needs build.py --variant VARIANT and audit.py --variant VARIANT PASS first)"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_kernel as MK  # noqa: E402


def run(src, env, scr):
    f = scr / "setup_exec.py"
    f.write_text(src[:src.index("\nDEC = ")] + '\nprint("SETUP OK", PKG, sorted(os.listdir(PKG + "/main_sd")), flush=True)\n')
    return subprocess.run([sys.executable, str(f)], capture_output=True, text=True, env=env, cwd=str(scr))


def main():
    v = sys.argv[1]
    pkg = HERE / f"pkg_{MK.B.tag(v)}"
    ad = json.load(open(pkg / "config.json"))["all_data"]
    ds = HERE / f"reg_{ad['registry']}" / "ds"
    scr = HERE / "_work" / f"kernel_test_{MK.B.tag(v)}"
    shutil.rmtree(scr, ignore_errors=True)
    inp = scr / "input"
    inp.mkdir(parents=True)
    os.symlink(MK.B.REPO / "data/original_data", inp / "rsna-knee-abnormality-detection")
    for s in ad["datasets"]:
        os.symlink(ds / s, inp / s)
    env = dict(os.environ, KNEE_INPUT=str(inp), TMPDIR=str(scr), CUDA_VISIBLE_DEVICES="")
    light = [s_ for s_ in ad["datasets"] if s_.endswith("-base")]
    for mode, lt, q in (("ends", (), 0), ("full", light, 4), ("full", (), 0)):    # the last one ('full', the default) is used by the negative test
        nb, meta, slug, expect = MK.render(v, pkg, mode, lt, q)
        src_ = nb.cells[0].source
        assert (f"tolist()[:{q}]   # QUICK commit" in src_ and 'n_chunks=1)' in src_) if q else ('tolist()[:8]' in src_ and 'n_chunks=2)' in src_), "quick patch"
        mode = f"{mode} light {list(lt)} quick {q}"
        r = run(nb.cells[0].source, env, scr)
        print(f"== MD5 mode {mode}\n" + "\n".join(ln for ln in r.stdout.splitlines() if ln.startswith(("FINAL package", "RUNTIME", "final modules", "SETUP OK")))[:3000],
              r.stderr[-1500:] if r.returncode else "")
        assert r.returncode == 0 and "SETUP OK" in r.stdout, f"the setup block failed ({mode})"
    cs = ad["datasets"][-1]                              # negative: a 1-byte change in (a copy of) the config dataset must stop the kernel
    cp = scr / "cfg_copy"
    shutil.copytree(ds / cs, cp / cs)
    victim = cp / cs / "main_sd_config.json"
    victim.write_text(victim.read_text() + " ")
    (inp / cs).unlink()
    os.symlink(cp / cs, inp / cs)
    r = run(nb.cells[0].source, env, scr)
    assert r.returncode != 0 and "package md5 != the audited build" in r.stderr, r.stderr[-800:]
    print(f"NEGATIVE OK: {cs}/main_sd_config.json changed by one byte -> 'package md5 != the audited build'")
    print(f"meta dataset_sources {meta['dataset_sources']}")
    print("KERNEL SETUP TEST PASS")


if __name__ == "__main__":
    main()
