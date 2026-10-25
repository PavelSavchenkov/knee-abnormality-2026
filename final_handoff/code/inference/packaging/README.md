# oct21_final_alldata — FINAL all-data package (draft for Master's review)

USER 10-21 13:5x (via Master): train the best fold-avg pipeline per class on 100 % of the data with new seeds, then ship it. OAI: no in every member.
Nothing is uploaded from here. Master reviews the code before any Kaggle action, and each submission needs the user's own go.

## What it builds

- `package/` is ONE flat, self-contained dataset: the runtime work config of the final form, with every member's fold models replaced by its recipe's all-data runs.
  - The final form is the work config the stack kernels build in WORK.
  - All-data runs are fold -1 with the fold runs' step count via `spe_ref`, several seeds.
- Form: v2 (57416173) + Contusion all-4 (57455847) + EF3 img_eff crop slots (57455943). Switches, default OFF:

| switch | probe | effect |
|---|---|---|
| `--frac` | Fracture v3c_fs 57475191 | the new bone members fuse in both groups (`fuse_sets ['cont', 'frac']`) + `frac_allmem` + `frac_fs` (bfs_rep, w 0.75) |
| `--effc` | Effusion form (c) 57479153 | `eff_main`: Effusion window slot main = ef3m, NFNet = e3bn, R50 = shipped |
| `--sxm` | Contusion sxm 57480087 | `cont_sxm`: sxm img_cont spliced into Contusion at w 0.3 |
| `--fc-off` | needs `--frac` | drop the then-dead b_v3 / b_lr3 / b_nf and skip the shipped fc recipe + fusion (`fc_shipped_off`, ensemble.py) |

- Merge: each kit's work config comes from `oct20_stack_all_v2/run_local.build_work`. Its diff vs v2 is asserted to touch only its own keys (`TOUCH` in build.py).
- `pkg_code/ensemble.py` = the frac kit's ensemble.py (cont_allmem + frac_allmem + frac_fs) + the sxm patch + the eff_c patch (with Master's run-list fix) + the guarded `fc_shipped_off` hook.
- `pkg_code/fc_stage.py` is identical to the all-4 work copy.
- The FAST runner (fast.py) calls `ensemble.predict` with faked stage outputs and builds bone fusion keys via `fc_stage.member_sets`. So `fuse_sets` and the hook run there too.

## Usage (MULTI-DATASET, Master 10-21 16:2x: weights uploaded once in frozen datasets, a small config dataset per variant)

```
B="conda run -n rsna python submissions/oct21_final_alldata/build.py"
$B --stock                 # every finished all-data run of every recipe -> reg_final/store/ (hardlinks; sd / knee strips; key-for-key vs shipped)
$B --freeze base           # all not-yet-frozen store files (+ the loc / detector assets on the first freeze) -> reg_final/ds/knee-oct21-fa-base/
$B --stock; $B --freeze d1 # later batches -> small delta datasets knee-oct21-fa-d1, -d2, ... (a file lives in ONE dataset)
$B --variant NAME          # its config dataset reg_final/ds/knee-oct21-fa-cfg-NAME/ (code + config.json + syn members + main_sd_config.json)
                           # + the local union mirror pkg_NAME/; refuses if any needed file is not frozen
conda run -n rsna python submissions/oct21_final_alldata/audit.py --variant NAME         # PASS -> audit_NAME.json
conda run -n rsna python submissions/oct21_final_alldata/make_kernel.py --variant NAME   # kernel_NAME/ (needs that PASS); dataset_sources = the variant's datasets
CUDA_VISIBLE_DEVICES= conda run -n rsna python submissions/oct21_final_alldata/test_kernel_setup.py NAME   # CPU: the kernel's setup on the real datasets
```

- Upload each frozen dataset ONCE (`kaggle datasets create -p reg_final/ds/<slug>`), then each variant's small config dataset.
- A lighter variant (fewer seeds) = another `--variant` with a smaller SEEDS table: a new config dataset only, no weight re-upload. Its kernel mounts the same weight datasets; files it doesn't use sit idle.
- `--reg test` keeps a separate registry for dry runs (10-21 16:2x: the partial end-to-end ran there).
- `last_ckpt` asserts the checkpoint epoch equals the run's own schedule end (guards a partial sync).

- A VARIANT (build.py `VARIANTS`) = the form (switches) + the seeds per recipe. Each variant gives one dataset `knee-oct21-final-<name>` and one kernel `knee-oct21-final-<name>-sub`.
- Seeds per recipe: None = every finished run. Reused recipes use `experiments/v2/oct14_all_data/runs/A<k>_<name>/fold-1` (labels A1..A5). New recipes use `experiments/v3/oct16_experts/runs/<name>_all_s<seed>/fold-1`.
- The package is FLAT, because `kaggle datasets create` skips subdirectories. The SD-2 main weights sit at the top level, and their config is `main_sd_config.json`. The kernel rebuilds `main_sd/` in WORK.
- Box runs must be synced to the local `runs/` first.
- Checkpoint = the LAST saved file, as in s5: main / mainp `model.pt`, Basin the highest `model_e*.pt`, oa / bone / syn the highest `model_ep*.pt`.
  - Example: ef3_all_s101 ends at model_ep16, because 435 x 20 steps at 543 steps / epoch.
- The SD-2 MCL members (sd1b main, bsd_nf Basin) drop their training-only `sd_*` keys, as oct18_probe_mcl_sd2.
- bsd_nf's json = an NFNet all-data run config (the deployed architecture json, as SD-2).
- `config.json` → `all_data` records, per file: source checkpoint, md5, {recipe, seed, run, src}.

## Audit (audit.py, CPU)

1. Every run used: done.txt, and its file = the last checkpoint.
   - New runs: config.json == the queue's `--src` config except fold / seed / spe_ref / name / out / stop_epoch / stop_step / save, plus silsep's `tgt_csv` (data/targets/v4b_fall.csv). The queue line must name that src with fold -1.
   - oct14 runs: configs equal across seeds except seed / name / out.
   - Keys absent from the source config (trainer DEFAULTS) are printed for review.
2. Seeds are distinct within each recipe (labels and config seeds).
3. Every weight == its source (md5). The SD-2 files: key set == source minus `sd_*`.
4. No fold model: every `.pt` is an all-data file or a localiser / detector asset.
5. Dead members are absent: s_fs / s_fc125 effb0, bfs_fs; and b_v3 / b_lr3 / b_nf with `fc_shipped_off`.
6. Config parity vs v2, key by key. Only these may differ: file lists, the switched blocks, EF3 Effusion stats / pipelines, `cont_fs.name`, `all_data`. The recipe constant blocks are checked as unchanged.
7. Strict loads of every file by the package loaders, one member at a time. RAM: `basin_predict.load_model.__wrapped__` skips its lru cache.

Status: audit.py compiles but is untested; it needs a real build (≥ 1 finished seed per recipe).

## Dead members (code proofs)

- **s_fs / s_fc125 effb0 syn**: the EF3 kernel deletes them (oct20_probe_eff_ef3_v2/make_kernel.py). Their only use was the old img_eff 'main' eff_pipeline. The Synovitis pipelines and fuse designs use cs_*.
- **bfs_fs**: nofuse, used only by the cont_fs splice. cont_allmem then overwrites Contusion with (1 - w) zf2 + w z(bfs_rep) using the same constants. So `cont_fs.name` → bfs_rep and bfs_fs is dropped.
- **b_v3 / b_lr3 / b_nf** (only if `--frac` wins): F and C both come from the new groups, so the shipped fusion / recipe result is overwritten. `--fc-off` removes them and their compute (13 models).
- **m_fs, pw main / NFNet / R50 (Q25), PF-Q20 align**: the v2 switches frac_mfs / bakers_allpw / lm_allpw / pf_q20 are False.

## Constants caveat (as oct15_all_data_s5)

Recipe constants are unchanged: OA / men / fc / eff recipes, fuse / align stats, splice w / mu / sd, DEC.
- The member-side constants (mu_m / sd_m, align mu_f / sd_f) were fitted on fold-model OOF logits.
- All-data members are also scored out of sample at test time, but their logit scale may differ slightly (more data, other seeds).
- s5 shipped the same way and scored 0.962+. No re-fit is possible without OOF of the all-data models.

## RUNTIME: models run per stage

v2 (from its work config, `stage_counts` in build.py): main 8, main_sd 5, basin 18, win_main 5, oa 21, fc 18, syn 33 (= 108).

Final form with n = seeds per new recipe; the reused oct14 recipes keep their counts (sub_b0 5, m_prob05 3, nfnet 5, r50 5, n_prob05 3, oa_v2cor 5, oa_v2cor_nf 3, b_v3 5, b_lr3 5, b_nf 3, s_fs_nf 3, s_fc125_nf 3):

| stage | v2 | final (all switches on) | n = 3 | n = 5 |
|---|---|---|---|---|
| main | 8 | 8 | 8 | 8 |
| main_sd | 5 | n (sd1b) | 3 | 5 |
| basin (one pass) | 18 | 13 + n (bsd_nf) + n (e3bn) | 19 | 23 |
| win_main | 5 | 5 + n (ef3m) | 8 | 10 |
| oa | 21 | 8 + 3n (oasep, oasep_nf, silsep) | 17 | 23 |
| fc | 18 | 4n (b3c, b3l, b3n, bfs_rep) + 13 shipped (0 with `--fc-off`) | 25 / 12 | 33 / 20 |
| syn | 33 | 6 + 10n (cs_fs, cs_fc125, w5b, z1, b3g, ef3, ef3fc, ef3nf, ef3nffc, sxm) | 36 | 56 |
| total | 108 | | 116 / 103 | 158 / 145 |

- The real build prints the exact counts (`RUNTIME ... final`).
- **Minutes: anchor on the SCORED probes, not on the bench** (correction 10-21 15:4x, Submission Timing's finding).
  - The 78-study FAST bench is warm-up dominated: its stage shares (basin 36 %, fc 20 %, oa 19 %, syn 15 %, ...) scaled to v2's 205 min underestimate the per-study crop stages about 3x.
  - Scored anchors:
    - v2 57416173: 205 min (108 models).
    - all-4 Contusion 57455847 (+18 bone): 329 min → ~6.9 min per bone copy.
    - b_v3 Contusion 57456024 (+5 bone): ~255 min.
    - EF3 57455943 (+12 syn incl. ConvNeXt, −6 effb0): ~234-256 min → ~3-4.5 min per heavy syn copy.
    - The 10-21 00:xx anchors scored 4 side by side (maybe +10-20 %).
  - Rough per-copy minutes: bone ~6.9, heavy syn ~3.5, basin ~3.3, oa ~2.3 (maybe more), main / window ~1-1.3.
  - 7 seeds of every recipe (ALL_ON, `--fc-off`) ≈ 205 + ~353 ≈ 558 raw min: OVER 540. Keep a margin (Kaggle ±20 %).
  - `--fc-off` saves ~13 x 6.9 ≈ 90 min if Fracture v3c_fs wins.
  - Update the anchors with the eff_c / sxm / Fracture scoring times.

## Seed shares (for the allocation)

work/alldata_final_2026-10-21/shares.py → shares.csv: d(final logit) / d(recipe seed-mean logit) per column. It runs the package math on the gold-58 fold dumps; forms are ALL_ON plus one switch off at a time.
- Main levers:
  - bfs_rep: .50 of Fracture with `--frac`, plus .15 of Contusion.
  - sxm: .28 of Contusion.
  - b3g: .28 ACL + .26 MCL.
  - z1: .35 ACL.
  - w5b: .25 MCL.
  - OA crop groups: .2-.3 of MOA / PF / MM / LM.
  - Window slots: r50 .30 of Effusion in form c.

## Kernel (make_kernel.py)

`kernel_<variant>/` = the LB-scored sxm kit notebook (oct21_probe_cont_sxm/kernel: FAST two-GPU runner, STRICT, the 8-study commit check ON) with only its setup region replaced:
- PKG = the variant's dataset. Every package file is md5-asserted against the audited local build (the EXPECT table is embedded). Also asserted: all_data runs / sources.
- WORK = a mirror (code copied, weights symlinked) + `main_sd/` rebuilt. Config and code are used unchanged.
- A RUNTIME print, plus the module-origin and code-feature asserts.
- DEC is asserted equal to v2's. Everything after the setup is the kit's text.
- make_kernel.py also runs pyflakes on the generated source (no undefined names).

test_kernel_setup.py (CPU, CUDA hidden) runs the generated setup block on a MOCK flat package made from the fold-form (b) package:
- md5 gate, mirror, main_sd, RUNTIME print, module asserts: PASS (193 files, 7.2 GB, md5 in 12 s locally).
- Negative test: a 1-byte change fails the md5 assert.

## Open

- **(a) vs (b) identity test** (`identity_test.py`): waits for the Fracture LB result.
  - If Fracture wins, Master opens a local window (gpu_go 0 → run → gpu_go 1).
  - Pass = max |dp| ≤ 1e-6 on all 12 columns, plus 0 failed studies.
- **Audit on a real partial build** as soon as each recipe has ≥ 1 done seed (~16:30).
