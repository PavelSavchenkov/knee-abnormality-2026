# Final selection input: Experts V3 (written 2026-10-22 ~13:3x BST; OAI: no in every package below)

All packages are ALL-DATA except v2. Form for all five probes = safe3 form: v2 form + Contusion all-4 + Fracture v3c_fs + EF3 Effusion,
fc_off, EFF_C off, sxm OFF (sxm was the common factor of both over-runs: sxm probe 499 min, final118 > 540 min).
Every probe is rendered on the read-ahead render (safe3w9ra) with --local-copy, kernel version v1.
Gold-58 numbers below: local_ref.py --ids gold (deterministic), 12-column mean, paired bootstrap over the 58 studies vs safe3w9.

## Reference points

| | ref | kernel | contents | public | runtime |
|---|---|---|---|---|---|
| v2 | 57416173 | oct20_stack_all_v2 kernel | FOLD package (v2 form) | 0.964 | 205 min |
| safe3 | 57497980 | knee-oct21-fa-safe3-lctest-sub v1 | all-data, 81 = 27 recipes x 3 seeds, --local-copy, no read-ahead | 0.965 | 190 min |

## Probes

### P1 inv1_a = 57517691 (knee-oct21-fa-inv1-a-sub v1), 93 models, PLAIN (no PL)
- What it is: safe3 + more plain seeds: w5b_mcl 9 seeds (USER), bfs_rep 5, oa_v2cor 5, sd1b 5.
- What it tests: seed count only. Expected gain vs safe3 ~ +1.5e-4: invisible at 3 decimals, inside the LB noise.
- Gates: audit PASS, setup test PASS, Kaggle commit gate vs safe3w9's commit PASS (73 shared parts, all 0.0).
- Gold: no gold-58 run of inv1_a itself. safe3w9 (87 = safe3 + w5b s104-s109) = 0.9552.
- Scored 0.965 in 175.7 min. Master's lb_order check puts it above safe3, so it is our best scored package.
- Code path: same as scored safe3, plus the read-ahead prefetcher (proven bit-identical locally and on Kaggle). No routing overlay.

### P2 best_b1 = 57520594 (knee-oct21-fa-best-b1-sub v1), 87 models, PL-ROUTED BY GROUP (one choice per recipe)
- PL twins (P201-P203, teacher-v1 pseudo-labels) replace plain for:
  - OA (oa_v2cor, oa_v2cor_nf, sd1b, bsd_nf);
  - ligaments (w5b_mcl P201-P209, z1_acl);
  - EF3 Effusion slots;
  - synovitis crops (cs_fs, cs_fc125, s_fs_nf, s_fc125_nf);
  - m_prob05 / n_prob05, b3g.
- Kept PLAIN: generalists (sub_b0, nfnet, r50), bone (b3c/b3l/b3n/bfs_rep), LOA extras (oasep, oasep_nf, silsep).
- What it tests: replacing plain with PL in the groups where PL looked neutral-to-good on gold.
- b2 (same, but generalists PL) was dropped: gold Medial OA -3.7 sd.
- Gates: audit PASS, setup PASS, commit gate PASS (13 shared parts 0.0; Baker's / Contusion / Fracture parts 0.0), 8b gold check PASS.
- Gold-58: 0.9578, +0.0025 vs safe3w9 (sd 0.0018; 95% CI [-0.0008, +0.0063]; P(<=0) 0.07). The lead is mostly PF OA +0.022 and MCL +0.009.
- Weakness: same model count as safe3w9 but every PL slot has only 3 seeds.

### P3 best_b6 = 57520062 (knee-oct21-fa-best-b6-sub v1), 117 models, PLAIN + PL BLEND
- Plain AND PL twins side by side (twins averaged as extra seeds) for sub_b0, nfnet, r50, oa_v2cor, oa_v2cor_nf, sd1b, w5b_mcl (9 + 9), z1_acl.
- PL only: m_prob05, n_prob05, b3g, bsd_nf, EF3 slots, synovitis crops.
- Plain only: bone, LOA extras.
- What it tests: PL as extra ensemble members (no slot is replaced wholesale for the main recipes).
- Gates: audit PASS, setup PASS, commit gate PASS (13 shared parts 0.0; Contusion 0.0), 8b gold PASS.
- Gold-58: 0.9571, +0.0019 (sd 0.0016; CI [-0.0011, +0.0051]; P(<=0) 0.10).

### P4 best_b6c = 57520478 (knee-oct21-fa-best-b6c-sub v1), 120 models, COLUMN-ROUTED (new code)
- Base: the b6 member set, with plain + PL routed per COLUMN for:
  - generalists sub_b0 / nfnet / r50: Medial OA plain; ACL, MM, LM, LOA, PF, Fracture PL; r50 MCL PL;
  - m_prob05: Synovitis PL;
  - oa_v2cor: Medial OA plain, PF PL, menisci both.
- The window stack holds plain + PL files.
- Unrouted columns are averaged as in b6.
- What it tests: whether picking plain or PL per column beats the blend.
- Gates:
  - audit PASS (incl. 6c routing invariants), setup PASS;
  - commit gate PASS: 15 shared parts 0.0; PF routed to PL moves 0.107 on edge-4 and MOA plain 0.002, both as designed;
  - routing gate PASS: switch-off bit-identical; routed columns == group-level PL twin exactly; unrouted main columns <= 9.8e-7 fp32 noise;
  - local-vs-Kaggle reference gate PASS: edge-4 <= 7.9e-5; 106 parts, worst 3.3e-4;
  - 8b gold PASS.
- Gold-58: 0.9584, the highest, +0.0031 (sd 0.0019; CI [-0.0004, +0.0070]; P(<=0) 0.04).
- BUT the lead is partly in-sample: the routes were chosen on this same gold-58, and the PF OA +0.027 carries most of it.

### P5 best_b6ca = 57521751 (knee-oct21-fa-best-b6ca-sub v1), 130 models, COLUMN-ROUTED + MORE SEEDS
- P4 + plain bfs_rep 5, oa_v2cor 5 (+ PL twins P204-P205), sd1b 5 (+ PL twins P204-P205), 1:1 plain/PL twins. The "max estimate".
- Gates: audit PASS, setup PASS, commit gate PASS (13 shared parts 0.0), routing gate PASS, local reference gate PASS (edge-4 <= 7.1e-5;
  106 parts, worst 3.3e-4), 8b gold PASS.
- Gold-58: 0.9578, +0.0026 (sd 0.0019; CI [-0.0010, +0.0065]; P(<=0) 0.08).
- The extra seeds cost LM -0.0025 and MOA -0.0016 vs P4 on gold: seed noise, not a signal.
- Highest runtime of the five (most models). Not yet measured on scoring.

## How to read the LB differences

- Noise scale: one full model redraw moves public ~+-0.0004-0.0005 (Investigate #1, OOF seed-replicate draws).
  - Differences of 1e-4 to 5e-4 between any two of these packages are coin flips, even with lb_order's exact ranking.
  - All probes share the same test studies, so a difference is mostly the members that changed, but those members' own noise is large.
- Gold-58 is weak evidence here:
  - seed-only null variants move the gold mean +-0.0013, while the PL leads are +0.0019 to +0.0031;
  - gold breadth has not predicted LB before;
  - all-data members were trained on the gold-58 studies (training labels), so every all-data gold read is partly in-sample;
  - P4's routes were picked on gold.
- What each pair isolates:
  - P1 - safe3: plain seeds only (expected +1.5e-4; observed P1 > safe3, consistent).
  - P2 - P1: PL replaces plain in most expert groups, with fewer seeds (87 vs 93). The cleanest read of "PL vs plain".
  - P3 - P1: PL added as extra members (117 vs 93). It mixes "more members" and "PL", so a P3 gain is not a pure PL gain.
  - P4 - P3: column routing vs blending, on the same member set except P4's routed generalists and +3 models.
  - P5 - P4: more plain/PL seeds (seed effect, expected ~ +1e-4).
- If P2-P5 all land within ~5e-4 of P1, the honest reading is that PL is neutral on public.
- A PL probe that clearly beats P1 (>= ~1e-3, i.e. visible at 3 decimals) is a real signal.

## Caveats you might miss

1. Fallbacks:
   - Prediction path: none. FAST strict, a dead worker raises, no silent fallbacks.
   - Speed only: if the read-ahead prefetcher process dies, the consumers read the DICOMs themselves (slower, IDENTICAL outputs).
     stop() records the exit code and stats; it does not raise. This can cost runtime, never correctness.
2. Read-ahead: proven bit-identical locally (88 studies, preds + 103 parts 0.0, hit rate 1.00) and on Kaggle (commit gate 72 parts 0.0).
   - It does NOT help a CPU-starved VM: on slow VMs, Basin waited even at hit rate 1.0.
   - Runtime spread between VMs stays the main timeout risk for the biggest packages.
3. Routing overlay (pkg_code_route/ensemble.py + knee_infer.py): ships ONLY in P4 and P5.
   - It is new code from today. It is gated, but only 2 commits (P4, P5) have run it on Kaggle.
   - Its main pass predicts 2 weight sets in one pass. Unrouted main columns differ from the plain code path by <= 9.8e-7: harmless.
   - P1-P3 run the same code as scored safe3 + read-ahead.
4. --local-copy: weights copied to local disk (/tmp/pkgw), full md5 check. This is the same setting as scored safe3 (lctest v1) and scored P1.
   - The USER veto on /tmp was about DICOM staging (not used in any probe), not this weight copy.
5. Runtime:
   - P1 93 models = 175.7 min (scored).
   - P3 117 / P4 120 / P5 130 add PL twins of light-to-medium members. Not measured in scoring yet; expected well under 540 but
     the slow-VM factor (~2x read waits) applies.
   - Prefer a final whose scoring time is KNOWN once P2-P5 complete.
6. PL teacher: all PL twins learn from teacher-v1 (fold-package OOF) pseudo-labels.
   - Its gold-58 gains (PF OA mostly) may partly reflect fitting the teacher's view of these studies.
   - The private LB is the only independent judge.
7. Hedge: P1 (plain, proven runtime, our best scored) and the best-scoring PL probe are the most different pair.
   - Two PL probes as the finals would put both on the same untested-on-private PL bet.
