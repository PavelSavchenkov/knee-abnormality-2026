# PL packages: gold sanity, prior ranking, final-pair advice

PL session (8b), 2026-10-22 13:28 BST. OAI: no in every package.

## 1. Gold-58 table

Exact package outputs (Experts' local_ref --ids gold) against safe3w9, in AUC x1e-3. Source: `work/pl_alldata_2026-10-21/best_gold_table.txt`; script: `best_gold_table.py`.

| package | slot | models | MACRO | 90% boot | ACL | MCL | MM | LM | MOA | LOA | PF | Eff | Syn | Bak | Cont | Frac | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seedrow (null) | - | 87 | -0.9 | | +1.2 | -2.3 | 0.0 | -6.2 | -3.1 | -11.6 | +11.6 | -2.5 | +1.2 | -3.6 | 0.0 | +4.2 | - |
| b1 | P2 | 87 | +2.5 | [-0.3, +5.6] | +2.5 | +9.1 | -2.4 | +1.2 | -4.7 | +3.9 | +21.9 | +3.7 | -4.8 | 0.0 | 0.0 | 0.0 | PASS |
| b6 | P3 | 117 | +1.9 | [-0.7, +4.7] | +1.2 | +9.1 | +1.2 | 0.0 | -3.1 | +1.9 | +18.0 | +1.2 | -4.8 | -3.6 | 0.0 | +1.4 | PASS |
| b6c | P4 | 120 | +3.1 | [+0.2, +6.4] | +2.5 | +9.1 | +3.6 | +1.2 | 0.0 | 0.0 | +27.0 | +1.2 | -4.8 | -3.6 | 0.0 | +1.4 | PASS |
| b6ca | P5 | 130 | +2.6 | [-0.4, +5.7] | +2.5 | +9.1 | +1.2 | -2.5 | -1.6 | +3.9 | +25.7 | +1.2 | -4.8 | -3.6 | 0.0 | 0.0 | PASS |
| b2 | not used | 87 | +2.0 | [-1.6, +5.7] | +3.7 | +9.1 | -1.2 | -3.7 | -6.2 | +1.9 | +27.0 | +2.5 | -4.8 | -5.4 | 0.0 | +1.4 | FLAG (MOA -3.7 sd) |
| full PL (pl_m) | info | 87 | +0.7 | [-3.7, +5.2] | +3.7 | +9.1 | -1.2 | -3.7 | -6.2 | -9.7 | +27.0 | +2.5 | -4.8 | -5.4 | -2.7 | 0.0 | FLAG (MOA) |

- safe3w9 macro on gold-58: 0.9552.
- Established seed-row sd (macro): 0.0014.
- Seed-noise sd per column (RMS of 5 seed-swap nulls, floored at one rank swap): ACL 1.2, MCL 4.3, MM 1.2, LM 4.8, MOA 1.7, LOA 7.7, PF 5.6, Eff 3.0, Syn 3.3, Bak 4.0, Cont 2.3, Frac 3.3.
- Rule (agreed with Master): FLAG if macro < -1.4, or if any column < -3 sd.

Caveats:
- **The routes are partly in-sample.** In b6c / b6ca, I chose the column routes (OA MOA plain, PF PL; generalist MOA plain) partly from the d1-dp4 gold reads of the same 58 studies. Their gold lead over b6 (+0.7 to +1.2) is therefore not independent evidence.
- **The null is a partial swap.** The fresh seedrow swaps only recipes with spare plain seeds:
  - fully swapped: sub_b0, oa_v2cor / _nf, sd1b, bfs_rep, z1_acl
  - 2 of 3 swapped: m_prob05, nfnet, r50, b3g
  - not swapped: bone, LOA extras, syn crops, EF3, w5b

  So the column sds understate noise for those groups, which errs toward flagging.
- **The bootstrap ranges all overlap.** No candidate is separable from another on gold. Package-level PL vs plain is not separable from seed noise either (full PL +0.7).
- **Synovitis is -4.8 in every PL package**, because the syn crops are PL in all of them. Gold has under-read synovitis against the public LB before.

## 2. Mechanism

Evidence from exact gold reads d1-dp4 plus the label-free stability and diversity reads.

- **PL members are steadier across seeds.** rho_PL / rho_plain is below 1 for every recipe and column (median ~0.53), so one PL seed is worth about 2 plain seeds of noise reduction. The noisiest plain member, w5b_mcl, gains the most (0.42).
- **PL makes members more alike in every column.** The corr between recipes rises from plain to PL; the biggest rise is Fracture, 0.88 -> 0.95.
  - Where plain members were noisy, the noise cut wins: MCL, PF (OA experts), Effusion (generalists).
  - Where the member's value was diversity or a label design, PL loses:
    - bone: Contusion -4 to -5 in every batch
    - LOA extras: silsep's v4b silent-cell labels were overwritten by the teacher (silent-cell mean 0.07 -> 0.13), LOA -10 to -19
    - Medial OA: -2 to -6, consistent across batches
- **Every candidate routes around these known losses.** Bone and LOA extras stay plain; the routed MOA is plain. Image-read heads do not lose under PL (EF3 gains).

## 3. Prior ranking of P1-P5 by expected PRIVATE AUC

Gold is too small to rank these, so the ranking rests on prior reasoning.

1. **P5 b6ca (130).** It uses every expected-positive lever:
   - plain + PL 'both' on the noisy high-weight groups
   - PL where stability and mechanism agree
   - plain where PL is known to hurt (bone, LOA extras, MOA)
   - extra plain seeds on the top-share members, plus 1:1 twins

   Risk: the longest runtime (slow-VM estimate ~335 min with read-ahead).
2. **P4 b6c (120).** Same design without the extra seeds. The routing is mechanism-backed, but its gold support is partly in-sample.
3. **P3 b6 (117).** Group-level 'both', so MOA and Baker's stay half exposed to PL's small losses. It is the most robust PL form if the routes don't transfer.
4. **P2 b1 (87).** Pure swaps, no hedge: it takes PL's PF / MCL / Eff gains and its MOA loss (-4.7) at 87 size. Higher variance than the 'both' forms.
5. **P1 inv1_a (93, plain).** safe3w9 + plain seeds; expected ~ safe3 + ~1.5e-4. It has the lowest variance and does not depend on PL transferring.

P1 and P2 are close: P2 has the higher mean only if PL transfers, and it has the larger spread.

The expected gaps within P3-P5 are ~1-3e-4, below public-LB noise (~4-5e-4). The LB cannot rank them reliably.

## 4. Advice for the two finals

Kaggle scores the best of the two selected submissions on private, so the pair should be strong AND different.

- **Pick one PL and one plain.** The largest open uncertainty is whether PL transfers to private; a plain + PL pair hedges exactly that.
- **PL final:** the best public LB of P3-P5. If they sit within ~0.0005 of each other, take P5 (prior), else P4. Prefer P3 only if it leads the routed ones by more than ~0.0008.
- **Plain final:**
  - safe3 (scored 0.965) or P1, whichever is higher on public.
  - If within noise, P1 (more seeds, same recipe).
  - Keep safe3 if P1's public score is lower by more than ~0.0005.
- **Do not pair two PL siblings** (P4 + P5, or P3 + P4). They share most models, so the best-of-2 gains almost nothing.
- **Exception:** if every PL package lands below the plain one on public by more than ~0.0008, PL likely doesn't transfer. Then take the plain best + the strongest PL anyway, as a cheap hedge, rather than two plain packages.
