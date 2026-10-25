# Investigate #1: reading the LB and picking the 2 finals (2026-10-22 ~12:30 UTC)

Status at writing: P1 57517691 (93 plain) = 0.965, hidden order ABOVE safe3 57497980 (81 plain) = 0.965. P2 (87), P3 (117), P4 (120),
P5 (130) pending. Sources: work/alldata_gap_2026-10-22/inv1.md, probes.md.

## Public-LB noise model

- Display: 3 decimals. The API sort gives the hidden order. Band estimates (uniform within the 0.001 band, order statistics) carry
  sd ~0.0002-0.0003 per estimate.
- Model-draw noise (same recipe, new seeds). OOF seed replicates: one model's column AUC moves sd 4-15e-3 on 400 studies. A full
  package redraw (e.g. all-data vs fold v2) moves the public LB ~+-0.0004-0.0005, assuming ~400 public studies; scale by
  sqrt(400 / n_public) if public is larger.
- Packages that share most models differ much less. P1 vs safe3 (81 of 93 shared; the changed members carry small column shares):
  difference noise ~1e-4. P2-P5 vs P1 change more members, partly with new (PL) models: ~2-4e-4.
- How to read gaps:
  - between packages that share most models: below ~2e-4 is not readable;
  - between unrelated draws (all-data vs fold v2, plain vs PL-heavy): below ~5e-4 is not readable;
  - a hidden-order flip of two 0.965s is a coin flip unless the band estimate gap > ~2 sd.

## Expected private gain of P1's plain seed adds (w5b 9, bfs_rep 5, oa_v2cor 5, sd1b 5)

- +1.45e-4 LB in expectation with the worst-case seed-noise table (the one our 1 -> 3 seed LB ladder supports; the nominal gold table
  gives ~+0.7e-4).
- More seeds reduce model-draw variance: the gain is not selected on the public LB, so it transfers to private in expectation, without
  winner's curse. P1 > safe3 on public agrees in sign; its size is below resolution, and it is not extra evidence.
- Same mechanism, larger: the all-data step itself (expected +0.0005-0.001 over fold v2 on private; the public reading +0.0001 was
  inside the redraw noise).

## Advice for the 2 finals

1. Private is a different study set. For package DIFFERENCES the public noise part (model draw x public studies) is ~independent of the
   private noise part, so a public gap transfers only shrunk: E[private gap] ~ true / (true + noise) x public gap. For gaps of the
   size of the noise above, expect about half to carry over, or less.
2. Winner's curse: the max of k candidates with similar true value is inflated by ~sigma x E[max of k] (k = 5: ~1.16 sigma). Pick the
   public max only when it leads by > ~2 sigma (>= ~5e-4 against unrelated forms, >= ~2-4e-4 against close relatives). Otherwise
   pick by mechanism / prior.
3. Priors by mechanism (private expectation, best first):
   - plain all-data with more seeds (P1: robust, small +);
   - PL variants: gold reads showed no detectable PL effect (pl_m - plain_m +0.0008, sd 0.0018). Their public rank comes partly from
     being chosen among several routing variants -> discount;
   - fold v2-family: lower prior (less data per model; its public edge was partly probe-selection luck).
4. Expected value vs hedge. Final 1 = the best robust-mechanism package that SCORED (P1, unless a PL / blend variant leads it by
   > ~2 sigma). Final 2 = the strongest scored package of a DIFFERENT mechanism (the best PL / blend that scored), so the two finals
   do not share a single failure mode. Avoid safe3 as final 2: P1 contains it, so it adds no hedge and has a lower expectation.
5. Runtime: choose only submissions that COMPLETED. P3-P5 (117-130 models) carry the slow-VM risk (final118 timed out at 118; slow VMs
   look CPU-starved, and read-ahead did not fix them). A pending large package that times out simply cannot be a final.
