# oct14_vlm: a one-fold Qwen 2B VLM classifier

Started 2026-10-14 ~01:35 BST (user goal: user_directives.md; user ideas: user_ideas.md). Results in time order: ledger.md; ideas
and ranking: ideas.md; numbered check-ins: reviews.md; what the old experiments teach for this model: old_experiments.md.

## SEALED 2026-10-15 ~11:20 BST (user: seal learnings, plans and docs, then quit for the time being)

Nothing runs: gpu_go.txt = 0, runner and watchdog stopped (PIDs killed), no GPU process. The GPU was with the All Data session at the
seal (user directive 10-15 ~11:15, 'GPU BACK' expected ~15:00); GPU allocation for any resume is the user's call.

### Learnings (fold 2, 1 seed, mostly 1 epoch: directions, not final numbers; x1e-3 deltas)

1. The pipeline works end to end: Qwen3.5-2B, ViT fully fine-tuned (fp32 master) + LM LoRA; trained heads on hidden states (no text
   generation); FeatNorm (per-feature standardisation; LM / ViT features are 99%+ shared across studies) fixed the G0 failure; WSD
   schedule (constant LR + AdamW state at every epoch end) makes runs resumable and branchable (cooldown branches, continuations).
2. Level: every variant sits at 0.86-0.88 trusted after 1 epoch (NFNet single 0.936 at e2, 0.975 at e14): far from parity, clearly
   under-trained. Read every new arm on its own curve (memory judge-by-own-learning-curve).
3. Per class (scripts/class_read.py; groups diffuse = Effusion / Synovitis / Baker's / Contusion, focal = ACL / MCL / menisci /
   Fracture, OA = MOA / LOA / PF): the VLM beats NFNet on the diffuse IMAGE reads (Codex effusion +10, synovitis +18), ties diffuse
   text labels (full VLM only), and is far behind on focal (-160..-260) and OA (-60..-120). Contusion behaves focal (-200), not diffuse.
4. Package (0.961) + VLM as a 4th pipeline: overall -1.3..-1.9 trusted at epoch 1, but the Effusion image side is positive in every
   checkpoint (Codex +2..+5 in 16 / 16, gold +2.5..+7.5 in 14 / 16; out/effusion_member_table.txt) while text-label effusion is
   negative: the main lead (user 19:27). Worth ~+0.0002..0.0006 LB as it stands; headroom on VLM-favourable classes is small
   (effusion <= +0.0018 LB; synovitis <= +0.0057 but supervision-limited per the synovitis LB probes; Baker's ~0).
5. Vision-only (ViT + per-finding gated pooling, no LM) = the full VLM at epoch 1 at 3x the speed (0.14 vs 0.43 s / study); the LM
   helps MCL / Baker's / ACL and the diffuse text labels (diffuse trusted -3 vs -51).
6. Inputs: localiser region crops (src/regions.py; notch / compartments / PF / MCL / pouch / Hoffa / popliteal, 56-100 mm squares)
   tie overall (LM +32, MM +15, PF +12..28, ACL -34); 384 px ties (image reads +, 2.4x cost); 256 px = the encoder's minimum input
   (SigLIP2-L/16, 48 x 48 native position grid; coarse semantic features).
7. Adaptation at 8 studies / step: 6 / 6 raises ended negative by epoch 1 (all LRs x3, LM x3, head x10, ViT x3, flat ViT LR, patch /
   position embeddings x12), each leading on the diffuse image reads at 0.25-0.75 epoch. 4x batch (accum 32) + all LRs x3 = the FIRST
   positive change: raw trusted 0.879 vs 0.866 (+13), text +19, gold_nofc +14, EMA gold_nofc +26; diffuse trusted -11 vs NFNet (was
   -51), focal gold -167 (was -240); OA gold worse, Codex synovitis -9. Reading: LR was limited by gradient noise (two knobs changed
   together; disentangling arm queued).
8. Medical-domain ViTs never trained in this repo: OrthoFoundation-L (DINOv3-L + 1.2M knee X-ray / MRI; local), MRI-CORE (plain
   ViT-B/16 DINOv2 on 6.9M MRI slices; runs at 256 px), MedSigLIP-448 / Curia (gated). Backbone Sweep prepped OrthoFoundation-L /
   MRI-CORE + controls on CPU as paused lines; natural ViTs there show the same profile (diffuse ~ NFNet, focal / OA below).

### Next plans (queue.txt order; each read with class_read.py + pkg_value.py + loss_pair.py where batches match)

1. g1v_b32c: the large-batch vision-only model +3 epochs (init g1v_b32@109, AdamW state, warm-up 0, seed 1); full raw + EMA eval per
   epoch; question: do focal / OA keep climbing, does the effusion package value grow?
2. g1v_b32x1 (4x batch at x1 LR: batch or LR?), g1v_b64 (8x batch + x5 LR: push the direction).
3. g1r_b32: the full VLM (regions, LM LoRA) in the large-batch regime vs g1r / g1v_b32 (does the LM add more once noise is lower?).
4. If the continuation keeps climbing: longer WSD runs + a cooldown branch; then retest resolution (384) and the adaptation knobs
   (flat ViT LR, embeddings) inside the large-batch regime; a diffuse-specialist member; T4 fp16 check before any build.
5. Postponed / elsewhere: class-definitions prompt (g1p, user ~18:15: parity first); medical-ViT screens (Backbone Sweep; GPU time
   is the user's call); MedSigLIP / Curia need the user's acceptance of gated terms. No Kaggle action without the user's go.

### Resume

- Restart from the repo root: `setsid nohup bash experiments/v2/oct14_vlm/scripts/run_queue.sh > experiments/v2/oct14_vlm/logs/runner.log 2>&1 < /dev/null &`
  and from experiments/v2/oct14_vlm: `setsid nohup bash scripts/watchdog.sh > logs/watchdog.log 2>&1 < /dev/null &` (pidfiles are rewritten).
- Open the gate only with the GPU: `echo 1 > experiments/v2/oct14_vlm/gpu_go.txt`. g1v_b32c already has 1 start in logs/events.txt
  (the runner gives up after 2). is_done accepts epoch >= epochs - 0.01 (epoch ends at step 436 / 437, or 109 / 110 at accum 32).
- Region / large-batch runs: gradient checkpointing only where noted (full VLM with 34 images needs grad_ckpt_lm=True).

## Question

Can the 2B VLM from the forum report (NguyenThanhNhan #735304: Qwen3.5-2B, 384 px, LoRA on the LM + the vision encoder fine-tuned
jointly, one fold LB 0.950 in ~3.5 h) be trained here, fast and reliably, into (1) a one-fold model that beats our best single model
and (2) a 4th member that adds to the 3-pipeline ensemble (main effb0 / NFNet-L0 / RadImageNet R50)? Labels: v1 text labels +
Codex image reads only; no pseudo-labels.

## What is known before any GPU time

- Forum: only feeding size (384 px), model (Qwen3.5-2B), adaptation (LM LoRA + vision encoder trained jointly) and cost (~3.5 h one
  fold) are disclosed. Slice feeding, labels, epochs: not disclosed. The author defended external data (#743416), so the 0.950 may
  include it. hengck23 (#745861, 10-11): a Qwen3.5-2B MRI model + another MRI model, 50 / 50, LB 0.926 (nothing else known). No
  public VLM notebook (pull 10-14 01:40).
- Our 10-08 attempt (work/vlm/train_vlm.py, Qwen3-VL-2B, 256 px, fold 3): LoRA on LM + vision tower (or the last 6 ViT blocks), 300
  steps or 1 epoch only. The loss stayed at the prior; trusted 0.50-0.69, frozen head-only 0.68, high LR collapsed. Never: full ViT
  fine-tune, 384 px, Qwen3.5, a realistic schedule (memory judge-by-own-learning-curve). Speed then (GPU shared): ~0.6 s / study
  training at 256 px k 16 with LoRA everywhere; bf16 vs pure fp16 corr 0.9999 (Qwen3-VL) / 0.9992 (Qwen3.5); fp16 activation peaks
  18k (Qwen3-VL LM) / 5.9k (Qwen3.5 ViT).
- In every pipeline here, frozen or barely adapting encoders failed (DINOv2-L LoRA -0.023 trusted, DINOv3 ConvNeXt <1% weight drift,
  OrthoDiffusion probe 0.86); fully fine-tuned encoders work. So the vision tower must be fully fine-tuned.
- Diversity comes from structure / pretraining domain (R50 RadImageNet rho 0.85-0.90 vs ImageNet CNN swaps 0.92); a ViT + LLM fusion
  is the most different family we have.

## Validation regime (pre-registered 2026-10-14 01:45)

- Fold 2 of `common.folds.get_folds(5, 0)` (gold excluded from training): ~3,480 train / ~870 val studies, gold-58 test-like. Fold 2
  is the least selected fold for all 3 pipelines (main gated on it; NFNet / R50 only confirmed on it) and has every comparator,
  including an NFNet seed and the image-head models. Confirmation fold (if time): 0.
- Comparators (refs/refs.parquet, scripts/copy_refs.py; out/refs_f2.txt): main effb0 3 replicates (e19), NFNet-L0 + seed 2 (e14),
  R50 (e20), image-head models m_prob05 (+ s1) / n_prob05 / r_prob05.

  | fold 2 | trusted | lookat_m | text | gold | gold_unc | cx_eff | cx_syn (img head) |
  |---|---|---|---|---|---|---|---|
  | main effb0 (3 replicates) | 0.9708-0.9727 | 0.9450-0.9515 | 0.9618-0.9634 | 0.929-0.933 | 0.902-0.905 | 0.947 | 0.987 (mprob) |
  | NFNet-L0 (+ seed 2) | 0.9749 / 0.9746 | 0.9541 / 0.9547 | 0.966 | 0.927 / 0.932 | 0.906 / 0.915 | 0.953-0.956 | 0.989 (nprob) |
  | R50 | 0.9736 | 0.9545 | 0.9663 | 0.935 | 0.919 | 0.954 | 0.984 (rprob) |
  | B3 = main + NFNet + R50 | 0.9774 | 0.9588 | 0.9705 | 0.937 | 0.916 | 0.955 | |
  | B3 + NFNet seed / + main seed | 0.9774 / 0.9771 | 0.9590 / 0.9584 | 0.970 / 0.971 | 0.938 / 0.937 | | | |

  Best single model on fold 2 = NFNet-L0 (trusted 0.9749). Seed noise per run: trusted ~0.0014, lookat ~0.0034, gold ~0.0032.
- Metrics per eval (scripts/vlm_lib.py panel, same cells as the PL / review sessions): trusted, lookat_m, text, thr, un (11 findings
  without synovitis), gold / gold_nofc / gold_unc / gold synovitis, Codex clear-cut reads (effusion; synovitis from the image head)
  incl. report-uncertain cells; per finding; paired study bootstrap (trusted / lookat_m / text) and gold bootstrap.
- Single-model read: VLM vs NFNet (best single) and vs the main replicates; the seed rows give the noise.
- Member read: B3 + VLM (equal 1/4 logit weight) vs B3 + NFNet seed (yardstick) and B3 + main seed; Spearman vs each member;
  per-finding blend deltas; later the 0.961 package's columns (deep-ensembling lib, copied).
- Slow-family rule: judge the VLM on its own validation curve over a realistic schedule; "broken" = prior-level loss for a whole
  epoch, constant predictions, LoRA below the frozen probe, collapse.

## Plan and gates (pre-registered; each gate written in ledger.md with a forecast before it runs)

| gate | what | cost | pass |
|---|---|---|---|
| G0 | code checks (CPU, truncated model): labels / prompt / patch layout / positions / folds / no gold or val leak; then GPU overfit 64 studies | CPU now; ~15 min GPU | train loss on the 64 studies < 50% of the prior-only loss within 20 passes |
| B | speed bench: Qwen3.5 / Qwen3-VL x 256 / 384 px x k x grad-ckpt on / off | ~20 min | pick the fastest setting that fits 32 GB |
| G1 | learning curve: Qwen3.5-2B, full ViT fine-tune + LM LoRA, 256 px, WSD schedule (16:45, user idea): constant LR 4 epochs, full eval + ckpt + AdamW state each epoch, 175-step cooldown branches read the lengths | ~2.5 h | trusted >= 0.93 after epoch 1 and rising; stop if < 0.88 after epoch 1 or flat |
| G2 | forum-like run: 384 px, 3 epochs (or what G1's curve asks for), eval each epoch, every eval checkpoint saved | ~3-4 h | single >= NFNet - 0.003 trusted, or member gain >= the NFNet-seed yardstick |
| G3 | levers by ROI on short runs (readout, input, labels, report aux), then the final one-fold model | rest | per ideas.md |

Forecast (01:45): P(learns properly: trusted >= 0.95 after G2) ~0.7; P(beats NFNet single on trusted on fold 2) ~0.25; P(B3 + VLM
beats B3 + NFNet seed by >= +1e-3 trusted or lookat) ~0.5. Correlation with the 3 pipelines expected ~0.80-0.85 (most different family).

## Layout

- src/: data.py (R2 window dataset copy + series order + eval offsets), labels.py (R2 targets, Codex img heads, explicit-only mask),
  vlm.py (model: visual + LM called directly, batched vision SDPA, read-outs names / attn / vis, LoRA, report loss), train.py
  (training / evals / checkpoints; --smoke, --bench, overfit=N), predict.py (re-score a checkpoint: k_eval, offset TTA, --fp16).
- scripts/: copy_refs.py (comparators -> refs/refs.parquet), vlm_lib.py (panel, blends, vectorised bootstrap), refs_table.py,
  read_run.py (standard read of a run), pkg_value.py (VLM as a 4th pipeline in the 0.961 package), run_queue.sh (queue.txt,
  gpu_go.txt gate), watchdog.sh (log silence -> kill the arm's python), cpu_smoke.sh / cpu_train_smoke.sh (CPU tests).
- Added 10-14 evening: src/regions.py (localiser region crops; data/loc_pred_v2 = localiser v2 predictions copied from
  oct11_meniscus), scripts/class_read.py (per-class / per-group read incl. package columns; --sub for mid-epoch), loss_pair.py
  (paired train loss of same-order runs), token_budget.py (prompt lengths per layout), region_check.py (crop contact sheet).
- data/codex_reads.csv (frozen img targets), refs/ (comparator and package predictions), pylib/ (transformers 5.19, peft 0.21,
  fla 0.5.2; private, rsna env untouched), runs/, logs/, out/.
- Run anything with: PYTHONPATH=experiments/v2/oct14_vlm/pylib:experiments/v2/oct14_vlm/src:. conda run -n rsna python ...

## Deployment plan (Kaggle T4; nothing uploaded without the user's go)

- Weights: merge the LM LoRA into the base linear weights, keep the fine-tuned ViT and heads; ship fp16 visual + LM state (~4.3 GB
  incl. the tied embedding) as a dataset. The read-outs need only the embedding rows of the prompt's tokens, so the 248k x 2048
  embedding can be cut to the used ids (~1 GB saved).
- Code: src/vlm.py's model part + data windows from knee_infer's canonical slices (byte-identical to the training cache). Prompt
  token ids precomputed (finite vocabulary: 30 series headers, mm positions -200..+200, fixed text), so no tokenizers / regex wheels.
  transformers 5.19 + peft are pure Python (pylib/); Qwen3.5's gated delta rule runs the torch reference path on the T4
  (VLM_NO_FLA=1; fla's Triton kernels unverified on sm75).
- Precision: Qwen3.5 activations peak ~5.9k (fp16-safe; 10-08 check: bf16 vs fp16 corr 0.9992 on the untrained model); re-check
  the fine-tuned model (fp16 vs bf16 on gold-58, non-finite guard) before any package.
- Time: estimate = 5090 eval s / study x ~7 (Basin's T4 ratio) + the torch-path overhead (measured on the 5090 with VLM_NO_FLA=1);
  budget < ~30 min for 1,300 studies next to the 0.961 package's 262 min.

## G1 decision tree (pre-registered 02:21, before any GPU run)

Signals: training loss vs the prior-only loss (logged), val-subset trusted every ~1/4 epoch, train-fit (300 training studies) at the
epoch evals, logit sd, collapse diagnostics (pc12, corr MM-LM, MOA-LOA), ViT weight drift.
1. Broken (loss at / above the prior after warm-up, or constant predictions, or trusted < 0.80 at epoch 0.5): stop; halve all
   LRs (G1b); if that is flat too, fall back to the vision-only arm (V5) to localise the failure (ViT vs LM path).
2. Slow (learning, but trusted < 0.93 at epoch 1): G1r (report aux 0.2) and V28 (ViT warm start from the vision-only arm) next;
   a 2x LR arm only if train-fit is far below val (under-fitting) and drift is tiny.
3. Healthy (trusted >= 0.93 at epoch 1, rising): G2 = 3-4 epochs, the better of {plain, report aux} after G1r, 256 px; then 320 px.
4. Strong (single within ~0.003 of NFNet on trusted / lookat at G1's end): straight to the member / package reads, a second fold
   (0) and the T4 checks.
5. Every branch (added 11:27, before any GPU run; research.md batch 16: a fine-tuned vision encoder alone beat full MLLMs on knee
   OA radiographs): G1v = the vision-only arm (readout vis, no LM, same schedule) runs right after G1. Read: full VLM minus G1v on
   trusted / lookat_m per finding against the pooled seed sd. LM adds nothing (within 2 seed sd on both) -> the vision-only model
   is the candidate (far cheaper on the T4) and LM levers drop in priority; LM adds -> LM levers (attnmid, report aux) stay.

## Changes from the old-experiment reads (02:34; details and sources in old_experiments.md)

- Reading order of every eval: trusted / lookat_m / text first, then gold_nofc / _xe; gold-58 alone decides nothing (OA: gold
  favoured lse, the LB favoured text).
- Per-finding changes are read against the single-model seed sd pooled over the comparators' replicates (vlm_lib.seed_sd,
  out/seed_sd.txt; read_run.py section 5 prints VLM - NFNet, sd and z): one fold's sd is unstable (Baker's trusted 0.2e-3 on fold 2
  vs 3.4e-3 pooled), and low-positive findings swing (review 29's Baker's alarm was one lucky seed pair; corrected 02:45).
  V31 reads where the attn read-out puts its mass per finding (diagnostic only).
- After the best checkpoint: re-score at k_eval 64 / 96 / 128 (src/predict.py; F / C gained at 64, main flat from 96), then the V29
  clean-label stage (1 epoch, explicit cells only, F / C / synovitis all cells, no teacher) read against the checkpoint itself.
- Package entry only with leave-one-pipeline-out and a seed-matched yardstick (B3 + NFNet seed); a member must stay different
  (Spearman vs B3 / NFNet in read_run.py); per-column inclusion is allowed.
- T4: measure the fp16 time of the fold-2 model early (pad to fixed shapes: cuDNN rebuilds per new batch shape on Kaggle) and
  estimate with the measured local -> Kaggle ratios (Basin ~7x, OA stage 7.9x).
- Process: queue lines may carry inline '#' comments (stripped); train.py writes logs/NAME_fF.pid = the python PID (kill that, never
  the conda wrapper); 'do not close an idea on one early check' (OA, user).

## State at 20:39 on 10-14 (superseded by the SEALED section above)

All runs: fold 2, 1 seed, 1 epoch, WSD (constant LR after a 0.3-epoch warm-up), EMA 0.995; x1e-3 deltas.

| run | what | trusted raw / EMA | lookat_m raw | gold_nofc raw | Codex eff / syn (EMA) | s / study |
|---|---|---|---|---|---|---|
| g1w_q35 | full VLM, 32 whole windows | 0.858 / 0.868 | 0.831 | 0.802 | 0.966 / 0.973 | 0.30 |
| g1r_q35 | full VLM, 16 whole + 18 localiser crops | 0.861 / 0.868 | 0.831 | 0.778 | 0.964 / 0.979 | 0.43 (LM ckpt) |
| g1v_r | vision-only (no LM), g1r images | 0.866 / 0.860 | 0.822 | 0.776 | 0.964 / 0.973 | 0.14 |
| g1v_384 | vision-only, 384 px | 0.861 / 0.859 | 0.817 | 0.769 | 0.971 / 0.978 | 0.35 |
| g1v_lr3 / flat / emb | ViT + head x3 / flat ViT LR / embeddings x12 | 0.850 / 0.831 / 0.846 | 0.814 / 0.783 / 0.806 | 0.754 / 0.737 / 0.760 | ~0.96 / ~0.97 | 0.14 |
| NFNet single (ref) | 20-ep CNN | 0.975 | 0.954 | - | 0.953 / 0.960 | - |

- Per class (scripts/class_read.py; groups diffuse / focal / OA): the VLM beats NFNet on the diffuse image reads (Codex effusion
  +10, synovitis +18) and ties on diffuse text labels (full VLM only); focal -180..-260 and OA -77..-123 below NFNet.
- Package (0.961) + VLM as a 4th pipeline: overall -1.3..-1.9 trusted; Effusion image side positive in every checkpoint (Codex +2..+5,
  gold +2.5..+7.5 in 14 / 16) while text-label effusion is negative (out/effusion_member_table.txt): the main lead (user 19:27).
- LR / adaptation: 6 / 6 raises (all x3, LM x3, head x10, ViT x3, flat ViT, embeddings x12) lead on diffuse reads at 0.25-0.75 epoch
  and end behind by the epoch end. Hypothesis: gradient noise at 8 studies / step (test g1v_b32 queued).
- Queue for the GPU's return: g1v_mrg (merger x5), g1v_b32 (4x batch + x3 LR), g1v_c (vision-only +3 epochs, per-class read per
  epoch), g1r_c (full VLM +3 epochs, HOLD). Medical ViTs (OrthoFoundation-L, MRI-CORE) prepped by Backbone Sweep (not run).

## After G1 epoch 1 (18:05): input first, then LR

G1 at epoch 1 under-fits and loses most on small structures (ACL 0.81, LM 0.73, MM 0.79 vs NFNet 0.996 / 0.95 / 0.96; MM-LM logit
corr 0.86-0.93 = side mixing). LR raises (all x3; LM-side x3) were negative on raw weights by 0.5-0.75 epoch (ledger 17:38 / 18:04).
In-repo evidence points at the input: the Basin locality test (a ViT's pooled features carry no medial / lateral information; crops
before the encoder restore it), DINOv2-S losing LM / ACL / PF like our VLM, the OA / meniscus experts' zoomed localiser crops
(OA: 'window count is not the limit, detail is'). Next arm g1r_q35 = V66 localiser region crops + per-view prompt text (user ideas
17:40-17:49); then V63 flat ViT LR (g1f_q35) on the better input. Read every arm vs g1w at equal steps (sub-val 110 / 220 / 330,
full raw + EMA at 436) and paired train loss (scripts/loss_pair.py) where the inputs match.

## Post-G1 screen plan (pre-registered 11:42, before any GPU run; options from research.md, all CPU-tested)

Fold 2, the G1 schedule unless G1's curve asks for more (WSD since 16:45: a screen = one WSD run + a 175-step cooldown at the length
G1's branches pick; a longer screen resumes a WSD checkpoint with warmup=0 init=RUN@STEP resume_opt=True instead of restarting); each screen = one knob vs the G1 base; read with read_run.py (per-finding z
vs pooled seed sd) and pkg_value.py; 'gain' = trusted or lookat_m up by > 2 x the run seed sd (trusted 1.0e-3 / lookat 1.9e-3 pooled
mean) with the other not down by more than 1 sd.
0. Eval-only on every finished checkpoint (no training, minutes each): k_eval 64 / 96 / 128, offsets TTA (0, +-0.33), --img 288 / 320
   (V49), EMA vs raw (V45), --fp16 finite check (T4), --attn location (V31).
1. G1v (queued): vision-only. Branch: LM adds nothing (G1 - G1v within 2 sd on trusted and lookat_m) -> continue with the vision-only
   model (screens 3, 5 apply; 2, 4 are LM-only and drop); LM adds -> all blocks.
2. LM read-out / prompt block (LM only): attnmid (V33) = an extra head, read per head at eval; then preamble (V55) and shuffle (V48) as
   separate single-knob arms only if the base is healthy.
3. Input block: tok_pool 2 with k 128 / k_eval 256 (V35: 4x windows at equal LM length) vs base; frames 2 (V41: 6-slice windows).
4. LM training block (LM only): report aux 0.1-0.2 (V14), lora_plus 16 (V51); frozen LM lora_r 0 (V50) only if the LM adds.
5. Capacity / resolution: img 320 (or 384 per the bench), then the best setting for 3-4 epochs (G2) with EMA and every eval saved.
Order inside a block by expected gain per GPU hour; a block stops at the first clear negative of its strongest arm (prioritise-ROI rule).

## T4 cost estimate from FLOPs (12:03; to be replaced by measured times)

Per study (ViT 302M on 256 patches / window; LM ~1.5B non-embedding params + 6 full-attention layers): full VLM at k 64 (256 px) ~24
TFLOP -> 1.2-2.0 s on a T4 at 20-12 effective fp16 TFLOPS -> 26-44 min for 1,300 studies (one fold model); k 128 with tok_pool 2 ~27
TFLOP -> 30-50 min; vision-only (G1v) ~10 TFLOP -> 11-18 min. Not counted: the torch reference gated-delta path (no fla on sm75),
DICOM loading (shared with the package), cuDNN / shape warm-up. Repo calibration: the OA stage ran 7.9x slower on Kaggle than locally;
measure, do not trust this estimate. The 0.961 package already takes ~262 min.
