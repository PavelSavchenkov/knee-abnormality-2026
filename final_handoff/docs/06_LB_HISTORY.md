# 6. Public LB history (key submissions)

Public LB = mean of 12 per-class AUCs on the public part of the hidden test (~1,150-1,299 test studies in total). "Above X" =
ranked above X at hidden full precision (Kaggle's server-side sort; see `04_INFERENCE_AND_KAGGLE.md` 4.5).

| date | submission (ref) | public | note |
|---|---|---|---|
| 10-07 | effb0, 1 fold / 5 folds / + Synovitis-Effusion logit mix | 0.931 / 0.938 / 0.940 | baseline R2 |
| 10-08 | `oct8_imghead3_5f` | 0.944 | Synovitis from the Codex image-read head |
| 10-09 | 5 single-class probes of the 0.944 package | LOA 0.884, Fracture 0.908, PF 0.920, LM 0.944, Baker's 0.968 (+-0.006) | weakest classes on the test |
| 10-10 | `oct10_3pipe_3f` | 0.948 | effb0 + NFNet-L0 + RadImageNet R50, folds 0-2 |
| 10-11 | `oct10_3x3_synprob05` | 0.952 | image-read Synovitis heads on the 3 pipelines |
| 10-11 | `oct11_oa_3pipe` | 0.955 | OA compartment experts |
| 10-12 | `oct12_syn_g` (57112379) | 0.955, above `oct11_oa_3pipe` | Synovitis G |
| 10-13 | `oct12_global_best` | 0.957 | + meniscus heads + bone experts |
| 10-13 | inner-world pseudo-label probes B0 / B0_xF / K6r / K12fkd | 0.943 / 0.945 / 0.946 / 0.946 | PL students: +0.001, not separable |
| 10-14 | `oct13_f5_win` (57154196) | **0.961** | folds 0-4 + window / crop fusion + Effusion image head; 262 min |
| 10-15 | all-data 1 / 2 / 3 / 5 seeds (57213373 / 57225427 / 57227263 / 57235072) | 0.960 / 0.961 / 0.962 / **0.962** | all-data retrain of the 0.961 configs |
| 10-15 | folds + all-data (57239412, 108 models) | 0.962 | 477 min |
| 10-16 | LOA k_eval 64 splice (57273283) | 0.961, above 0.961 | on all-data s5 (57283796): below s5, so not robust |
| 10-16 | FS-only Synovitis image heads (57279368) | 0.961, BELOW 0.961 | gold gain did not transfer |
| 10-17 | ConvNeXt Synovitis crop slots (57283240) | 0.961, above 0.961 | kept |
| 10-17 | silL Lateral OA member (57284893) | 0.961, above 0.961 | LOA follows gold |
| 10-17 | s5 + Q6 MCL <- Contusion (57307950) / s5 + Q8 LM <- ACL (57307925) | **0.963** / 0.963 | post-hoc de-couplings |
| 10-17 | MCL zoom member w5b (57315944) | 0.962 (fold package) | |
| 10-18 | MCL zoom + Q6 (57325143) | 0.962, above the zoom probe | |
| 10-18 | ACL notch zoom z1 (57324534) / + b3g ACL Q16 (57346196) | 0.961, above 0.961 / above z1 | |
| 10-18 | LOA Q15 (oasep + oasep_nf + silsep) (57352132) | 0.962, above silL | |
| 10-18 | MCL Q21 stack (SD heads + zoom + b3g + Q6 re-fit) (57352127) | kept in v2 | |
| 10-19 | Fracture fsmain splice (57399358), PF Q20 (57399364), Baker's allpw Q25 (57399558) | not kept | switches OFF in v2 |
| 10-20 | **fold stack v2** `oct20_final_v2` (57416173) | **0.964** | every LB-confirmed column; 205 min; team rank 31 / 6,249 (last of 13 teams at 0.964) |
| 10-21 | Contusion all-4 v3c (57455847) | 0.964, above v2 | Opus image + report targets |
| 10-21 | EF3 Effusion crop slots (57455943) | above v2 | Codex + Qwen + Opus effusion target |
| 10-21 | Fracture v3c_fs (57475191) | 0.964, top of our 0.964 band | |
| 10-21 | Effusion form (c) (57479153) | 0.963 | lost |
| 10-21 | Contusion sxm (57480087) | 0.964, small win (~+0.00013) | 499 min |
| 10-22 | **all-data safe3** (57497980, 81 models) | **0.965** | the merged fold form on all data, 3 seeds; 190.5 min |
| 10-22 | all-data final118 (57497935) | timed out | slow Kaggle machine |
| 10-22 | **P1** plain 93 (57517691) | **0.965**, our top by hidden order | **final 1**; 176 min |
| 10-22 | **P3** plain + PL blend 117 (57520062) | **0.965**, below P1, above safe3 | **final 2**; 315 min |
| 10-22 | P5 column-routed + plain extras 130 (57521751) | 0.964 | 526 min |
| 10-22 | P2 PL-by-group 87 (57520594), P4 column-routed 120 (57520478) | timed out | slow Kaggle machines |

Context: on 10-16 the public top was 0.968 (we were 26th of 6,095 at 0.962); public notebooks clustered at 0.954-0.958.
