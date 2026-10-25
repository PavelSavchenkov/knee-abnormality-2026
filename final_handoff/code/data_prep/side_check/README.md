# Knee-side check by anatomy (2026-10-07)

The cache orients a series by its Laterality tag, else by the study side (study tag, else the majority of series sides: series tag, else the
sign of the image-centre x). Tags, SeriesDescription side words and geometry each disagree with the others on ~1-2% of series, and no fixed
order of trust is right (index 54: tag wrong, geometry right; most other conflicts: tag right, geometry wrong).

`side_check.py` trains two ResNet18 classifiers (160 px, 3 epochs, ~1 min on a shared GPU) on series whose orientation is trusted (|x| >= 40 mm
geometry agrees with the side used and with the tag):
- mirror (coronal/axial): was the slice flipped left-right? Held-out studies: 0 of 2,871 series flagged as stored; 99.7% of flipped copies caught.
- lateral (sagittal): is the slice on the lateral side? Held-out: 1.4% of series wrong in each direction at 0.5.
`summarize.py` turns per-series p_flip into decisions: the whole study is flipped when all its coronal/axial series (and all but at most one
series) are flipped; otherwise coronal/axial series flip at p_flip >= 0.8 and sagittal series at >= 0.9.

Result: 387 series in 100 studies flipped (49 whole studies, 51 partial: bilateral exams or mixed tags); 2 gold studies (bundle index 54 and 55)
are whole-study flips. Outputs: data/derived/series_side_final.csv (SeriesInstanceUID, side used, side_final, flip_final, p_flip, tag, geo, x),
out/series_side.csv (all scores), out/study_side.csv, out/report.txt, out/mirror.pt and out/lateral.pt (for use at inference).

Revision 2026-10-08: second signal from the Ideas session's nnU-Net Knee masks (anatomical meniscus naming on sagittal fluid-sensitive
series, copied here as nnunet_sides.csv from work/knee_seg/nnunet_knee/out_all/sides.csv). A sagittal series below 0.9 also flips when p_flip >= 0.5 and the masks say it is
reversed: +18 series, now 405 series in 101 studies (49 whole studies, 2 gold). Agreement with the masks: flips 79/81, non-flips
2,783/2,872. series_side_final.csv gains flip_source; the previous table is out/series_side_final_v1.csv.

