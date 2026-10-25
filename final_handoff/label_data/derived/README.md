# Derived data assets (moved here from work/ on 2026-10-08)

| File | Content | Built by |
|---|---|---|
| `series_side_final.csv` | Knee side per series after an anatomy check. Index SeriesInstanceUID; StudyInstanceUID, plane, `side` (side used by the slice cache), `side_final`, `flip_final`, `flip_source`, `p_flip`, `tag`, `geo`, `x`, `other_knee`. 405 series in 101 studies are flipped (49 whole studies, 2 gold). | `data/build/side_check/` |
| `slice_boxes.parquet` | Knee structure boxes per slice: StudyInstanceUID, SeriesInstanceUID, plane, slice, `row` (slice cache row), `structure`, `y0`, `y1`, `x0`, `x1` (pixels in the `data/cache/fov160_px400` grid, `common.cache`), `frac` (coverage weight). nnU-Net Knee (aagatti/nnunet_knee, MIT) segments one sagittal fat-suppressed series per study; its boxes are projected into every series by patient coordinates. 4,118 of 4,407 studies; good on sagittal PD-FS. | Ideas session |
| `synovitis_labels_codex.csv` | Blind Codex reads of the images (not the reports) for Effusion and Synovitis on non-gold studies: probabilities, `*_blockrank` (ranks within a read block, cancel within-session drift), `folder`, `sampling` (`queue` / `batch`: which selection scheme picked the study), `stratum`, `cell`, `weight` (post-stratified: one value per (stratum, cell) group; use it for anything population-level). Re-exported while labelling continues. | Image Labeling session |

Old paths `work/knee_seg/nnunet_knee/out_all/slice_boxes.parquet` and `work/image_labeling/out/synovitis_labels_codex.csv` are
symlinks to these files, so older code keeps working. Read and write the files here.
