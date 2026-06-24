# Phase 3 Grad-CAM Method Notes

- Method: Grad-CAM-style decision-targeted visual explanation.
- Hook layer: `visual.pooler_output`.
- Coordinate alignment: Qwen visual `pooler_output` is used by default because it is produced after window-token reverse indexing.
- Target mode: `do_not_proceed`.
- Default target: `Decision: Do not proceed`, fixed across clean and semantic perturbations for comparable drift scores.
- Artifact masking: `True` for uniform black letterbox rows/columns.
- Outputs for PDF: category grids, clean-vs-top5 comparison, change maps, bbox debug image, peak CSV, validity flags.
- Interpretation limit: this is Grad-CAM, not Attention Rollout; it is a post-hoc explanation for the selected decision target.

Recommended quality checks before using figures:
1. Inspect `bbox_debug_clean.png` to verify SOAR safety boxes.
2. Inspect `overlays/clean_overlay.png` and `cam_peak_debug.csv` to verify peak locations are not dominated by borders.
3. Run `--target-contrast-sanity` at least once and confirm `Proceed` and `Do not proceed` maps are not identical.
4. Check `cam_valid` and `cam_valid_reason` in `phase3_gradcam_metrics.csv`.
