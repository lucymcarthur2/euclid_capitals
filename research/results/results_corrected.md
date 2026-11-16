| model | pages | capitals | AP50 | AP75 | AP50_95 | score_thr | precision | recall | F1 | TP | FP | FN | mean_IoU_matched | pages_perfect | mean_abs_count_error | sec_per_page | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A  baseline (dissertation weights) | 280 | 510 | 1.000 | 1.000 | 0.990 | per-fold 1.00-1.00 | 0.996 | 0.990 | 0.993 | 505 | 2 | 5 | 0.968 | 0.979 | 0.025 | 5.500 | 41.4M |
| A  baseline @ dissertation cut-off 0.99 | 280 | 510 | 1.000 | 1.000 | 0.990 | 0.990 | 0.971 | 1.000 | 0.986 | 510 | 15 | 0 | 0.968 | 0.954 | 0.054 | 5.500 | 41.4M |
| B  classical proposals + GBM | 280 | 510 | 0.902 | 0.351 | 0.433 | per-fold 0.05-0.86 | 0.924 | 0.888 | 0.906 | 453 | 37 | 57 | 0.763 | 0.793 | 0.200 | 0.241 | None |
| C  MobileNetV3-FRCNN v2, real pages + 3 extra books (CV) | 280 | 510 | 0.990 | 0.990 | 0.907 | per-fold 0.49-0.91 | 1.000 | 0.994 | 0.997 | 507 | 0 | 3 | 0.945 | 0.996 | 0.011 | 0.368 | 18.9M |
