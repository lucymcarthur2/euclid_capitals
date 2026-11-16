| book | model | pages | capitals | AP50 | AP50_CI | AP75 | AP50_95 | score_thr | precision | recall | F1 | TP | FP | FN | mean_IoU_matched | pages_perfect | sec_per_page |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Compilatio (1489) | Detector v2 (+3 extra training books) | 236 | 109 | 0.993 | 0.983-1.000 | 0.993 | 0.885 | 0.943 | 0.972 | 0.945 | 0.958 | 103 | 3 | 6 | 0.937 | 0.962 | 0.407 |
| Compilatio (1489) | Detector v1 (Euclid only) | 236 | 109 | 0.976 | 0.957-0.991 | 0.976 | 0.874 | 0.943 | 0.904 | 0.954 | 0.929 | 104 | 11 | 5 | 0.937 | 0.941 | 0.352 |
| Compilatio (1489) | Classical (proposals + GBM) | 236 | 109 | 0.756 | 0.663-0.846 | 0.315 | 0.402 | 0.382 | 0.833 | 0.642 | 0.725 | 70 | 14 | 39 | 0.798 | 0.826 | None |
| Compilatio (1489) | Original dissertation model (@ its own 0.99 cut-off) | 236 | 109 | 0.952 | 0.912-0.985 | 0.952 | 0.937 | 0.990 | 0.641 | 1.000 | 0.781 | 109 | 61 | 0 | 0.965 | 0.864 | 2.903 |
| Artes Orandi (1485) | Detector v2 (+3 extra training books) | 165 | 93 | 1.000 | 1.000-1.000 | 1.000 | 0.912 | 0.943 | 0.989 | 1.000 | 0.995 | 93 | 1 | 0 | 0.943 | 0.994 | 0.404 |
| Artes Orandi (1485) | Detector v1 (Euclid only) | 165 | 93 | 1.000 | 0.999-1.000 | 1.000 | 0.901 | 0.943 | 0.894 | 1.000 | 0.944 | 93 | 11 | 0 | 0.938 | 0.982 | 0.321 |
| Artes Orandi (1485) | Classical (proposals + GBM) | 165 | 93 | 0.910 | 0.855-0.969 | 0.725 | 0.664 | 0.382 | 0.851 | 0.860 | 0.856 | 80 | 14 | 13 | 0.870 | 0.891 | None |
| Artes Orandi (1485) | Original dissertation model (@ its own 0.99 cut-off) | 165 | 93 | 0.998 | 0.995-1.000 | 0.998 | 0.985 | 0.990 | 0.948 | 0.989 | 0.968 | 92 | 5 | 1 | 0.967 | 0.970 | 3.923 |
