# Research code: improved detector and woodblock matching

The follow-on work behind Initials Studio.

| Script | Purpose |
|---|---|
| `common.py` | Paths, label loading, metrics (precision, recall, F1, AP50, AP75, AP50-95) and the cross-validation folds. |
| `train_detector.py` | Faster R-CNN (MobileNetV3-Large FPN, COCO-pretrained) fine-tuned on real labelled pages. Train on cross-validation folds (`fold k`) or on everything (`all --extra`). |
| `compare.py`, `evaluate_books.py`, `evaluate_reviewed.py` | Euclid 5-fold cross-validation, and tests on unseen books. |
| `classical_model.py`, `proposals.py` | Non-neural baseline: ink-blob proposals scored by a gradient-boosted classifier. |
| `run_baseline.py` | Score the dissertation's weight files on the same tests. |
| `make_*_labels.py` | How the labels in `labels/` were made and corrected. |
| `block_matcher.py`, `block_report.py`, `blocks_all.py`, `block_report_all.py` | Same-woodblock matching: frame-normalised crops, ECC alignment, ink-map correlation, all rotations; grouping with average linkage. |
| `fetch_iiif.py`, `prepare_other_books.py` | Download digitised books from IIIF libraries (e.g. the Munich digital collections) and prepare page images. |
| `make_paper_figures.py` | High-resolution figures for the paper. |

Results tables are in `results/`. The label files are in `labels/`: Pascal VOC XML per page, plus the JSON
review files.
