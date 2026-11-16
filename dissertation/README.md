# Dissertation code

Lucy McArthur, *Using Machine Learning Techniques to Analyse the Printing of Euclid's Elements*,
MSc dissertation, UCL, 2024. The paper's decorated-capital method (sections 2.2–2.4) is Chapters III and V.

Only the scripts with evidence that they produced the dissertation's reported results are included. The
project's earlier versions, abandoned approaches and test scripts (about 70) are not published. Each script
is kept as written, apart from one change: absolute file paths now point to a `data/` folder (scans and
generated datasets) and a `fonts/` folder.

**How each script was checked:** a script is marked **confirmed** when it either writes or reads files that
still exist from the project and match the dissertation (datasets, model files, figure sources), or its
parameters match the text exactly.

All the code behind the paper is here: synthetic pages, detector training with the final settings,
extraction, and ORB grouping. The only things not kept are the loops that re-ran training with one setting
changed for the dissertation's tuning plots (Figs 9–12 and 22); the README explains how to reproduce them.

## Chapter III: Extraction of decorated capitals (Faster R-CNN) → `ch3_capital_detection/`

| Script | Where in the dissertation | Evidence | Status |
|---|---|---|---|
| `pdf_to_jpgs.py` | III.C: "a short script is used to convert the PDF into JPEGs" | Converts a scanned PDF to page JPEGs; used for the other books | confirmed |
| `aligned_initial_capitals.py` | III.B.1 Dataset generation, Fig. 8 | Writes `aligned_<n>_capital*.jpg/.xml`, the 100 synthetic training pages and their boxes (`mock_labels/`, `xmls/`) | confirmed |
| `make_negative_samples.py` | III.B.1: "25 negatives are included" | Turns the 25 text-only pages in `negative_samples/` into training images with empty XML annotations | confirmed |
| `train_faster_r_cnn.py` | III.B.2 Model training; Table 1 | Fine-tunes the COCO-pretrained `fasterrcnn_resnet50_fpn` on the synthetic pages with exactly the Table 1 settings: batch size 4, learning rate 0.005, momentum 0.9, weight decay 0.0005, 10 epochs | confirmed (final configuration) |
| `hyper_param_tuning.py` | III.B.3: the first, coarser tuning experiment | Grid over momentum (0.9, 0.95) and weight decay at batch size 16. Not the sweep plotted in Figs 9–12. | earlier experiment |
| `run_RPN.py` | III.C Results, Fig. 13; extraction of capitals from *Elements*, *Compilatio*, *Artes orandi* | Runs the trained detector on page images and saves each detection (objectness ≥ 0.99) | confirmed (Jul 2024; points at a weights file since renamed) |

**Figs 9–12 (detector tuning):** each curve was produced by retraining `train_faster_r_cnn.py` with one
setting changed: batch size 2/4/8, learning rate 0.001/0.005/0.01, momentum 0.3/0.5/0.7/0.9, epochs 5/10.
The trained models from those runs survive (`model_lr_<lr>_momentum_<m>_epochs_<e>_batch_<b>.pth`,
30–31 July 2024), but the loop that ran them was not kept. To reproduce the figures, change those values
in `train_faster_r_cnn.py`.

## Chapter IV: Categorisation of decorated capitals (CNN letter classifier) → `ch4_letter_classification/`

| Script | Where in the dissertation | Evidence | Status |
|---|---|---|---|
| `generate_capital_from_scratch_8.py` | IV.B.1 Dataset generation, Fig. 16 | Writes the synthetic font-letter training set (`june_manual_caps/`) that the final classifier was trained on | confirmed |
| `dynamic_background_removal.py` | IV.B.1, Fig. 17: dynamic preprocessing (adaptive threshold, contour noise filter) | Implements exactly the steps the text describes | confirmed |
| `may_20.py` | IV.B.2 Model training, Figs 18–19; training-set sizes for Fig. 20; confusion matrices | CNN with 64/128/256/512 filters and dropout 0.3/0.4, as in the text; trains on `june_manual_caps/`; saves `model_<size>_samples_per_class.h5` | confirmed |
| `plot_accuracy_vs_training_set.py` | Fig. 20 | Evaluates the saved 100/500/1000/1500/2000-per-class models | confirmed |
| `change_training_batch.py`, `plot_accuracy_vs_batch_size.py` | Fig. 21 | Trains and evaluates `model_batch_size_{16,32,64,128}.h5` | confirmed |
| `plot_accuracy_vs_lr.py` | Fig. 22 | Evaluates `model_lr_{0.0001,0.0005,0.001}.h5` | confirmed. The three models were trained with `may_20.py`, whose learning rate (0.0001) was changed to 0.0005 and 0.001 for the other two runs. |
| `vary_optimizer.py`, `plot_accuracy_vs_optimizer.py` | Fig. 23 | Trains and evaluates the RMSProp, Adam and SGD models | confirmed |
| `plot_activation.py` | Figs 24–25 | Activation maps of the final model (`best_1000_images.h5`) for the "L" from *Elements* p. 188 | confirmed |
| `test_cnn.py` | IV.C results (per-class accuracy, Table 5) | Per-class accuracy of the final model on the extracted capitals | likely |
| `pytesseract.py` | Fig. 30: Tesseract OCR comparison | Per-class OCR accuracy on the extracted capitals | confirmed |

## Chapter V: Grouping by woodblock, and the GUI appendix → `ch5_woodblock_grouping_gui/`

| Script | Where in the dissertation | Evidence | Status |
|---|---|---|---|
| `tkinter_gui.py` | Chapter V (Figs 31–36, Table 6) and the GUI figures (Figs 43–47) | ORB matching with Lowe ratio 0.73 and gradient threshold 0.3 (as in the text), flipping, grouping; the GUI tabs shown in Figs 43–47 | confirmed |

## Chapter VI: Analysis of complex diagrams → `ch6_diagrams/`

| Script | Where in the dissertation | Evidence | Status |
|---|---|---|---|
| `diagram_training_cnn.py` | VI.B, Fig. 37 | Trains the diagram detector on the 100 synthetic diagram pages (`images_diagrams/`, `xmls_diagrams/`); writes `diagrams_weights.pth` | confirmed. The script that generated the synthetic diagram pages was not kept. |
| `test_model_diagrams.py` | Figs 38–42 | Runs the diagram detector and writes `extracted_diagrams/` | confirmed |
| `feature_plotting.py` | Fig. 41 and the diagram comparisons | ORB matches, overlay and difference between the extracted diagrams of pp. 241 and 242 | confirmed |

## Requirements

Python 3.10+, PyTorch and torchvision, TensorFlow/Keras, OpenCV, Pillow, NumPy, Matplotlib, scikit-learn,
PyMuPDF, pytesseract.
