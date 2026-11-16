# Dissertation code

Lucy McArthur, *Using Machine Learning Techniques to Analyse the Printing of Euclid's Elements*,
MSc dissertation, UCL, 2024. The paper's decorated-capital method (sections 2.2–2.4) is Chapters III and V.

Only the scripts with evidence that they produced the dissertation's reported results are included. The
project's earlier versions, abandoned approaches and test scripts (about 70) are not published. Each script
is kept as written, apart from one change: absolute file paths now point to a `data/` folder (scans and
generated datasets) and a `fonts/` folder.

**How each script was checked:** a script is marked **confirmed** when it either writes or reads files that
still exist from the project and match the dissertation (datasets, model files, figure sources), or its
parameters match the text exactly. It is marked **earlier version** when it is the closest surviving script
but the exact version used for the reported result was not kept.

## Chapter III: Extraction of decorated capitals (Faster R-CNN) → `ch3_capital_detection/`

| Script | Where in the dissertation | Evidence | Status |
|---|---|---|---|
| `pdf_to_jpgs.py` | III.C: "a short script is used to convert the PDF into JPEGs" | Converts a scanned PDF to page JPEGs; used for the other books | confirmed |
| `aligned_initial_capitals.py` | III.B.1 Dataset generation, Fig. 8 | Writes `aligned_<n>_capital*.jpg/.xml`, the 100 synthetic training pages and their boxes (`mock_labels/`, `xmls/`) | confirmed |
| `make_negative_samples.py` | III.B.1: "25 negatives are included" | Turns the 25 text-only pages in `negative_samples/` into training images with empty XML annotations | confirmed |
| `train_faster_r_cnn.py` | III.B.2 Model training | Fine-tunes the COCO-pretrained `fasterrcnn_resnet50_fpn` on the synthetic pages | earlier version (Jul 2023). The training run for the final model (Jul 2024) was not kept as a separate script. |
| `hyper_param_tuning.py` | III.B.3 Hyperparameter tuning, Figs 9–12, Table 1 | Batch size, learning rate, momentum and epoch sweep for the same detector | earlier version (Jun 2023). The weight files of the final sweep (`model_lr_*_momentum_*_epochs_*_batch_*.pth`, Jul 2024) survive, but the script that wrote them does not. |
| `run_RPN.py` | III.C Results, Fig. 13; extraction of capitals from *Elements*, *Compilatio*, *Artes orandi* | Runs the trained detector on page images and saves each detection (objectness ≥ 0.99) | confirmed (Jul 2024; points at a weights file since renamed) |

## Chapter IV: Categorisation of decorated capitals (CNN letter classifier) → `ch4_letter_classification/`

| Script | Where in the dissertation | Evidence | Status |
|---|---|---|---|
| `generate_capital_from_scratch_8.py` | IV.B.1 Dataset generation, Fig. 16 | Writes the synthetic font-letter training set (`june_manual_caps/`) that the final classifier was trained on | confirmed |
| `dynamic_background_removal.py` | IV.B.1, Fig. 17: dynamic preprocessing (adaptive threshold, contour noise filter) | Implements exactly the steps the text describes | confirmed |
| `may_20.py` | IV.B.2 Model training, Figs 18–19; training-set sizes for Fig. 20; confusion matrices | CNN with 64/128/256/512 filters and dropout 0.3/0.4, as in the text; trains on `june_manual_caps/`; saves `model_<size>_samples_per_class.h5` | confirmed |
| `plot_accuracy_vs_training_set.py` | Fig. 20 | Evaluates the saved 100/500/1000/1500/2000-per-class models | confirmed |
| `change_training_batch.py`, `plot_accuracy_vs_batch_size.py` | Fig. 21 | Trains and evaluates `model_batch_size_{16,32,64,128}.h5` | confirmed |
| `plot_accuracy_vs_lr.py` | Fig. 22 | Evaluates `model_lr_{0.0001,0.0005,0.001}.h5` | confirmed for the plot. The script that trained these models was not kept. |
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
