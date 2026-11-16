# Dissertation code

Lucy McArthur, *Using Machine Learning Techniques to Analyse the Printing of Euclid's Elements*,
MSc dissertation, UCL, 2024. This is the method described in the paper's sections 2.2–2.4 for decorated capitals.

The scripts are kept as they were written, apart from one change: absolute file paths now point to a
`data/` folder (put the scans and generated datasets there) and a `fonts/` folder (TrueType fonts for
synthetic letters).

## The pipeline (`pipeline/`)

| Step | Script | What it does |
|---|---|---|
| 1 | `pdf_to_jpgs.py` | Convert a scanned PDF to page images. |
| 2 | `aligned_initial_capitals.py` | Make synthetic training pages: real capitals pasted at the left of the text block on blank pages from *Elements*, with Pascal VOC XML boxes. |
| 2 | `generate_capital_from_scratch_8.py` | Make fully synthetic capitals: modern-font letters on historical decorative backgrounds. |
| 2 | `make_negative_samples.py` | Make text-only pages with empty annotations, as negative samples. |
| 3 | `train_faster_r_cnn.py` | Fine-tune Faster R-CNN (ResNet-50 + FPN, initialised from COCO-pretrained torchvision weights) to detect capitals. |
| 3 | `hyper_param_tuning.py` | Sweep batch size, learning rate, momentum and epochs (Table 1 of the paper). |
| 4 | `run_RPN.py` | Detect capitals on real pages and save each one as an image (objectness ≥ 0.99). |
| 5 | `dynamic_background_removal.py` | Adaptive binarisation and noise removal of the extracted capitals. |
| 6 | `tkinter_gui.py` | Desktop GUI for the whole process: detection, then grouping impressions by woodblock with ORB keypoints (Lowe ratio 0.73; at least 20 good matches, over 80% of match lines near-horizontal). |
| — | `train_letter_classification.py`, `pretrained_letter_classification.py` | CNN letter classifiers (the earlier approach, before grouping by woodblock directly). |
| — | `diagram_training_cnn.py`, `test_model_diagrams.py`, `feature_plotting.py` | Geometric-diagram detection and ORB match plots. |

## `experiments/`

Everything else written during the project: earlier versions, ablations (training-set size, batch size,
optimiser), plotting scripts and test code. These are kept for completeness; they are not needed to run
the pipeline.

## Requirements

Python 3.10+, PyTorch and torchvision, OpenCV, Pillow, NumPy, Matplotlib, scikit-learn, PyMuPDF.
The letter classifiers also need TensorFlow/Keras.
