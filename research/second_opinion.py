"""Second opinion for a hand-labelled TRAINING book: the classical model (independent of the
detector) proposes regions; any it scores >= MIN that overlap no label are rendered for a look.

usage: python second_opinion.py albumasar|arithmetica [min_score]
"""
import sys

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from common import load_pages, iou_matrix, save_preds, OUT, EXTRA_TRAIN_BOOKS
from classical_model import features, nms
from proposals import propose
from train_detector import load_rgb

book = sys.argv[1]
min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

# train the classical scorer on all Euclid pages (as in classical_model.py)
X, y = [], []
for p in load_pages():
    rgb, s = load_rgb(p.image_path)
    c = propose(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if len(c) == 0:
        continue
    X.append(features(rgb, c))
    y.append(iou_matrix(c / s, p.boxes).max(1) >= 0.5 if len(p.boxes) else np.zeros(len(c), bool))
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(np.concatenate(X), np.concatenate(y))

out = {}
for p in load_pages(*EXTRA_TRAIN_BOOKS[book]):
    rgb, s = load_rgb(p.image_path)
    c = propose(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if len(c) == 0:
        continue
    b, sc = nms(c / s, clf.predict_proba(features(rgb, c))[:, 1])
    keep = sc >= min_score
    if len(p.boxes):
        keep &= iou_matrix(b, p.boxes).max(1) < 0.3
    if keep.any():
        out[p.page_id] = (b[keep], sc[keep])
save_preds(out, OUT / f"preds_secondopinion_{book}.json")
print(book, "unlabelled candidates:", sum(len(v[1]) for v in out.values()))
