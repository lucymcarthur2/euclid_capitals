"""Improved capital detector: COCO-pretrained Faster R-CNN with a MobileNetV3-Large FPN backbone,
fine-tuned on REAL labelled Euclid pages (not synthetic pastes), at 1024 px long side, on CPU.

Cross-validation: 5 contiguous page-block folds (common.kfold_ids). For fold k the model trains on
the other 4 folds and predicts fold k, so every page gets an out-of-fold prediction from a model
that never saw it. All hyper-parameters below are fixed up front (torchvision fine-tuning
defaults) - none are tuned on test pages.

usage:
  python train_detector.py fold 0          # train on folds != 0, predict fold 0
  python train_detector.py all             # train on all 280 pages (deployment model)
  python train_detector.py fold 0 --synthetic   # ablation: also add Diss/mock_labels synthetic pages
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from common import DISS, OUT, CACHE, HERE, load_pages, kfold_ids, read_voc

LONG = 1024
CFG = dict(epochs=12, batch=4, lr=0.01, momentum=0.9, weight_decay=1e-4, warmup_iters=50, seed=0)
torch.set_num_threads(4)


def load_rgb(path: Path, long_side: int = LONG):
    # key on the book folder too: several books keep their scans in a folder called "pages"
    f = CACHE / f"rgb_{path.parent.parent.name}_{path.parent.name}_{path.stem}_{long_side}.npy"
    with Image.open(path) as im:
        W, H = im.size
        s = long_side / max(W, H)
        if f.exists():
            return np.load(f), s
        im = im.convert("RGB").resize((round(W * s), round(H * s)), Image.BILINEAR)
    a = np.asarray(im)
    np.save(f, a)
    return a, s


class PageSet(torch.utils.data.Dataset):
    def __init__(self, items, train: bool):
        self.items = items  # list of (image_path, boxes_in_original_px)
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes = self.items[i]
        img, s = load_rgb(path)
        img = img.astype(np.float32) / 255.0
        b = boxes * s
        if self.train:
            img, b = augment(img, b)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        target = {"boxes": torch.as_tensor(b, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.ones(len(b), dtype=torch.int64)}
        return t, target


def augment(img: np.ndarray, b: np.ndarray):
    """Photometric changes that mimic other scans/copies, plus mild geometry."""
    h, w, _ = img.shape
    # contrast / brightness / gamma
    img = np.clip((img - 0.5) * random.uniform(0.7, 1.3) + 0.5 + random.uniform(-0.12, 0.12), 0, 1)
    img = img ** random.uniform(0.7, 1.4)
    # colour cast (paper tone varies between copies)
    img = np.clip(img * np.random.uniform(0.9, 1.1, size=3), 0, 1)
    if random.random() < 0.3:  # random greyscale
        img = np.repeat(img.mean(2, keepdims=True), 3, 2)
    if random.random() < 0.3:  # blur (focus / lower-res scans)
        import cv2
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.5))
    if random.random() < 0.3:  # sensor / paper noise
        img = np.clip(img + np.random.normal(0, random.uniform(0.01, 0.04), img.shape), 0, 1)
    if random.random() < 0.5:  # horizontal flip (recto/verso layouts mirror the margin side)
        img = img[:, ::-1]
        b = b.copy(); b[:, [0, 2]] = w - b[:, [2, 0]]
    if random.random() < 0.3:  # hue / saturation shift (hand-coloured capitals in other copies)
        import cv2
        hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-25, 25)) % 360
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.6, 1.8), 0, 1)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if random.random() < 0.6:  # zoom: other books print capitals at other sizes relative to the page.
        # (A plain resize would be undone by the model's internal resize to LONG, so zoom by
        #  pasting onto a padded canvas (f < 1) or cropping (f > 1) at a fixed canvas size.)
        img, b = zoom(img, b, random.uniform(0.7, 1.5))
    return img.astype(np.float32), b


def zoom(img: np.ndarray, b: np.ndarray, f: float):
    import cv2
    h, w, _ = img.shape
    small = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_LINEAR)
    sh, sw, _ = small.shape
    b = b * f
    if f < 1:  # zoom out: page on a paper-coloured canvas
        canvas = np.empty_like(img); canvas[:] = np.median(img.reshape(-1, 3), 0)
        ox, oy = random.randint(0, w - sw), random.randint(0, h - sh)
        canvas[oy:oy + sh, ox:ox + sw] = small
        return canvas, b + [ox, oy, ox, oy]
    # zoom in: random crop back to (h, w); keep boxes at least 60% inside, clipped
    ox, oy = random.randint(0, sw - w), random.randint(0, sh - h)
    crop = small[oy:oy + h, ox:ox + w]
    b = b - [ox, oy, ox, oy]
    cb = b.copy(); cb[:, [0, 2]] = cb[:, [0, 2]].clip(0, w); cb[:, [1, 3]] = cb[:, [1, 3]].clip(0, h)
    area = lambda x: (x[:, 2] - x[:, 0]).clip(0) * (x[:, 3] - x[:, 1]).clip(0)
    keep = (area(cb) >= 0.6 * area(b)) & (cb[:, 2] - cb[:, 0] >= 2) & (cb[:, 3] - cb[:, 1] >= 2)
    return np.ascontiguousarray(crop), cb[keep]


def build_model(pretrained: bool = True):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, min_size=LONG, max_size=LONG,
                                          box_detections_per_img=30, box_score_thresh=0.01)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    return m


def collate(b):
    return tuple(zip(*b))


def train(items, log_path: Path, synthetic_items=()):
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    ds = PageSet(list(items) + list(synthetic_items), train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=CFG["batch"], shuffle=True, collate_fn=collate, num_workers=0)
    model = build_model()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=CFG["lr"], momentum=CFG["momentum"], weight_decay=CFG["weight_decay"])
    total = CFG["epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / CFG["warmup_iters"]) * 0.5 * (1 + math.cos(math.pi * it / total)))
    model.train()
    it, skipped, t0 = 0, 0, time.time()
    with open(log_path, "w") as log:
        for ep in range(CFG["epochs"]):
            run = []
            for imgs, tgts in dl:
                loss = sum(model(list(imgs), list(tgts)).values())
                opt.zero_grad()
                if torch.isfinite(loss):  # skip the rare non-finite batch instead of corrupting the weights
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 10.0)
                    opt.step()
                    run.append(loss.item())
                else:
                    skipped += 1
                sched.step(); it += 1
            msg = f"epoch {ep+1}/{CFG['epochs']} loss {np.mean(run):.4f} skipped {skipped} elapsed {time.time()-t0:.0f}s"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
    return model


@torch.no_grad()
def predict(model, pages):
    model.eval()
    preds, times = {}, []
    for p in pages:
        img, s = load_rgb(p.image_path)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        t0 = time.perf_counter()
        o = model([t])[0]
        times.append(time.perf_counter() - t0)
        preds[p.page_id] = (o["boxes"].numpy() / s, o["scores"].numpy())
    return preds, float(np.mean(times))


def synthetic_items():
    xs = []
    for xml in sorted((DISS / "xmls").glob("*.xml")):
        boxes, w, h, fname = read_voc(xml)
        img = DISS / "mock_labels" / fname
        if img.exists():
            xs.append((img, boxes))
    return xs


if __name__ == "__main__":
    from common import save_preds
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fold", "all"])
    ap.add_argument("k", nargs="?", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--extra", action="store_true",
                    help="v2: also train on the extra labelled books (Ptolemy, Albumasar, Arithmetica). "
                         "They are never test books; their unlabelled woodcuts act as hard negatives.")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    if a.epochs:
        CFG["epochs"] = a.epochs

    pages = load_pages()
    tag = "mbv3" + ("_syn" if a.synthetic else "") + ("_v2" if a.extra else "")
    syn = synthetic_items() if a.synthetic else []
    if a.extra:
        from common import EXTRA_TRAIN_BOOKS
        for img_dir, ann_dir in EXTRA_TRAIN_BOOKS.values():
            syn += [(p.image_path, p.boxes) for p in load_pages(img_dir, ann_dir)]
    if a.mode == "fold":
        test_ids = set(kfold_ids(pages)[a.k])
        tr = [p for p in pages if p.page_id not in test_ids]
        te = [p for p in pages if p.page_id in test_ids]
        model = train([(p.image_path, p.boxes) for p in tr], OUT / f"log_{tag}_fold{a.k}.txt", syn)
        preds, spp = predict(model, te)
        save_preds(preds, OUT / f"preds_{tag}_fold{a.k}.json",
                   dict(model=tag, fold=a.k, cfg=CFG, sec_per_page=spp,
                        params=sum(p.numel() for p in model.parameters()), n_train=len(tr), n_syn=len(syn)))
        print(f"fold {a.k}: predicted {len(te)} pages, {spp:.2f}s/page")
    else:
        model = train([(p.image_path, p.boxes) for p in pages], OUT / f"log_{tag}_all.txt", syn)
        (HERE / "weights").mkdir(exist_ok=True)
        torch.save(model.state_dict(), HERE / "weights" / f"{tag}_all_pages.pth")
        print("saved", HERE / "weights" / f"{tag}_all_pages.pth")
