import os
import numpy as np
import torchvision
import torch
import torch.utils.data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.transforms import functional as F

# Your image directory paths
IMAGE_DIR = r"data\circles_polygons_training_images"
XML_DIR = r"data\circles_polygons_training_xmls"
# Paths to your real-world images and annotations
REAL_IMAGE_DIR = r"C:\path\to\your\real\images"
REAL_XML_DIR = r"C:\path\to\your\real\xmls"

class_names = {'circle': 1, 'line': 2, 'polygon': 3}

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, IMAGE_DIR, XML_DIR, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, IMAGE_DIR))))
        self.xmls = list(sorted(os.listdir(os.path.join(root, XML_DIR))))
        first_image_path = os.path.join(self.root, IMAGE_DIR, self.imgs[0])
        first_image = Image.open(first_image_path)
        self.image_size = first_image.size
        print(f'CustomDataset initialized with {len(self.imgs)} images and {len(self.xmls)} xmls')

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, IMAGE_DIR, self.imgs[idx])
        xml_path = os.path.join(self.root, XML_DIR, self.xmls[idx])
        img = Image.open(img_path).convert("RGB")
        img = F.resize(img, self.image_size)
        target = self.parse_xml(ET.parse(xml_path).getroot())
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def parse_xml(self, node):
        target = {}
        boxes = []
        labels = []
        for object in node.findall('.//object'):
            name = object.find('name').text.strip()
            if name not in class_names:
                continue
            label = class_names[name]
            xmin = int(float(object.find('bndbox/xmin').text))
            ymin = int(float(object.find('bndbox/ymin').text))
            xmax = int(float(object.find('bndbox/xmax').text))
            ymax = int(float(object.find('bndbox/ymax').text))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.tensor([0], dtype=torch.int64)
        target["boxes"] = boxes
        target["labels"] = labels
        return target

def get_transforms(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def my_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

train_dataset = CustomDataset(root=".", IMAGE_DIR=IMAGE_DIR, XML_DIR=XML_DIR, transforms=get_transforms(train=True))
indices = torch.randperm(len(train_dataset)).tolist()
val_indices = indices[-50:]
train_indices = indices[:-50]
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=20, shuffle=True, num_workers=0, collate_fn=my_collate)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=20, shuffle=False, num_workers=0)

real_dataset = CustomDataset(root=".", IMAGE_DIR=REAL_IMAGE_DIR, XML_DIR=REAL_XML_DIR, transforms=get_transforms(train=True))
real_train_loader = torch.utils.data.DataLoader(real_dataset, batch_size=20, shuffle=True, num_workers=0, collate_fn=my_collate)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 4 # Three object types plus one for the background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
num_epochs = 50

# Main training loop
patience = 3
no_improvement = 0
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    i = 0
    epoch_loss = 0
    for imgs, annotations in train_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
        i += 1

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for imgs, annotations in val_loader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            val_loss += sum(loss for loss in loss_dict.values())
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement > patience:
            print("Early stopping")
            break

    lr_scheduler.step()

print('Main training completed')

# Reduce the learning rate for fine-tuning
for g in optimizer.param_groups:
    g['lr'] = 0.001  # Adjust as needed

# Fine-tuning loop
fine_tune_epochs = 10
for epoch in range(fine_tune_epochs):
    model.train()
    i = 0
    epoch_loss = 0
    for imgs, annotations in real_train_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
        i += 1

print('Fine-tuning completed')

# Save the fine-tuned model
torch.save(model.state_dict(), 'model_weights_shapes_finetuned.pth')
print('Fine-tuned model saved')
