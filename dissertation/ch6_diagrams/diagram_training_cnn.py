# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 21:52:38 2023

@author: Lucy McArthur
"""

import os
import numpy as np
import torchvision
import torch
import torch.utils.data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image
import xml.etree.ElementTree as ET

# Your image directory paths
IMAGE_DIR = r"data\images_diagrams"
XML_DIR = r"data\xmls_diagrams"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, IMAGE_DIR))))
        self.xmls = list(sorted(os.listdir(os.path.join(root, XML_DIR))))
        print(f'CustomDataset initialized with {len(self.imgs)} images and {len(self.xmls)} xmls')

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, IMAGE_DIR, self.imgs[idx])
        xml_path = os.path.join(self.root, XML_DIR, self.xmls[idx])
        #print(f'Opening image {img_path}')
        img = Image.open(img_path).convert("RGB")
        #print(f'Parsing XML {xml_path}')
        target = self.parse_xml(ET.parse(xml_path).getroot())
        if self.transforms is not None:
            img = self.transforms(img)  # Only apply transforms to the image
        return img, target


    def __len__(self):
        return len(self.imgs)

    def parse_xml(self, node):
        target = {}
        boxes = []
        labels = []
        for object in node.findall('.//object'):
            name = object.find('name').text
            xmin = int(float(object.find('bndbox/xmin').text))
            ymin = int(float(object.find('bndbox/ymin').text))
            xmax = int(float(object.find('bndbox/xmax').text))
            ymax = int(float(object.find('bndbox/ymax').text))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # object class
        if len(boxes) == 0:  # Negative sample, no objects present
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # no boxes
            labels = torch.tensor([0], dtype=torch.int64)  # background class
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        target["boxes"] = boxes
        target["labels"] = labels
        return target

def get_transforms():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def my_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)

    return images, targets





dataset = CustomDataset(root=".", transforms=get_transforms())
indices = torch.randperm(len(dataset)).tolist()
indices = indices[:100]  # limit to 10 items for testing
dataset = torch.utils.data.Subset(dataset, indices[:-2])
dataset_test = torch.utils.data.Subset(dataset, indices[-2:])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=my_collate)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=0)

print('Data loaders prepared')

# Get the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (your object) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

print(f'Model initialized and moved to {device}')

# Train the model
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 5

for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1} of {num_epochs}')

    # Training
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        print(f'Starting training batch {i+1}')
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        #print(f'Data moved to {device}')
        loss_dict = model(images, targets)
        #print(f'Loss calculated: {loss_dict}')
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        #print(f'Zeroed gradients')
        losses.backward()
        #print(f'Backpropagation done')
        optimizer.step()
        #print(f'Optimization step done')

        if i % 50 == 0:
            print(f'Training batch {i+1}, loss: {losses.item()}')




print('Training completed')


# Save the model
torch.save(model.state_dict(), 'diagrams_weights.pth')
print('Model saved')

