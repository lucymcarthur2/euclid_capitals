# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 21:51:29 2023

@author: Lucy McArthur
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def plot_image_with_boxes(img, boxes):
    """Plot image with bounding boxes."""
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # add bounding boxes
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

# Get the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (your object) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

# Load the model
model.load_state_dict(torch.load(r"data\models\model_weights_momentum_0.9.pth"))
model.eval()

# Your image directory path
TEST_IMAGE_DIR = r"data\ptolemy\pages"

# Extracted objects directory path
EXTRACTED_OBJECTS_DIR = r"data\ptolemy\diagrams"


# Add normalization to match your training dataset's preprocessing (if you had normalization during training)
transform = T.Compose([
    T.ToTensor(),
    # Uncomment the next line if you had normalization during training
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load all test images
test_image_paths = [os.path.join(TEST_IMAGE_DIR, img_name) for img_name in os.listdir(TEST_IMAGE_DIR)]

for image_path in test_image_paths:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.to(device)  # send the image to the device
    image_tensor = image_tensor.unsqueeze(0)  # add a batch dimension

    with torch.no_grad():
        prediction = model(image_tensor)  # forward pass

    # Print raw predictions to inspect them
    #print("Predicted Boxes:", prediction[0]['boxes'])
    #print("Scores:", prediction[0]['scores'])

    # Filter out predictions with confidence
    boxes = prediction[0]['boxes'][prediction[0]['scores'] > 0].detach().cpu().numpy()

    # visualize the output
    print(f"Displaying Predicted boxes for {image_path}:")
    plot_image_with_boxes(np.array(image), boxes)
    
    # Save the objects in the boxes as images
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        cropped_image.save(os.path.join(EXTRACTED_OBJECTS_DIR, f"extracted_object_{os.path.basename(image_path)}_{i}.png"))