import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision
import numpy as np

def get_transform():
    custom_transforms = []
    custom_transforms.append(T.ToTensor())
    return T.Compose(custom_transforms)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "mock_labels"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "xmls"))))

    def __getitem__(self, idx):
        # Load image and bounding boxes
        img_path = os.path.join(self.root, "mock_labels", self.imgs[idx])
        ann_path = os.path.join(self.root, "xmls", self.annotations[idx])
    
        img = Image.open(img_path).convert("RGB")
    
        # Parse XML file
        tree = ET.parse(ann_path)
        boxes = []
        for obj in tree.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
    
        # Convert into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Ensure boxes tensor is 2D
        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)
    
        # Only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        
        # Calculate the area after ensuring boxes tensor is 2D
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
    
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
    
        if self.transforms is not None:
            img = self.transforms(img)
    
        return img, target




    def __len__(self):
        return len(self.imgs)

# The model
def get_model(num_classes):
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

if __name__ == '__main__':
    # Use GPU if it's available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Update these to your own directories
    data_dir = r"data"

    # Define the training dataset
    # Define the training dataset
    dataset = CustomDataset(data_dir, transforms=get_transform())

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Define the model
    num_classes = 2 # background and your object
    model = get_model(num_classes)
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # And a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Number of epochs
    num_epochs = 10

    # The training loop
    for epoch in range(num_epochs):
        model.train()
        i = 0    
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Iteration: {i}/{len(data_loader)}, Loss: {losses.item()}")

        # update the learning rate
        lr_scheduler.step()

        print(f"Epoch: {epoch}/{num_epochs}")
    # Save the model after training
    torch.save(model.state_dict(), "model_weights_pretrained.pth")
