import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import os
import shutil

# Define the model
class FineTunedResNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet, self).__init__()

        # Load a pre-trained ResNet-18
        self.resnet = models.resnet18(pretrained=True)

        # Freeze the layers of ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the output layer of ResNet-18
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

# Define your transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define your training data
train_data = datasets.ImageFolder(r"data\initial_capitals\specific\train augmented", transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Create a dictionary to map indices to class labels
idx_to_class = {i: c for c, i in train_data.class_to_idx.items()}

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
num_classes = 15
model = FineTunedResNet(num_classes=num_classes)
model.to(device)

# Define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Perform backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'fine_tuned_resnet.pth')

# Now let's load the model and make predictions
model.load_state_dict(torch.load('fine_tuned_resnet.pth'))
model.eval()

# Define the path to your test data
path = r"data\initial_capitals\specific\test"
# Define the path to your new directory
new_path = r"data\initial_capitals\specific\test_clustered"

# Load images and make predictions
for img_name in os.listdir(path):
    img_path = os.path.join(path, img_name)
    
    img = Image.open(img_path).convert("RGB")
    
    # Apply the transform to the image
    img = transform(img)
    
    # Add an extra batch dimension since pytorch treats all images as batches
    img = img.unsqueeze(0).to(device)
    
    # Forward pass the image through the model
    outputs = model(img)
    
    # Get the index of the highest probability class
    _, preds = torch.max(outputs, 1)
    predicted_class = idx_to_class[preds.item()]
    
    # Print the predicted class
    print(f"Predicted class for image {img_name}: {predicted_class}")
    
    # Create a new directory for this class, if it doesn't exist already
    new_dir = os.path.join(new_path, predicted_class)
    os.makedirs(new_dir, exist_ok=True)
    
    # Copy the image to the new directory
    shutil.copy(img_path, new_dir)

print('Finished Prediction and Clustering')
