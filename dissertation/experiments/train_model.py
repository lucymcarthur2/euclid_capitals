import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# New size of your images
img_width, img_height = 3250, 4314  # original size
batch_size = 32  # Smaller batch size

# Function to create the model
def create_model():
    model = Sequential([
        Flatten(input_shape=(img_height, img_width, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(4)  # 4 units for the 4 bounding box coordinates
    ])
    model.compile(optimizer=Adam(), loss='mse')  # using mean squared error loss for regression
    return model

# Data generator
def data_generator(image_folder, xml_folder, batch_size):
    while True:
        batch_images = np.empty((batch_size, img_height, img_width, 3))
        batch_boxes = np.empty((batch_size, 4))
        for i in range(batch_size):
            img_idx = np.random.randint(0, 1000)
            image_path = os.path.join(image_folder, f'new_image_{img_idx}.jpg')
            xml_path = os.path.join(xml_folder, f'new_annotation_{img_idx}.xml')
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.  # convert color style and normalize
            tree = ET.parse(xml_path)
            root = tree.getroot()
            box = root.find('object').find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            batch_images[i] = img
            batch_boxes[i] = [xmin/img_width, ymin/img_height, xmax/img_width, ymax/img_height]  

        yield batch_images, batch_boxes

# Paths to images and XMLs
image_folder = r"data\mock_labels"
xml_folder = r"data\xmls"

# Create the generator for training data
train_gen = data_generator(image_folder, xml_folder, batch_size=batch_size)

# Create and train the model
model = create_model()
history = model.fit(train_gen, steps_per_epoch=100, epochs=10)

# Create the generator for validation data
val_gen = data_generator(image_folder, xml_folder, batch_size=batch_size)

# Evaluate the model on validation data
val_loss = model.evaluate(val_gen, steps=100)  # Assuming you want to use 50 steps for validation

print(f'Validation loss: {val_loss}')

# Plot training loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Generate some images and boxes
val_images, val_boxes = next(val_gen)

# Predict bounding boxes on the generated images
predictions = model.predict(val_images)

# Loop through the generated images
for i in range(len(val_images)):
    # Create a PIL Image object
    img = Image.fromarray((val_images[i] * 255).astype(np.uint8))
    
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    # Create a Rectangle patch (we multiply by the image size to rescale the coordinates)
    rect = patches.Rectangle((predictions[i][0] * img.size[0], predictions[i][1] * img.size[1]),
                             (predictions[i][2] - predictions[i][0]) * img.size[0], 
                             (predictions[i][3] - predictions[i][1]) * img.size[1], 
                             linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

# Print the coordinates of the bounding box
    print("Coordinates (xmin, ymin, xmax, ymax): ", predictions[i][0] * img.size[0], predictions[i][1] * img.size[1],
          predictions[i][2] * img.size[0], predictions[i][3] * img.size[1])

    plt.show()

