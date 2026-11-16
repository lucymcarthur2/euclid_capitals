# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:23:27 2023

@author: Lucy McArthur
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Function to create the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation=
               'relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model

# Size of your images
img_width, img_height = 812, 1078
batch_size = 32

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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
            tree = ET.parse(xml_path)
            root = tree.getroot()
            box = root.find('object').find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            batch_images[i] = img
            batch_boxes[i] = [xmin, ymin, xmax, ymax]

        yield batch_images, batch_boxes

# Paths to images and XMLs
image_folder = r"data\mock_labels"
xml_folder = r"data\xmls"

# Create the generator for training data
train_gen = data_generator(image_folder, xml_folder, batch_size=batch_size)

# Create and train the model
model = create_model()
history = model.fit(train_gen, steps_per_epoch=10, epochs=1)

# Function to display images and bounding boxes
def display_images_and_boxes(images, boxes):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    for i, (img, box) in enumerate(zip(images, boxes)):
        if i == 4:
            break
        axs[i].imshow(img)
        rect = patches.Rectangle((box[0], box[1]),
                                 (box[2] - box[0]),
                                 (box[3] - box[1]),
                                 linewidth=1, edgecolor='r', facecolor='none')
        axs[i].add_patch(rect)
    plt.show()

# Function to load test images
def load_test_images(image_folder, num_images):
    images = np.empty((num_images, img_height, img_width, 3))
    for i in range(num_images):
        image_path = os.path.join(image_folder, f'new_image_test_{i}.jpg')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        images[i] = img
    return images

# Load some test images
num_test_images = 3
image_folder_test = r"data\test_images"
test_images = load_test_images(image_folder_test, num_test_images)

# Predict bounding boxes on the test images
predicted_boxes = model.predict(test_images)

# Scale the predicted boxes back up to the original size of the image
predicted_boxes_scaled = predicted_boxes * np.array([img_width, img_height, img_width, img_height])

# Display the test images and predicted boxes
display_images_and_boxes(test_images, predicted_boxes_scaled)
