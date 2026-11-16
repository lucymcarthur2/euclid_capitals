# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:49:40 2023

@author: Lucy McArthur
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array



np.random.seed(42)
tf.random.set_seed(42)

# Directory paths in Google Drive
TRAIN_DIR = r"data\manual_capitals"
TEST_DIR = r"data\extracted_letters"

# Image data generators for preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess images
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(64, 64),

    batch_size=16,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    shuffle=False)

# Class labels
class_labels = list(train_generator.class_indices.keys())

# Function to count class instances in a directory
def count_classes(directory, class_labels):
    counts = {label: 0 for label in class_labels}
    for subdir in os.listdir(directory):
        if subdir in class_labels:
            path = os.path.join(directory, subdir)
            counts[subdir] = len(os.listdir(path))
    return counts

# Print class distribution in training and test sets
train_class_counts = count_classes(TRAIN_DIR, class_labels)
test_class_counts = count_classes(TEST_DIR, class_labels)
print("Training set class distribution:", train_class_counts)
print("Test set class distribution:", test_class_counts)





# Building the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),  # Added a third convolutional layer
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),  # Increased the size of the dense layer
    Dropout(0.5),
    Dense(26, activation='softmax')
])

# Define early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min')

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model and save history, include callbacks
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Specify the path where you want to save the model
model_save_path = r"data\model_extracted_letters.h5"
  # Update with your actual path
model.save(model_save_path)


