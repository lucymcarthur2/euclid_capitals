# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:01:03 2023

@author: Lucy McArthur
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:02:16 2023

@author: Lucy McArthur
"""
import shutil
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def organize_predicted_images(predicted_classes, test_generator, source_test_dir, target_dir):
    """
    Organize images into predicted class folders within the target directory.
    Copy images from source_test_dir that have corresponding predictions.
    
    :param predicted_classes: Array of predicted class indices.
    :param test_generator: The generator for test data.
    :param source_test_dir: Directory path of the source test images.
    :param target_dir: Base directory path where subfolders will be created.
    """
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Reset the generator
    test_generator.reset()

    for i in range(len(test_generator.filenames)):
        # Class label of the predicted class
        # Map class indices to labels
        index_to_label_map = {v: k for k, v in train_generator.class_indices.items()}

        class_index = predicted_classes[i]
        class_label = index_to_label_map[class_index]

        # Create a directory for the predicted class if it doesn't exist
        class_dir = os.path.join(target_dir, class_label)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Source path of the image in source test directory
        src_filename = os.path.basename(test_generator.filenames[i])
        src_path = os.path.join(source_test_dir, src_filename)

        # Destination path in the target directory
        dst_path = os.path.join(class_dir, src_filename)

        # Copy the image to the corresponding class directory
        shutil.copy(src_path, dst_path)
        

# Define the model architecture
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1, zoom_range=0.1, horizontal_flip=False, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

# Directory paths
TRAIN_DIR = r"data\manual_capitals"
TEST_DIR = r"data\extracted_letters"

# Data generators
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(64, 64), batch_size=16, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=(64, 64), batch_size=8, class_mode='categorical', shuffle=False)

# Get class labels
class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min')

# Initialize a list to hold indices of correctly predicted samples for each run
correct_prediction_indices_per_run = []

# Train the model multiple times
num_runs = 10
for run in range(num_runs):
    print(f"Training run {run+1}/{num_runs}")
    model = create_model(num_classes)
    history = model.fit(train_generator, epochs=15, validation_data=test_generator, callbacks=[early_stopping, reduce_lr])

    # Predict on the test set
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    correct_indices = set(np.where(predicted_classes == test_generator.classes)[0])
    correct_prediction_indices_per_run.append(correct_indices)

# Aggregate correct predictions
aggregated_correct_indices = set.union(*correct_prediction_indices_per_run)

# Calculate and print accuracy per class
accuracy_per_class = {}
for class_index in np.unique(test_generator.classes):
    class_accuracy = np.mean([index in aggregated_correct_indices for index in np.where(test_generator.classes == class_index)[0]])
    class_label = class_labels[class_index]
    accuracy_per_class[class_label] = class_accuracy

for class_label, accuracy in accuracy_per_class.items():
    print(f"Accuracy for class {class_label}: {accuracy:.2f}")

# Calculate and print overall accuracy
overall_accuracy = len(aggregated_correct_indices) / len(test_generator.classes)
print(f"Overall Accuracy: {overall_accuracy:.2f}")

# Optional: Plot the training and validation accuracy and loss from the last run
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Directories for source test images and for organizing predicted images
SOURCE_TEST_DIR = r"data\initial_capitals_improved\capitals\test_capitals"
TARGET_DIR = r"data\initial_capitals_improved\capitals\trained_capitals"

# Organize images based on aggregated predictions
organize_predicted_images(predicted_classes, test_generator, SOURCE_TEST_DIR, TARGET_DIR)