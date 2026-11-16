# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 22:40:34 2023

@author: Lucy McArthur
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

np.random.seed(42)
tf.random.set_seed(42)

# Directory paths
TRAIN_DIR = r"data\manual_capitals"
TEST_DIR = r"data\manual_capitals"

# Image data generators for preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
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

# Load the pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Building the model with VGGNet
model = Sequential([
    base_model,
    InputLayer(input_shape=(128, 128, 3)),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Flatten(),
    
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    
    Dense(1024, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    
    Dense(26, activation='softmax')  # Assuming 26 classes for letters
])

# Define callbacks
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

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# Predictions and evaluation
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(true_classes, predicted_classes))

class_labels = list(test_generator.class_indices.keys())

# Calculate accuracy per class and overall accuracy
class_correct = Counter()
class_total = Counter()

for true, pred in zip(true_classes, predicted_classes):
    class_total[class_labels[true]] += 1
    if true == pred:
        class_correct[class_labels[true]] += 1

total_correct = sum(class_correct.values())
total_predictions = len(true_classes)
overall_accuracy = (total_correct / total_predictions) * 100

# Print accuracy per class
for class_label in class_labels:
    if class_total[class_label] > 0:
        accuracy = (class_correct[class_label] / class_total[class_label]) * 100
        print(f"Class {class_label}: Correctly Predicted {class_correct[class_label]} out of {class_total[class_label]} ({accuracy:.2f}%)")
    else:
        print(f"Class {class_label}: No test images found")

# Print overall accuracy
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

# Reset the test_generator before going through the images
test_generator.reset()

# Get the indices of the top 3 predictions for each image
top_3_pred_indices = np.argsort(-predictions, axis=1)[:, :3]

# Get the probabilities of the top 3 predictions for each image
top_3_pred_probabilities = np.take_along_axis(predictions, top_3_pred_indices, axis=1)



# Display a limited number of images and their predictions
num_images_to_display = 507  # Set the number of images you want to display

for i in range(num_images_to_display):
    # Load the image
    img_path = os.path.join(TEST_DIR, test_generator.filenames[i])
    img = load_img(img_path, target_size=(64, 64))
    img_array = img_to_array(img)

    # Get the top 3 predictions and their probabilities
    top_predictions = top_3_pred_indices[i]
    top_probabilities = top_3_pred_probabilities[i]

    # Plot the image and the top 3 predictions with probabilities
    plt.figure(figsize=(5, 5))
    plt.imshow(img_array / 255.)
    plt.title(f"Top Predictions:\n1. {class_labels[top_predictions[0]]} ({top_probabilities[0]:.2f})\n"
              f"2. {class_labels[top_predictions[1]]} ({top_probabilities[1]:.2f})\n"
              f"3. {class_labels[top_predictions[2]]} ({top_probabilities[2]:.2f})")
    plt.axis('off')
    plt.show()