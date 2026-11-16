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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import mode

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)



        
# Define the model architecture in a function to ensure it can be reset every run
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

# Load and prep

# Directory paths
TRAIN_DIR = r"data\manual_capitals"
TEST_DIR = r"data\extracted_letters"


train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(64, 64), batch_size=16, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=(64, 64), batch_size=8, class_mode='categorical', shuffle=False)

# Get class labels
class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)

# Define early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min')


# Initialize a list to hold predictions from each run for each sample
all_run_predictions = []

# Train the model multiple times
num_runs = 5  # Set the number of runs
for run in range(num_runs):
    print(f"Training run {run+1}/{num_runs}")
    model = create_model(num_classes)#
    
    history = model.fit(train_generator, epochs=15, validation_data=test_generator, callbacks=[early_stopping, reduce_lr])
    
    # Predict on the test set
    predictions = model.predict(test_generator)
    all_run_predictions.append(predictions)

# Aggregate predictions: take the mode of the predictions across the runs for each sample
aggregated_predictions = mode(np.array(all_run_predictions), axis=0).mode.squeeze()
predicted_classes = np.argmax(aggregated_predictions, axis=1)
true_classes = test_generator.classes

# Map class indices to labels
index_to_label_map = {v: k for k, v in train_generator.class_indices.items()}

# Calculate accuracy per class
accuracy_per_class = {}
for class_index in np.unique(true_classes):
    class_accuracy = np.mean(predicted_classes[true_classes == class_index] == class_index)
    class_label = index_to_label_map[class_index]
    accuracy_per_class[class_label] = class_accuracy

# Print accuracy per class
for class_label, accuracy in accuracy_per_class.items():
    print(f"Accuracy for class {class_label}: {accuracy:.2f}")

# Calculate and print overall accuracy
overall_accuracy = np.mean(predicted_classes == true_classes)
print(f"Overall Accuracy: {overall_accuracy:.2f}")

# Plot the training and validation accuracy and loss from the best run
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

