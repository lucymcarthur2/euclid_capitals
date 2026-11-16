# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:42:09 2024

@author: Lucy McArthur
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array

np.random.seed(42)
tf.random.set_seed(42)

# Directory paths
TRAIN_DIR = r"data\manual_capitals"
#TEST_DIR = r"data\compilatio\extracted\processed"
TEST_DIR = r"data\extracted_letters"


accuracies = []
for i in range(0, 1):

    # Image data generators for preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load and preprocess images
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        batch_size=32,  # Increased batch size
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
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
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')
    ])
    
    # Define early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min')
    
    # Compile the model with a reduced learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        metrics=['accuracy']
    )
    
    # Train the model and save history, include callbacks
    history = model.fit(
        train_generator,
        epochs=30,  # Increased epochs
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
    
    # Predictions and Confusion Matrix
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Update class_labels to match the actual classes in the test set
    actual_class_labels = [class_labels[i] for i in np.unique(true_classes)]
    
    # Get unique classes from predictions and ground truth
    unique_classes = np.unique(np.concatenate((true_classes, predicted_classes)))
    
    # Map indices to class names
    actual_class_labels = [class_labels[i] for i in unique_classes]
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes, labels=unique_classes)
    print("Confusion Matrix:\n", cm)
    
    # Classification Report
    print("Classification Report:\n", classification_report(true_classes, predicted_classes, labels=unique_classes, target_names=actual_class_labels))
    
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
    accuracies.append(overall_accuracy)

print("all accuracies", accuracies)

