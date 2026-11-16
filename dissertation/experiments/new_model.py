# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:28:26 2024

@author: Lucy McArthur
"""

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
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.random.set_seed(42)

# Directory paths
TRAIN_DIR = r"data\manual_capitals"
#TEST_DIR = r"data\compilatio\extracted\processed"
TEST_DIR = r"data\extracted_letters"

accuracies = []
f1_scores = []
balanced_accuracies = []

for i in range(5):
    # Image data generators for preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load and preprocess images
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        batch_size=200,  # Increased batch size
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
    
    # Calculate class weights to handle imbalanced data
    total_samples = sum(train_class_counts.values())
    class_weights = {
        idx: total_samples / count
        for idx, count in enumerate(train_class_counts.values())
    }
    
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
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(26, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
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
    # Define a Learning Rate Scheduler
    def lr_time_based_decay(epoch, lr):
        return lr * 1 / (1 + 0.01 * epoch)
    
    # Include this in your model training callbacks
    from tensorflow.keras.callbacks import LearningRateScheduler
    lr_scheduler = LearningRateScheduler(lr_time_based_decay, verbose=1)
    
    # Update your model.fit() call to include this new callback
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=test_generator,
        callbacks=[early_stopping, reduce_lr, lr_scheduler]
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
    classification_report_result = classification_report(true_classes, predicted_classes, labels=unique_classes, target_names=actual_class_labels, output_dict=True)
    print("Classification Report:\n", classification_report_result)
    
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
    
    # Calculate F1 score and balanced accuracy
    f1_score = classification_report_result['weighted avg']['f1-score'] * 100
    balanced_accuracy = balanced_accuracy_score(true_classes, predicted_classes) * 100
    
    f1_scores.append(f1_score)
    balanced_accuracies.append(balanced_accuracy)

print("All accuracies:", accuracies)
print("All F1 Scores:", f1_scores)
print("All Balanced Accuracies:", balanced_accuracies)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(accuracies, label='Accuracy', marker='o')
plt.title('Accuracy across Runs')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(f1_scores, label='F1 Score', marker='o')
plt.title('F1 Score across Runs')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(balanced_accuracies, label='Balanced Accuracy', marker='o')
plt.title('Balanced Accuracy across Runs')
plt.legend()

plt.show()
