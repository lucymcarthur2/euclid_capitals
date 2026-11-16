# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:30:05 2024

@author: Lucy McArthur
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:12:29 2024

@author: Lucy McArthur
"""

import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

np.random.seed(42)
tf.random.set_seed(42)

# Directory paths
TRAIN_DIR = r"data\manual_capitals"
TEST_DIR = r"data\extracted_letters"

# Function to count class instances in a directory                                        
def count_classes(directory, class_labels):
    counts = {label: 0 for label in class_labels}
    for subdir in os.listdir(directory):
        if subdir in class_labels:
            path = os.path.join(directory, subdir)
            counts[subdir] = len(os.listdir(path))
    return counts

def plot_history(history):
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

def get_sampled_data(directory, class_labels, sample_size):
    sampled_file_paths = []
    sampled_labels = []

    for label in class_labels:
        class_dir = os.path.join(directory, label)
        all_files = os.listdir(class_dir)
        sampled_files = np.random.choice(all_files, size=sample_size, replace=False)

        for file in sampled_files:
            sampled_file_paths.append(os.path.join(class_dir, file))
            sampled_labels.append(label)

    return sampled_file_paths, sampled_labels

accuracies = []
f1_scores = []
balanced_accuracies = []
num_runs = 5
batch_size = 32
training_set_size = 1000

# Define the residual block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same', strides=stride)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    if stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Create a temporary directory to store the sampled data
temp_train_dir = r"data\temp_train"

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")

    # Clear and recreate the temporary directory
    if os.path.exists(temp_train_dir):
        shutil.rmtree(temp_train_dir)
    os.makedirs(temp_train_dir)

    # Sample training data
    class_labels = os.listdir(TRAIN_DIR)
    sampled_file_paths, sampled_labels = get_sampled_data(TRAIN_DIR, class_labels, training_set_size)

    for label in class_labels:
        os.makedirs(os.path.join(temp_train_dir, label), exist_ok=True)

    for path, label in zip(sampled_file_paths, sampled_labels):
        temp_path = os.path.join(temp_train_dir, label, os.path.basename(path))
        shutil.copy(path, temp_path)

    # Image data generators for preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load and preprocess images
    train_generator = train_datagen.flow_from_directory(
        temp_train_dir,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Class labels
    class_labels = list(train_generator.class_indices.keys())
    
    # Print class distribution in training and test sets
    train_class_counts = count_classes(temp_train_dir, class_labels)
    test_class_counts = count_classes(TEST_DIR, class_labels)
    print("Training set class distribution:", train_class_counts)
    print("Test set class distribution:", test_class_counts)
    
    # Calculate class weights to handle imbalanced data
    total_samples = sum(train_class_counts.values())
    class_weights = {
        idx: total_samples / count
        for idx, count in enumerate(train_class_counts.values())
    }
    
    # Building the CNN Model with Residual Blocks
    input_img = Input(shape=(64, 64, 3))

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(26, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=output)
    
    # Compile the model with a reduced learning rate
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    # Define early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min')
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=test_generator,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Plot training history
    plot_history(history)
    
    # Predictions and Confusion Matrix
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:\n", cm)
    
    # Classification Report
    class_labels = list(test_generator.class_indices.keys())
    classification_report_result = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
    print("Classification Report:\n", classification_report_result)
    
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
print("Average accuracy across runs:", np.mean(accuracies))

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(range(1, num_runs + 1), accuracies, label='Accuracy', marker='o')
plt.title('Accuracy across Different Runs')
plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, num_runs + 1), f1_scores, label='F1 Score', marker='o')
plt.title('F1 Score across Different Runs')
plt.xlabel('Run')
plt.ylabel('F1 Score')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(range(1, num_runs + 1), balanced_accuracies, label='Balanced Accuracy', marker='o')
plt.title('Balanced Accuracy across Different Runs')
plt.xlabel('Run')
plt.ylabel('Balanced Accuracy')
plt.legend()

plt.show()

# Clean up the temporary directory
shutil.rmtree(temp_train_dir)
