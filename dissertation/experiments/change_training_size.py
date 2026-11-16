# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:31:53 2024

@author: Lucy McArthur
"""

import numpy as np
import os
import shutil
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

TRAIN_DIR = r"data\simple_caps"
TEST_DIRS = [
    r"data\extracted_letters",
    #r"data\ptolemy\processed",
    #r"data\compilatio\extracted\processed",
    #r"data\artes_orandi\processed"
]

def merge_directories(directories, destination):
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)

    for directory in directories:
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                dest_subdir = os.path.join(destination, subdir)
                if not os.path.exists(dest_subdir):
                    os.makedirs(dest_subdir)
                for item in os.listdir(subdir_path):
                    s = os.path.join(subdir_path, item)
                    d = os.path.join(dest_subdir, item)
                    shutil.copy2(s, d)

def count_classes(directory, class_labels):
    counts = {label: 0 for label in class_labels}
    for subdir in os.listdir(directory):
        if subdir in class_labels:
            path = os.path.join(directory, subdir)
            counts[subdir] = len(os.listdir(path))
    return counts

def plot_history(history, title_suffix):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy over Epochs {title_suffix}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss over Epochs {title_suffix}')
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

MODEL_SAVE_DIR = r"data\models"
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

TEMP_TEST_DIR = r"data\temp_test"
merge_directories(TEST_DIRS, TEMP_TEST_DIR)

accuracies = []
f1_scores = []
balanced_accuracies = []
training_set_sizes = [1000]
history_dict = {}
results_dict = {}

for size in training_set_sizes:
    try:
        print(f"Training with {size} samples per class")

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        class_labels = os.listdir(TRAIN_DIR)
        sampled_file_paths, sampled_labels = get_sampled_data(TRAIN_DIR, class_labels, size)
        
        temp_train_dir = r"data\temp_train"
        
        if os.path.exists(temp_train_dir):
            shutil.rmtree(temp_train_dir)
        os.makedirs(temp_train_dir)
        
        for label in class_labels:
            os.makedirs(os.path.join(temp_train_dir, label), exist_ok=True)
        
        for path, label in zip(sampled_file_paths, sampled_labels):
            temp_path = os.path.join(temp_train_dir, label, os.path.basename(path))
            shutil.copy(path, temp_path)
        
        train_generator = train_datagen.flow_from_directory(
            temp_train_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical'
        )
        
        test_generator = test_datagen.flow_from_directory(
            TEMP_TEST_DIR,
            target_size=(128, 128),
            batch_size=64,
            class_mode='categorical',
            shuffle=False
        )
        
        class_labels = list(train_generator.class_indices.keys())
    
        train_class_counts = count_classes(temp_train_dir, class_labels)
        test_class_counts = count_classes(TEMP_TEST_DIR, class_labels)
        print("Training set class distribution:", train_class_counts)
        print("Test set class distribution:", test_class_counts)
        
        total_samples = sum(train_class_counts.values())
        class_weights = {
            idx: total_samples / count
            for idx, count in enumerate(train_class_counts.values())
        }
        
        # Building the CNN Model
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.4),
            
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.4),
            
            Flatten(),
            Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.5),
            Dense(26, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min')
    
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_generator,
            epochs=30,
            validation_data=test_generator,
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr]
        )
        
        history_dict[size] = history.history
        
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"model_{size}_0.25erosion_0.25black_0.25_image_0.25_holes.h5")
        model.save(model_save_path)
        print(f"Model saved at {model_save_path}")
        
        plot_history(history.history, f"with {size} samples per class")

        predictions = model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        actual_class_labels = [class_labels[i] for i in np.unique(true_classes)]
        
        unique_classes = np.unique(np.concatenate((true_classes, predicted_classes)))
        
        actual_class_labels = [class_labels[i] for i in unique_classes]
        
        cm = confusion_matrix(true_classes, predicted_classes, labels=unique_classes)
        print("Confusion Matrix:\n", cm)
        
        classification_report_result = classification_report(true_classes, predicted_classes, labels=unique_classes, target_names=actual_class_labels, output_dict=True)
        print("Classification Report:\n", classification_report_result)
        
        class_labels = list(test_generator.class_indices.keys())
        
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
        
        f1_score = classification_report_result['weighted avg']['f1-score'] * 100
        balanced_accuracy = balanced_accuracy_score(true_classes, predicted_classes) * 100
        
        f1_scores.append(f1_score)
        balanced_accuracies.append(balanced_accuracy)
        
        results_dict[size] = {
            'accuracy': overall_accuracy,
            'f1_score': f1_score,
            'balanced_accuracy': balanced_accuracy,
            'confusion_matrix': cm,
            'classification_report': classification_report_result
        }
        
        # Clean up the temporary training directory
        shutil.rmtree(temp_train_dir)
        
    except Exception as e:
        print(f"An error occurred for training set size {size}: {e}")
        continue

