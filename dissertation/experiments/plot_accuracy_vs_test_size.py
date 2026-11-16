# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:01:22 2024

@author: Lucy McArthur
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

# Function to load models from a list of paths
def load_models(model_paths):
    models = []
    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path)
        models.append(model)
    return models

from PIL import Image, ImageDraw, ImageFont

def evaluate_models(models, test_generator, class_labels, output_dir='predicted_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    accuracies = []
    f1_scores = []
    balanced_accuracies = []
    class_wise_accuracies = []
    class_wise_balanced_accuracies = []
    class_total_counts_list = []

    for model in models:
        num_images = len(test_generator.filenames)
        correct_predictions = 0
        all_true_labels = []
        all_predicted_labels = []
        class_correct_counts = np.zeros(len(class_labels))
        class_total_counts = np.zeros(len(class_labels))

        for i in range(num_images):
            x, y_true = test_generator[i]
            y_true_index = np.argmax(y_true)
            y_true_label = class_labels[y_true_index]

            predictions = model.predict(x)
            top_index = np.argmax(predictions[0])
            top_label = class_labels[top_index]

            all_true_labels.append(y_true_index)
            all_predicted_labels.append(top_index)

            class_total_counts[y_true_index] += 1
            if top_label == y_true_label:
                correct_predictions += 1
                class_correct_counts[y_true_index] += 1

            # Draw the predicted letter on the image
            img_array = (x[0] * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", 40)  # Use a default font and size 40
            draw.text((10, 10), top_label, fill="red", font=font)
            
            # Save the image with the predicted label
            image_path = os.path.join(output_dir, f"pred_{top_label}_{i}.png")
            img.save(image_path)

        overall_accuracy = correct_predictions / num_images
        accuracies.append(overall_accuracy)
        f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
        balanced_acc = balanced_accuracy_score(all_true_labels, all_predicted_labels)
        f1_scores.append(f1)
        balanced_accuracies.append(balanced_acc)

        class_accuracy = np.divide(class_correct_counts, class_total_counts, out=np.zeros_like(class_correct_counts, dtype=float), where=class_total_counts != 0)
        class_wise_accuracies.append(class_accuracy)
        class_total_counts_list.append(class_total_counts)
        
        class_balanced_accuracy = np.zeros(len(class_labels))
        for i in range(len(class_labels)):
            if class_total_counts[i] > 0:
                true_positives = class_correct_counts[i]
                false_positives = np.sum(class_total_counts) - class_total_counts[i] - (num_images - correct_predictions)
                false_negatives = class_total_counts[i] - true_positives
                true_negatives = num_images - class_total_counts[i] - false_positives

                sensitivity = true_positives / class_total_counts[i]
                specificity = true_negatives / (true_negatives + false_positives)
                class_balanced_accuracy[i] = (sensitivity + specificity) / 2
        
        class_wise_balanced_accuracies.append(class_balanced_accuracy)

    return accuracies, f1_scores, balanced_accuracies, class_wise_accuracies, class_wise_balanced_accuracies, class_total_counts_list

# Function to plot the results
def plot_metrics_vs_size(batch_sizes, accuracies, f1_scores, balanced_accuracies):
    plt.figure(figsize=(10, 6))

    plt.plot(batch_sizes, accuracies, marker='o', linestyle='-', label='Accuracy')
    plt.plot(batch_sizes, f1_scores, marker='o', linestyle='--', label='F1 Score')
    plt.plot(batch_sizes, balanced_accuracies, marker='o', linestyle='-.', label='Balanced Accuracy')

    plt.title('Metrics vs Test Batch Size', fontsize=16)
    plt.xlabel('Test Batch Size', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Function to plot class-wise accuracy for each model
def plot_class_wise_accuracy(models, class_labels, class_wise_accuracies, batch_sizes, class_total_counts_list):
    plt.figure(figsize=(10, 6))
    
    for i, class_accuracy in enumerate(class_wise_accuracies):
        class_total_counts = class_total_counts_list[i]
        valid_indices = np.where(class_total_counts != 0)[0]
        valid_class_labels = [class_labels[j] for j in valid_indices]
        valid_class_accuracy = class_accuracy[valid_indices]
        plt.plot(valid_class_labels, valid_class_accuracy, marker='o', label=f'Model with batch size {batch_sizes[i]}')

    plt.title('Class accuracy with varying test batch size', fontsize=24)
    plt.xlabel('Class', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.xticks(rotation=45, fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot class-wise balanced accuracy for each model
def plot_class_wise_balanced_accuracy(models, class_labels, class_wise_balanced_accuracies, batch_sizes, class_total_counts_list):
    plt.figure(figsize=(10, 6))
    
    for i, class_balanced_accuracy in enumerate(class_wise_balanced_accuracies):
        class_total_counts = class_total_counts_list[i]
        valid_indices = np.where(class_total_counts != 0)[0]
        valid_class_labels = [class_labels[j] for j in valid_indices]
        valid_class_balanced_accuracy = class_balanced_accuracy[valid_indices]
        plt.plot(valid_class_labels, valid_class_balanced_accuracy, marker='o', label=f'Model with batch size {batch_sizes[i]}')

    plt.title('Class-wise balanced accuracy with varying test batch size', fontsize=24)
    plt.xlabel('Class', fontsize=22)
    plt.ylabel('Balanced Accuracy', fontsize=22)
    plt.xticks(rotation=45, fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
model_paths = [
    r"data\models\model_1000_16_test_set.h5",
    r"data\models\model_1000_32_test_set.h5",
    r"data\models\model_1000_64_test_set.h5",
    r"data\models\model_1000_128_test_set.h5"
]
batch_sizes = [16, 32, 64, 128]  # Corresponding batch sizes
test_dir = r"data\extracted_letters"

models = load_models(model_paths)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(test_generator.class_indices.keys())
print("Class Labels:", class_labels)

accuracies, f1_scores, balanced_accuracies, class_wise_accuracies, class_wise_balanced_accuracies, class_total_counts_list = evaluate_models(models, test_generator, class_labels)
plot_metrics_vs_size(batch_sizes, accuracies, f1_scores, balanced_accuracies)
plot_class_wise_accuracy(models, class_labels, class_wise_accuracies, batch_sizes, class_total_counts_list)
plot_class_wise_balanced_accuracy(models, class_labels, class_wise_balanced_accuracies, batch_sizes, class_total_counts_list)
