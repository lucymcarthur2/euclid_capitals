# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:42:35 2023

@author: Lucy McArthur
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator,DirectoryIterator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())

np.random.seed(42)
tf.random.set_seed(42)

# Paths to your dataset
LARGE_DATASET_DIR = r"data\external\archive (5)\Images\Images"
BESPOKE_DATASET_DIR = r"data\manual_capitals"
TEST_DIR = r"data\extracted_letters"


print("Directories set...")

class CustomDataGenerator(DirectoryIterator):
    def __init__(self, directory, image_data_generator, use_data_percentage=100, *args, **kwargs):
        self.use_data_percentage = use_data_percentage
        super().__init__(directory, image_data_generator, *args, **kwargs)
        self.on_epoch_end()

    def on_epoch_end(self):
        all_indices = np.arange(self.n)
        if self.use_data_percentage < 100:
            self.sampled_indices = np.random.choice(all_indices, int(len(all_indices) * (self.use_data_percentage / 100)), replace=False)
        else:
            self.sampled_indices = all_indices
        print(f"Using {len(self.sampled_indices)} samples out of {self.n}")

    def _get_batches_of_transformed_samples(self, index_array):
        adjusted_index_array = [self.sampled_indices[i] for i in index_array if i < len(self.sampled_indices)]
        return super()._get_batches_of_transformed_samples(adjusted_index_array)

# Training data generator for the large dataset (no augmentation)
large_datagen = ImageDataGenerator(rescale=1./255)
print("Large dataset generator created...")

# Use 1% of the large dataset for testing
large_train_generator = CustomDataGenerator(
    LARGE_DATASET_DIR,
    large_datagen,
    use_data_percentage=1,  # Use 1% for testing purposes
    target_size=(64, 64),
    batch_size=100,
    class_mode='categorical',
    shuffle=True
)
print("Large training generator set...")

# Augmentation for the smaller, bespoke dataset
bespoke_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)
print("Bespoke dataset generator with augmentation created...")

# Bespoke dataset generator
bespoke_train_generator = bespoke_datagen.flow_from_directory(
    BESPOKE_DATASET_DIR,
    target_size=(64, 64),
    batch_size=100,
    class_mode='categorical',
    shuffle=True
)
print("Bespoke training generator set...")

# Combined Generator
# Combined Generator
def combined_generator(generator1, generator2):
    print("Combining generators...")
    while True:
        try:
            X1, y1 = generator1.next()
            X2, y2 = generator2.next()
            # Handle different batch sizes for the last batch of the epoch
            min_batch_size = min(X1.shape[0], X2.shape[0])
            if min_batch_size == 0:
                continue
            yield np.concatenate([X1[:min_batch_size], X2[:min_batch_size]], axis=0), \
                  np.concatenate([y1[:min_batch_size], y2[:min_batch_size]], axis=0)
        except (StopIteration, ValueError) as e:
            print("Error encountered:", e)
            break

combined_train_generator = combined_generator(large_train_generator, bespoke_train_generator)
print("Combined generator ready...")

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(64, 64),
    batch_size=100,
    class_mode='categorical',
    shuffle=False
)
print("Test generator set...")

# Model Building
def build_model(num_classes):
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

input_shape = (64, 64, 3)
num_classes = len(test_generator.class_indices)

model = build_model(num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min')

history = model.fit(
    combined_train_generator,
    epochs=3,
    steps_per_epoch=(len(large_train_generator) + len(bespoke_train_generator)),
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

# Evaluate the model
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Confusion Matrix and Classification Report
cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", cm)
class_labels = list(test_generator.class_indices.keys())
# Ensure all expected classes are accounted for
expected_classes = set(class_labels)
true_classes_set = set(class_labels[true] for true in true_classes)
predicted_classes_set = set(class_labels[pred] for pred in predicted_classes)

missing_classes = expected_classes - true_classes_set - predicted_classes_set
if missing_classes:
    print(f"Warning: Missing classes in the dataset - {missing_classes}")

# Provide labels argument to classification_report to handle missing classes
print("Classification Report:\n", classification_report(true_classes, predicted_classes, target_names=class_labels, labels=range(len(class_labels))))

# Calculate and print accuracy per class
class_correct = Counter()
class_total = Counter()
for true, pred in zip(true_classes, predicted_classes):
    class_total[class_labels[true]] += 1
    if true == pred:
        class_correct[class_labels[true]] += 1

total_correct = sum(class_correct.values())
total_predictions = len(true_classes)
overall_accuracy = (total_correct / total_predictions) * 100

for class_label in class_labels:
    if class_total[class_label] > 0:
        accuracy = (class_correct[class_label] / class_total[class_label]) * 100
        print(f"Class {class_label}: Correctly Predicted {class_correct[class_label]} out of {class_total[class_label]} ({accuracy:.2f}%)")

print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

model.save('categorise_initial_capitals.h5')
