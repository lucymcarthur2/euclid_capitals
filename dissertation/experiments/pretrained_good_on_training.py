import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory paths
DATA_DIR = r"data\test_s"

# Ensure the data directory exists
if not os.path.isdir(DATA_DIR):
    raise ValueError(f"Data directory not found: {DATA_DIR}")

# Image data generators without augmentation
data_gen = ImageDataGenerator(rescale=1./255)

# Load images from the same directory for both training and testing
data_generator = data_gen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True  # Enable shuffling for training purposes
)

# Retrieve class labels
class_labels = list(data_generator.class_indices.keys())

# Ensure consistent class mapping
print("Data class indices:", data_generator.class_indices)

# Count classes
def count_classes(directory, class_labels):
    counts = {label: 0 for label in class_labels}
    for subdir in os.listdir(directory):
        if subdir in class_labels:
            path = os.path.join(directory, subdir)
            counts[subdir] = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    return counts

# Verify class distribution
class_counts = count_classes(DATA_DIR, class_labels)
print("Data set class distribution:", class_counts)

# Calculate class weights to address class imbalance
total_samples = sum(class_counts.values())
class_weights = {
    idx: total_samples / count
    for idx, count in enumerate(class_counts.values())
}

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(class_labels), activation='softmax')
])

# Define early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min')

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model with class weights
history = model.fit(
    data_generator,
    epochs=30,
    validation_data=data_generator,
    class_weight=class_weights,
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
predictions = model.predict(data_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = data_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(true_classes, predicted_classes, target_names=class_labels))

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
