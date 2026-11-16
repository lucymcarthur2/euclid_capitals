import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score
import os

# Paths
TRAIN_DIR = r"data\manual_capitals"
TEST_DIR = r"data\extracted_letters"

# Parameters
BATCH_SIZE = 16
IMG_SIZE = (224, 224)  # ResNet50 default input size
VALIDATION_SPLIT = 0.2  # Use 20% of the test data for validation

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # Additional augmentation parameters can be added here if needed
    validation_split=VALIDATION_SPLIT  # Split off for validation
)

# Train data generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'  # Set as training data
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='validation'  # Set as validation data
)

# Test data generator
test_datagen = ImageDataGenerator(
    rescale=1./255,
    # Preprocessing_function can be added here if needed
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load ResNet50 with weights pretrained on ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')  # 26 classes for the English alphabet
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Compute class weights
labels = train_generator.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

# Train the model using both training and validation data
history = model.fit(
    train_generator,
    epochs=3,
    validation_data=validation_generator,
    class_weight=class_weights
)

# Display training and validation accuracy
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Training Accuracy (from history): {train_acc*100:.2f}%")
print(f"Validation Accuracy (from history): {val_acc*100:.2f}%")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"Evaluation on Test Set - Loss: {test_loss:.4f}, Accuracy: {test_accuracy*100:.2f}%")

# Predict on the test set
test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
predictions = model.predict(test_generator, steps=test_steps_per_epoch)

# Get most likely class index
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())  # Get class labels from the generator

# Calculate per-class accuracy and other metrics
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Calculate the accuracy for each class
accuracies = accuracy_score(true_classes, predicted_classes, normalize=False) / np.bincount(true_classes)
print("Per-class accuracy:")
for idx, accuracy in enumerate(accuracies):
    print(f"Class {class_labels[idx]}: {accuracy*100:.2f}%")
