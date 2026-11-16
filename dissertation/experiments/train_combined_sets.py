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

# List of directories to use for training and testing
TRAIN_DIRS = [
    r"data\organised_capitals",
    r"data\compilatio\extracted\sorted_capitals",
    r"data\ptolemy\organised"
]

# Separate directory for generated capitals (100% for training)
GENERATED_DIR = r"data\manual_capitals"

TEMP_TRAIN_DIR = r"data\temp_train"
TEMP_TEST_DIR = r"data\temp_test"

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

def sample_data(directory, class_labels, max_samples_per_class=1000):
    sampled_file_paths = []
    sampled_labels = []

    for label in class_labels:
        class_dir = os.path.join(directory, label)
        if not os.path.exists(class_dir):
            continue
        all_files = os.listdir(class_dir)
        np.random.shuffle(all_files)
        sampled_files = all_files[:max_samples_per_class]

        for file in sampled_files:
            sampled_file_paths.append(os.path.join(class_dir, file))
            sampled_labels.append(label)

    return sampled_file_paths, sampled_labels

def split_data(directories, class_labels, sample_size_ratio):
    train_file_paths = []
    train_labels = []
    test_file_paths = []
    test_labels = []

    for directory in directories:
        for label in class_labels:
            class_dir = os.path.join(directory, label)
            if not os.path.exists(class_dir):
                continue
            all_files = os.listdir(class_dir)
            np.random.shuffle(all_files)
            split_index = int(len(all_files) * sample_size_ratio)
            
            train_files = all_files[:split_index]
            test_files = all_files[split_index:]
            
            for file in train_files:
                train_file_paths.append(os.path.join(class_dir, file))
                train_labels.append(label)
            
            for file in test_files:
                test_file_paths.append(os.path.join(class_dir, file))
                test_labels.append(label)

    return train_file_paths, train_labels, test_file_paths, test_labels

def prepare_directory_structure(temp_dir, class_labels):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    for label in class_labels:
        os.makedirs(os.path.join(temp_dir, label), exist_ok=True)

def copy_files_to_directory(file_paths, labels, dest_dir):
    for path, label in zip(file_paths, labels):
        temp_path = os.path.join(dest_dir, label, os.path.basename(path))
        shutil.copy(path, temp_path)

# Define directories
MODEL_SAVE_DIR = r"data\models"
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# Training configurations
accuracies = []
f1_scores = []
balanced_accuracies = []
sample_size_ratio = 0.7  # 30% for training

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
)
test_datagen = ImageDataGenerator(rescale=1./255)


class_labels = os.listdir(TRAIN_DIRS[0])


manual_capitals_file_paths, manual_capitals_labels = sample_data(TRAIN_DIRS[0], class_labels, max_samples_per_class=1000)
train_file_paths, train_labels, test_file_paths, test_labels = split_data(TRAIN_DIRS[1:], class_labels, sample_size_ratio)
generated_file_paths, generated_labels = sample_data(GENERATED_DIR, class_labels)

all_train_file_paths = manual_capitals_file_paths + train_file_paths + generated_file_paths
all_train_labels = manual_capitals_labels + train_labels + generated_labels


prepare_directory_structure(TEMP_TRAIN_DIR, class_labels)
prepare_directory_structure(TEMP_TEST_DIR, class_labels)


copy_files_to_directory(all_train_file_paths, all_train_labels, TEMP_TRAIN_DIR)
copy_files_to_directory(test_file_paths, test_labels, TEMP_TEST_DIR)


train_generator = train_datagen.flow_from_directory(
    TEMP_TRAIN_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    TEMP_TEST_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Count classes
train_class_counts = count_classes(TEMP_TRAIN_DIR, class_labels)
test_class_counts = count_classes(TEMP_TEST_DIR, class_labels)
print("Training set class distribution:", train_class_counts)
print("Test set class distribution:", test_class_counts)

# Class weights
total_samples = sum(train_class_counts.values())
class_weights = {
    idx: total_samples / count
    for idx, count in enumerate(train_class_counts.values())
}

# Adjust class weights for specific classes
if 'E' in train_generator.class_indices:
    class_weights[train_generator.class_indices['E']] *= 2
if 'M' in train_generator.class_indices:
    class_weights[train_generator.class_indices['M']] *= 2
if 'T' in train_generator.class_indices:
    class_weights[train_generator.class_indices['T']] *= 2
if 'R' in train_generator.class_indices:
    class_weights[train_generator.class_indices['R']] *= 1.5

# Build the CNN model
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

# Compile the model
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


model_save_path = os.path.join(MODEL_SAVE_DIR, f"model_{int(sample_size_ratio * 100)}_percent_samples.h5")
model.save(model_save_path)
print(f"Model saved at {model_save_path}")


plot_history(history)


predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes


unique_classes = np.unique(np.concatenate((true_classes, predicted_classes)))
actual_class_labels = [class_labels[i] for i in unique_classes]

cm = confusion_matrix(true_classes, predicted_classes, labels=unique_classes)
print("Confusion Matrix:\n", cm)

classification_report_result = classification_report(true_classes, predicted_classes, labels=unique_classes, target_names=actual_class_labels, output_dict=True)
print("Classification Report:\n", classification_report_result)


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

print("All accuracies:", accuracies)
print("All F1 Scores:", f1_scores)
print("All Balanced Accuracies:", balanced_accuracies)

# Plot accuracies
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot([int(sample_size_ratio * 100)], accuracies, label='Accuracy', marker='o')
plt.title('Accuracy')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot([int(sample_size_ratio * 100)], f1_scores, label='F1 Score', marker='o')
plt.title('F1 Score')
plt.xlabel('Training Set Size (%)')
plt.ylabel('F1 Score')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot([int(sample_size_ratio * 100)], balanced_accuracies, label='Balanced Accuracy', marker='o')
plt.title('Balanced Accuracy')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Balanced Accuracy')
plt.legend()

plt.show()

# Clean up temporary directories
shutil.rmtree(TEMP_TRAIN_DIR)
shutil.rmtree(TEMP_TEST_DIR)