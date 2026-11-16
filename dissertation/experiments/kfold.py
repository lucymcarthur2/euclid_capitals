import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from datetime import datetime

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

def load_data(file_paths, labels):
    data = []
    for path in file_paths:
        image = tf.keras.preprocessing.image.load_img(path, target_size=(64, 64))
        image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        data.append(image)
    return np.array(data), np.array(labels)

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        
        Conv2D(64, (3, 3), activation='relu'),
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
        Dropout(0.3),
        
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(26, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate(train_data, train_labels, val_data, val_labels, class_weights):
    model = create_model()
    
    # Define early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min')
    
    # Define a Learning Rate Scheduler
    def lr_time_based_decay(epoch, lr):
        return lr * 1 / (1 + 0.01 * epoch)
    
    lr_scheduler = LearningRateScheduler(lr_time_based_decay, verbose=1)
    
    history = model.fit(
        train_data,
        train_labels,
        epochs=30,
        validation_data=(val_data, val_labels),
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, lr_scheduler]
    )
    
    plot_history(history)
    
    predictions = model.predict(val_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(val_labels, axis=1)
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:\n", cm)
    
    # Classification Report
    classification_report_result = classification_report(true_classes, predicted_classes, output_dict=True)
    print("Classification Report:\n", classification_report_result)
    
    overall_accuracy = classification_report_result['accuracy'] * 100
    f1_score = classification_report_result['weighted avg']['f1-score'] * 100
    balanced_accuracy = balanced_accuracy_score(true_classes, predicted_classes) * 100
    
    return model, overall_accuracy, f1_score, balanced_accuracy

# Define a directory to save the models
MODEL_SAVE_DIR = r"data\models"
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

accuracies = []
f1_scores = []
balanced_accuracies = []
training_set_sizes = [1000]  # Example sizes

class_labels = os.listdir(TRAIN_DIR)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

best_model = None
best_accuracy = 0

for size in training_set_sizes:
    print(f"Training with {size} samples per class")

    sampled_file_paths, sampled_labels = get_sampled_data(TRAIN_DIR, class_labels, size)
    
    # Convert class labels to numerical labels for StratifiedKFold
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    numerical_labels = [label_to_index[label] for label in sampled_labels]
    
    fold_accuracies = []
    fold_f1_scores = []
    fold_balanced_accuracies = []
    
    for fold, (train_index, val_index) in enumerate(kfold.split(sampled_file_paths, numerical_labels)):
        train_file_paths = [sampled_file_paths[i] for i in train_index]
        val_file_paths = [sampled_file_paths[i] for i in val_index]
        
        train_labels = [sampled_labels[i] for i in train_index]
        val_labels = [sampled_labels[i] for i in val_index]
        
        train_data, train_labels = load_data(train_file_paths, train_labels)
        val_data, val_labels = load_data(val_file_paths, val_labels)
        
        train_labels = tf.keras.utils.to_categorical([label_to_index[label] for label in train_labels], num_classes=len(class_labels))
        val_labels = tf.keras.utils.to_categorical([label_to_index[label] for label in val_labels], num_classes=len(class_labels))
        
        # Calculate class weights to handle imbalanced data
        class_counts = Counter(train_labels.argmax(axis=1))
        total_samples = sum(class_counts.values())
        class_weights = {idx: total_samples / class_counts[idx] for idx in class_counts.keys()}
        
        model, accuracy, f1_score, balanced_accuracy = train_and_evaluate(train_data, train_labels, val_data, val_labels, class_weights)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
        
        fold_accuracies.append(accuracy)
        fold_f1_scores.append(f1_score)
        fold_balanced_accuracies.append(balanced_accuracy)
    
    accuracies.append(np.mean(fold_accuracies))
    f1_scores.append(np.mean(fold_f1_scores))
    balanced_accuracies.append(np.mean(fold_balanced_accuracies))

# Save the best performing model
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
best_model_save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{current_time}_{best_accuracy:.2f}.h5")
best_model.save(best_model_save_path)
print(f"Best model saved at {best_model_save_path} with accuracy: {best_accuracy:.2f}")

print("All accuracies:", accuracies)
print("All F1 Scores:", f1_scores)
print("All Balanced Accuracies:", balanced_accuracies)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(training_set_sizes, accuracies, label='Accuracy', marker='o')
plt.title('Accuracy across Different Training Set Sizes')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(training_set_sizes, f1_scores, label='F1 Score', marker='o')
plt.title('F1 Score across Different Training Set Sizes')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(training_set_sizes, balanced_accuracies, label='Balanced Accuracy', marker='o')
plt.title('Balanced Accuracy across Different Training Set Sizes')
plt.xlabel('Training Set Size')
plt.ylabel('Balanced Accuracy')
plt.legend()

plt.show()


