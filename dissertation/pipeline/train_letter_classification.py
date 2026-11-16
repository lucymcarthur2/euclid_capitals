import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import BatchNormalization

np.random.seed(42)
tf.random.set_seed(42)




# Directory paths
TRAIN_DIR = r"data\manual_capitals"
TEST_DIR = r"data\organised_capitals"

def binarize_image(img):
    """Convert an image to binary format."""
    image = img_to_array(img)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Apply threshold
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)  # Convert back to RGB to keep 3 channels
    return binary / 255.0  # Normalize to 0-1 range

# Image data generators for preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    preprocessing_function=binarize_image
)

test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=binarize_image)

# Load and preprocess images
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(128, 128),
    batch_size=8,
    class_mode='categorical',
    shuffle=False)

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
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

# Define early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min')

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model and save history, include callbacks
history = model.fit(
    train_generator,
    epochs=15,
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

# Print accuracy per class
for class_label in class_labels:
    if class_total[class_label] > 0:
        accuracy = (class_correct[class_label] / class_total[class_label]) * 100
        print(f"Class {class_label}: Correctly Predicted {class_correct[class_label]} out of {class_total[class_label]} ({accuracy:.2f}%)")
    else:
        print(f"Class {class_label}: No test images found")

# Print overall accuracy
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

# Reset the test_generator before going through the images
test_generator.reset()

# Get the indices of the top 3 predictions for each image
top_3_pred_indices = np.argsort(-predictions, axis=1)[:, :3]

# Get the probabilities of the top 3 predictions for each image
top_3_pred_probabilities = np.take_along_axis(predictions, top_3_pred_indices, axis=1)



# Display a limited number of images and their predictions
num_images_to_display = 0  # Set the number of images you want to display

for i in range(num_images_to_display):
    # Load the image
    img_path = os.path.join(TEST_DIR, test_generator.filenames[i])
    img = load_img(img_path, target_size=(64, 64))
    img_array = img_to_array(img)

    # Get the top 3 predictions and their probabilities
    top_predictions = top_3_pred_indices[i]
    top_probabilities = top_3_pred_probabilities[i]

    # Plot the image and the top 3 predictions with probabilities
    plt.figure(figsize=(5, 5))
    plt.imshow(img_array / 255.)
    plt.title(f"Top Predictions:\n1. {class_labels[top_predictions[0]]} ({top_probabilities[0]:.2f})\n"
              f"2. {class_labels[top_predictions[1]]} ({top_probabilities[1]:.2f})\n"
              f"3. {class_labels[top_predictions[2]]} ({top_probabilities[2]:.2f})")
    plt.axis('off')
    plt.show()
    
    
    
