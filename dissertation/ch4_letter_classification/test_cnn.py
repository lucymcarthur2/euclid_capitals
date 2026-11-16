import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

MODEL_PATH = r"data\models\best_1000_images.h5"
TEST_DIR = r"data\ptolemy\processed"

model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(128, 128),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

class_labels = list(test_generator.class_indices.keys())
print("Class Labels:", class_labels)

correct_predictions_per_class = {label: 0 for label in class_labels}
total_images_per_class = {label: 0 for label in class_labels}

def display_image_with_predictions(image, true_label, predicted_label):
    plt.imshow(image)
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.axis('off')
    plt.show()

def calculate_accuracy(generator, model, class_labels, display=False):
    num_images = len(generator.filenames)
    correct_predictions = 0

    for i in range(num_images):
        x, y_true = generator[i]
        y_true_index = np.argmax(y_true)
        y_true_label = class_labels[y_true_index]
        
        predictions = model.predict(x)
        top_index = np.argmax(predictions[0])
        
        if top_index >= len(class_labels):
            print(f"Warning: predicted index {top_index} is out of range for class labels.")
            continue
        
        top_label = class_labels[top_index]
        
        if top_label == y_true_label:
            correct_predictions += 1
            correct_predictions_per_class[y_true_label] += 1
        
        total_images_per_class[y_true_label] += 1

        if display:
            display_image_with_predictions(x[0], y_true_label, top_label)
    
    overall_accuracy = correct_predictions / num_images
    print(f"Overall Accuracy: {overall_accuracy:.2f}")
    
    for label in class_labels:
        if total_images_per_class[label] > 0:
            class_accuracy = correct_predictions_per_class[label] / total_images_per_class[label]
            print(f"Accuracy for class {label}: {class_accuracy:.2f}")
        else:
            print(f"No images for class {label}")

calculate_accuracy(test_generator, model, class_labels, display=True)
