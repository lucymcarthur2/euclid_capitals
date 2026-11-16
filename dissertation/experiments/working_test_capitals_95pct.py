
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array

TEST_DIR = r"data\extracted_letters"
# Specify the path to the saved model file in your Google Drive directory
model_path = r"data\95pct_model.h5"  # Update with the correct path
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the saved model
model = tf.keras.models.load_model(model_path)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    shuffle=False)



class_labels = list(test_generator.class_indices.keys())

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
#print("Confusion Matrix:\n", cm)

# Classification Report
#print("Classification Report:\n", classification_report(true_classes, predicted_classes, labels=unique_classes, target_names=actual_class_labels))

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
num_images_to_display = 507  # Set the number of images you want to display
num_wrong_predictions_to_display = 10  # Set the number of wrong predictions to display

wrong_predictions_displayed = 0  # Initialize the count of displayed wrong predictions

for i in range(num_images_to_display):
    # Load the image
    img_path = os.path.join(TEST_DIR, test_generator.filenames[i])
    img = load_img(img_path, target_size=(64, 64))
    img_array = img_to_array(img)

    # Get the top 3 predictions and their probabilities
    top_predictions = top_3_pred_indices[i]
    top_probabilities = top_3_pred_probabilities[i]

    # Check if the prediction is wrong
    if true_classes[i] != top_predictions[0]:
        wrong_predictions_displayed += 1
        plt.figure(figsize=(5, 5))
        plt.imshow(img_array / 255.)
        plt.title(f"True Class: {class_labels[true_classes[i]]}\n"
                  f"Top Predictions:\n1. {class_labels[top_predictions[0]]} ({top_probabilities[0]:.2f})\n"
                  f"2. {class_labels[top_predictions[1]]} ({top_probabilities[1]:.2f})\n"
                  f"3. {class_labels[top_predictions[2]]} ({top_probabilities[2]:.2f})")
        plt.axis('off')
        plt.show()

    # Break the loop after displaying a specified number of wrong predictions
    if wrong_predictions_displayed >= num_wrong_predictions_to_display:
        break
