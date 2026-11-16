import os
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_and_classify(image_path):
    """
    Perform OCR on the given image using pytesseract and return the detected character.
    
    Args:
        image_path (str): Path to the image to be processed.
    
    Returns:
        str: The detected character.
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config='--psm 10').strip()
    if len(text) > 0:
        return text[0].upper()
    return ""

def calculate_accuracy(directory):
    """
    Calculate the accuracy of OCR classification for each class in the given directory.
    
    Args:
        directory (str): Path to the directory containing subfolders of images.
    
    Returns:
        dict: A dictionary mapping each class to its accuracy.
    """
    class_accuracies = {}
    for class_folder in os.listdir(directory):
        class_path = os.path.join(directory, class_folder)
        if not os.path.isdir(class_path):
            continue

        total_images = 0
        correct_predictions = 0
        
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                continue
            
            detected_class = ocr_and_classify(image_path)
            total_images += 1
            if detected_class == class_folder:
                correct_predictions += 1
        
        if total_images > 0:
            class_accuracies[class_folder] = correct_predictions / total_images
        else:
            class_accuracies[class_folder] = 0.0

    return class_accuracies

def plot_accuracy(accuracies):
    """
    Plot the accuracy of OCR classification for each class.
    
    Args:
        accuracies (dict): A dictionary mapping each class to its accuracy.
    """
    classes = sorted(accuracies.keys())
    accuracy_values = [accuracies[c] for c in classes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(classes, accuracy_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Class', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.title('Class accuracy using OCR', fontsize=22)
    plt.xticks(rotation=45, fontsize=22)
    plt.yticks(fontsize=22)
    
    plt.show()

# Directory containing subfolders of images (each subfolder represents a class)
directory = r"data\extracted_letters"

# Calculate accuracy and plot the results
accuracies = calculate_accuracy(directory)
plot_accuracy(accuracies)
