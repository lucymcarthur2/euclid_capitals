import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory paths
SOURCE_DIR = r"data\organised_capitals"
TARGET_DIR = r"data\augmented_capitals"
TARGET_COUNT = 100  # Target number of images per class

# Create target directory if it doesn't exist
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# Function to add random noise, blur, and brightness adjustment
def random_transforms(image):
    # Random Gaussian blur
    if np.random.rand() > 0.5:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Random brightness adjustment
    if np.random.rand() > 0.5:
        factor = 0.5 + np.random.uniform()
        image = cv2.convertScaleAbs(image, alpha=factor, beta=0)

    # Adding random noise
    if np.random.rand() > 0.5:
        scale = np.random.rand() * 0.05 * np.max(image)
        noise = np.random.normal(loc=0, scale=scale, size=image.shape)
        image = np.clip(image + noise, 0, 255)

    return image

# Define ImageDataGenerator with the custom preprocessing function
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=False,
    fill_mode='nearest',
    preprocessing_function=random_transforms
)

# Function to augment images and save them
def augment_images(source_folder, target_folder, target_count):
    # Make sure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    
    # List existing images
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg') or f.endswith('.png')]
    if not image_files:  # Skip processing if no images are found
        print(f"No images to process in {source_folder}. Skipping...")
        return

    generated_count = len(os.listdir(target_folder))

    while generated_count < target_count:
        for image_file in image_files:
            if generated_count >= target_count:
                break
            img_path = os.path.join(source_folder, image_file)
            img = load_img(img_path)  # Load image
            img_array = img_to_array(img)  # Convert it to numpy array
            img_array = img_array.reshape((1,) + img_array.shape)  # Reshape it to (1, height, width, channels)
            
            # Generate batches of augmented images from this image
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=target_folder, save_prefix='aug', save_format='jpeg'):
                generated_count += 1
                if generated_count >= target_count:
                    break

# Process each class directory
for class_name in os.listdir(SOURCE_DIR):
    class_source_dir = os.path.join(SOURCE_DIR, class_name)
    class_target_dir = os.path.join(TARGET_DIR, class_name)
    if os.path.isdir(class_source_dir):  # Ensure it is a directory
        augment_images(class_source_dir, class_target_dir, TARGET_COUNT)

print("Augmentation complete. Each class directory now contains 200 images, if originally populated.")
