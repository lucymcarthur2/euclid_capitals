

import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

def adaptive_preprocess_image_remove_noise(image):
    # Determine adaptive parameters
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    # Adaptive Gaussian blur kernel size based on image noise level
    if std_intensity < 20:
        blur_ksize = (3, 3)
    elif std_intensity < 50:
        blur_ksize = (5, 5)
    else:
        blur_ksize = (7, 7)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, blur_ksize, 0)

    # Adaptive threshold based on mean intensity
    _, thresh = cv2.threshold(blurred, mean_intensity, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Adaptive noise threshold based on image characteristics
    noise_threshold = max(50, std_intensity * 20)
    
    # Filter out small noise by contour area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > noise_threshold]

    # Create mask and apply to image to remove small noise
    mask = np.zeros_like(image)
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

def process_image_and_remove_small_contours(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None  # If the image couldn't be read, return None

    # Define border sizes
    top_border_size = 20
    bottom_border_size = 20
    left_border_size = 23
    right_border_size = 33

    # Crop the image to remove the border
    cropped_image = image[top_border_size:-bottom_border_size, left_border_size:-right_border_size]

    # Preprocess and remove noise
    processed_image = adaptive_preprocess_image_remove_noise(cropped_image)

    # Binarize the processed image to ensure it is fully binary
    _, final_binary_image = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY)

    # Add back the borders removed at the start
    final_image_with_border = cv2.copyMakeBorder(
        final_binary_image,
        top_border_size, bottom_border_size, left_border_size, right_border_size,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return final_image_with_border

def process_directory(input_directory, output_directory, contour_area_dict):
    for root, dirs, files in os.walk(input_directory):
        if not dirs and not files:  # Check if directory is empty
            continue

        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            output_subdir = subdir_path.replace(input_directory, output_directory)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            image_paths = glob(os.path.join(subdir_path, '*.png'))
            for image_path in image_paths:
                processed_image = process_image_and_remove_small_contours(image_path)
                if processed_image is None:
                    continue
                filename = os.path.basename(image_path)
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_processed{ext}"
                output_path = os.path.join(output_subdir, new_filename)
                cv2.imwrite(output_path, processed_image)
                print(f"Processed and saved to {output_path}")
                # Uncomment below lines to display the image
                plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                plt.show()

        # Process files in the current directory (if there are no subdirectories)
        if not dirs:
            output_subdir = root.replace(input_directory, output_directory)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            image_paths = glob(os.path.join(root, '*.png'))
            for image_path in image_paths:
                processed_image = process_image_and_remove_small_contours(image_path)
                if processed_image is None:
                    continue
                filename = os.path.basename(image_path)
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_processed{ext}"
                output_path = os.path.join(output_subdir, new_filename)
                cv2.imwrite(output_path, processed_image)
                print(f"Processed and saved to {output_path}")
                # Uncomment below lines to display the image
                plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                plt.show()

# Define input and output directories
input_dir = r"data\ptolemy\organised"
output_dir = r"data\ptolemy\processed"

# Define minimum contour areas
min_contour_areas = {}

# Call the process_directory function
process_directory(input_dir, output_dir, min_contour_areas)
