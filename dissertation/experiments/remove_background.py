# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:55:21 2024

@author: Lucy McArthur
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:09:54 2024

@author: Lucy McArthur
"""

import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

def preprocess_image_remove_noise(image, noise_threshold=300):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small noise by contour area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > noise_threshold]

    # Create mask and apply to image to remove small noise
    mask = np.zeros_like(image)
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

def process_image_and_remove_small_contours(image_path, noise_threshold=300):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None  # If the image couldn't be read, return None

    # Define border sizes
    top_border_size = 27
    bottom_border_size = 16
    left_border_size = 23
    right_border_size = 30

    # Crop the image to remove the border
    cropped_image = image[top_border_size:-bottom_border_size, left_border_size:-right_border_size]

    # Preprocess and remove noise
    processed_image = preprocess_image_remove_noise(cropped_image, noise_threshold)

    # Binarize the processed image to ensure it is fully binary
    _, final_binary_image = cv2.threshold(processed_image, 127, 255, cv2.THRESH_BINARY)

    # Add back the borders removed at the start
    final_image_with_border = cv2.copyMakeBorder(
        final_binary_image,
        top_border_size, bottom_border_size, left_border_size, right_border_size,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return final_image_with_border

def process_directory(input_directory, output_directory, contour_area_dict, intensity):
    # Adjust noise_threshold based on intensity (0 to 100 scale)
    noise_threshold = int(300 * (intensity / 100))

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
                image = cv2.imread(image_path)
                min_contour_area = contour_area_dict.get(subdir, 0)
                processed_image = process_image_and_remove_small_contours(image_path, noise_threshold=noise_threshold)
                if processed_image is None:
                    continue
                filename = os.path.basename(image_path)
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_processed{ext}"
                output_path = os.path.join(output_subdir, new_filename)
                cv2.imwrite(output_path, processed_image)
                print(f"Processed and saved to {output_path}")
                # Uncomment below lines to display the image
                # plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                # plt.show()

        # Process files in the current directory (if there are no subdirectories)
        if not dirs:
            output_subdir = root.replace(input_directory, output_directory)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            image_paths = glob(os.path.join(root, '*.png'))
            for image_path in image_paths:
                image = cv2.imread(image_path)
                min_contour_area = contour_area_dict.get(root, 0)
                processed_image = process_image_and_remove_small_contours(image_path, noise_threshold=noise_threshold)
                if processed_image is None:
                    continue
                filename = os.path.basename(image_path)
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_processed{ext}"
                output_path = os.path.join(output_subdir, new_filename)
                #cv2.imwrite(output_path, processed_image)
                #print(f"Processed and saved to {output_path}")
                # Uncomment below lines to display the image
                plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                plt.show()

# Define input and output directories
input_dir = r"data\music_score\extracted"
output_dir = r"data\mixed_backgrounds"

# Define minimum contour areas
min_contour_areas = {}

# Set intensity (0 to 100)
intensity = 0  # Adjust this value to control the amount of background removal

# Call the process_directory function
process_directory(input_dir, output_dir, min_contour_areas, intensity)
