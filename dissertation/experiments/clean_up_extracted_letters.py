import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

#rescribe OCR

def preprocess_image_remove_max_noise(gray_image):
    # Apply a Gaussian blur to the image to remove noise
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Perform morphological operations to remove small noise
    # Use a larger kernel size to remove more noise
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # Find contours and filter out small noise
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Define a more aggressive threshold for small noise to be removed
    small_noise_threshold = 50  # Increase the threshold to remove even smaller contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > small_noise_threshold]
    # Create an empty mask to draw the filtered contours
    mask = np.zeros_like(opening)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    # Apply the mask to the original grayscale image
    processed_image = cv2.bitwise_and(gray_image, mask)
    return processed_image

def process_image_and_remove_small_contours(image, threshold_value=180, min_contour_area=1000, use_adaptive_threshold=False, border_size=18):
    # Crop the image by 40 pixels from each border
    cropped_image = image[border_size:-border_size, border_size:-border_size]

    # Convert to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Additional noise removal
    gray_image = preprocess_image_remove_max_noise(gray_image)

    # Apply thresholding
    if use_adaptive_threshold:
        img_binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        _, img_binary = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Mild noise removal using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for drawing contours
    mask = np.zeros_like(gray_image)

    # Draw the largest contours based on the specified criteria, excluding inner contours
    if contours and hierarchy is not None:
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_contour_area and hierarchy[0][idx][3] == -1:
                cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

    # Bitwise-and to extract the relevant parts of the image
    extracted_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    # Final conversion to binary
    _, final_binary_image = cv2.threshold(extracted_image, 128, 255, cv2.THRESH_BINARY)

    # Add a black border around the image
    final_image_with_border = cv2.copyMakeBorder(final_binary_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return final_image_with_border


def process_directory(input_directory, output_directory, threshold_dict):
    # Walk through the input directory
    for root, dirs, files in os.walk(input_directory):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            output_subdir = subdir_path.replace(input_directory, output_directory)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            image_paths = glob(os.path.join(subdir_path, '*.png'))  # Adjust as needed
            for image_path in image_paths:
                image = cv2.imread(image_path)

                # Retrieve the threshold value for the current subdir
                threshold_value = threshold_dict.get(subdir, 180)  # Default is 180

                processed_image = process_image_and_remove_small_contours(image, threshold_value=threshold_value)
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_subdir, filename)
                #cv2.imwrite(output_path, processed_image)
                print(f"Processed and saved to {output_path}")
                # Display the processed image
                plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                #plt.title(f"Processed Image: {filename}")
                plt.show()
# Example usage
threshold_values = {
    # Adjust the threshold values as needed for each class
    'A': 90, 'B': 180, 'C': 1, 'D': 180, 'E': 30
    ,
    'I': 150, 'M': 120, 'O': 160, 'P': 180, 'Q': 180,
    'R': 170, 'S':  5, 'T': 180, 'F':140, 'N':100, 'L':110
}



# Example usage:
input_dir = r'data\organised_capitals'
output_dir = r'data\extracted_letters'
process_directory(input_dir, output_dir, threshold_values)