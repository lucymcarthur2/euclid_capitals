import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

def preprocess_image_remove_max_noise(gray_image):
    # Reduced blur compared to the original, but higher than the last revision
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Slightly larger kernel for morphological opening
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_noise_threshold = 50  # This value can be adjusted as needed
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > small_noise_threshold]
    mask = np.zeros_like(opening)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    processed_image = cv2.bitwise_and(gray_image, mask)
    return processed_image

def process_image_and_remove_small_contours(image, min_contour_area=1000, threshold_value=180, use_adaptive_threshold=False, border_size=15):
    cropped_image = image[border_size:-border_size, border_size:-border_size]
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    gray_image = preprocess_image_remove_max_noise(gray_image)
    if use_adaptive_threshold:
        img_binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        _, img_binary = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    # Intermediate kernel size for less aggressive morphological opening
    kernel = np.ones((2, 2), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray_image)
    if contours and hierarchy is not None:
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_contour_area and hierarchy[0][idx][3] == -1:
                cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
    extracted_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    _, final_binary_image = cv2.threshold(extracted_image, 128, 255, cv2.THRESH_BINARY)
    final_image_with_border = cv2.copyMakeBorder(final_binary_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return final_image_with_border

def process_directory(input_directory, output_directory, contour_area_dict):
    for root, dirs, files in os.walk(input_directory):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            output_subdir = subdir_path.replace(input_directory, output_directory)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            image_paths = glob(os.path.join(subdir_path, '*.png'))
            for image_path in image_paths:
                image = cv2.imread(image_path)
                min_contour_area = contour_area_dict.get(subdir, 1000)
                processed_image = process_image_and_remove_small_contours(
                    image, 
                    min_contour_area=min_contour_area
                )
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_subdir, filename)
                cv2.imwrite(output_path, processed_image)
                print(f"Processed and saved to {output_path}")
                plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                plt.show()

min_contour_areas = {
    'A': 300, 'B': 200, 'C': 100, 'D': 200, 'E': 30,
    'I': 280, 'M': 120, 'O': 250, 'P': 300, 'Q': 180,
    'R': 170, 'S': 0, 'T': 350, 'F': 140, 'N': 0, 'L': 110
}

input_dir = r'data\organised_capitals'
output_dir = r'data\extracted_letters'
process_directory(input_dir, output_dir, min_contour_areas)
