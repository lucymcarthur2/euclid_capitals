import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image


def preprocess_image_remove_max_noise(gray_image):
    # Apply initial threshold to remove maximum noise
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_noise_threshold = 50
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > small_noise_threshold]
    mask = np.zeros_like(opening)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    processed_image = cv2.bitwise_and(gray_image, mask)

    # Convert to PIL image, apply Gaussian blur, and convert back to numpy array
    pil_image = Image.fromarray(processed_image)
    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=1))
    processed_image = np.array(pil_image)

    return processed_image

def process_image_and_remove_small_contours(image, min_contour_area=1000, threshold_value=180, use_adaptive_threshold=False):
    # Define different border sizes for each side
    top_border_size = 27
    bottom_border_size = 16
    left_border_size = 23
    right_border_size = 25
    
    # Crop the image according to the border sizes
    cropped_image = image[top_border_size:-bottom_border_size, left_border_size:-right_border_size]
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    gray_image = preprocess_image_remove_max_noise(gray_image)
    
    # Apply Gaussian Blur for noise reduction after initial contour removal
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
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
    
    # Add borders with specific sizes
    final_image_with_border = cv2.copyMakeBorder(
        final_binary_image,
        top_border_size, bottom_border_size, left_border_size, right_border_size,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    return final_image_with_border

def process_directory(input_directory, output_directory, min_contour_area=200):
    image_paths = glob(os.path.join(input_directory, '*.png'))
    for image_path in image_paths:
        image = cv2.imread(image_path)
        processed_image = process_image_and_remove_small_contours(
            image, 
            min_contour_area=min_contour_area
        )
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, processed_image)
        print(f"Processed and saved to {output_path}")
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.show()

# Directories
input_dir = r'data\initial_capitals_improved\capitals'
output_dir = r'data\initial_capitals_improved\capitals_processed'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process images in the directory
process_directory(input_dir, output_dir)
