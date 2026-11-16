# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 08:51:13 2023

@author: Lucy McArthur
"""

import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

FONT_PATHS = [
    r"fonts\trebuc.ttf",
    r"fonts\cambria.ttc",
    r"fonts\georgia.ttf",
    r"fonts\BASKVILL.TTF",
    r'fonts\CENTAUR.ttf',
]

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def add_black_noise_to_background(image, noise_fraction=0.05):
    """
    Add black noise to the white sections of the background of the image.
    noise_fraction: fraction of white pixels to set to black
    """
    noisy_image = image.copy()
    white_pixels = np.where(np.all(noisy_image == [255, 255, 255], axis=-1))
    num_white_pixels = len(white_pixels[0])
    num_black_pixels = int(noise_fraction * num_white_pixels)

    for _ in range(num_black_pixels):
        idx = random.randint(0, num_white_pixels - 1)
        y, x = white_pixels[0][idx], white_pixels[1][idx]
        noisy_image[y, x] = [0, 0, 0]

    return noisy_image

def add_white_noise_to_border(image, border_mask, noise_fraction=0.1):
    """
    Add white noise to the border of the letters.
    noise_fraction: fraction of border pixels to set to white
    """
    noisy_image = image.copy()
    border_pixels = np.where(border_mask == 255)
    num_border_pixels = len(border_pixels[0])
    num_white_pixels = int(noise_fraction * num_border_pixels)

    for _ in range(num_white_pixels):
        idx = random.randint(0, num_border_pixels - 1)
        y, x = border_pixels[0][idx], border_pixels[1][idx]
        noisy_image[y, x] = [255, 255, 255]

    return noisy_image

def draw_text_with_border(image, text, font_size=380, border_thickness=5):
    font_path = random.choice(FONT_PATHS)
    font = ImageFont.truetype(font_path, font_size)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Adjust font size to fit text within the image
    text_width, text_height = textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size -= 10
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = textsize(text, font=font)
        if font_size <= 10:  # Prevents font size from becoming too small
            break

    # Randomize position while keeping text within the image
    text_x = random.randint(0, max(pil_image.width - text_width, 0))
    text_y = random.randint(0, max(pil_image.height - text_height, 0))

    # Draw text with a border (stroke_fill set to grey)
    draw.text((text_x, text_y), text, font=font, fill="white", stroke_width=border_thickness, stroke_fill="grey")

    # Convert back to OpenCV format
    noisy_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Create a mask for the border
    border_mask = np.zeros_like(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY))
    pil_border_mask = Image.fromarray(border_mask)
    draw_border_mask = ImageDraw.Draw(pil_border_mask)
    draw_border_mask.text((text_x, text_y), text, font=font, fill="white", stroke_width=border_thickness, stroke_fill="white")
    border_mask = np.array(pil_border_mask)

    # Add white noise to the border
    noisy_image = add_white_noise_to_border(noisy_image, border_mask)

    return noisy_image

def draw_text(image, text, font_size=380):
    font_path = random.choice(FONT_PATHS)
    font = ImageFont.truetype(font_path, font_size)
    
    pil_image = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0, 0, 0))
    draw = ImageDraw.Draw(pil_image)

    # Adjust font size to fit text within the image
    text_width, text_height = textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size -= 10
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = textsize(text, font=font)
        if font_size <= 10:  # Prevents font size from becoming too small
            break

    # Randomize position while keeping text within the image
    text_x = random.randint(0, max(pil_image.width - text_width, 0))
    text_y = random.randint(0, max(pil_image.height - text_height, 0))

    # Draw text and apply thresholding
    draw.text((text_x, text_y), text, font=font, fill="white")
    text_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray_text = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    _, binary_text = cv2.threshold(gray_text, 10, 255, cv2.THRESH_BINARY)

    # Use the binary text as a mask to create holes if needed
    contours, _ = cv2.findContours(binary_text, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes_mask = np.zeros_like(binary_text)
    for i, contour in enumerate(contours):
        if i % 2 == 1:
            cv2.drawContours(holes_mask, [contour], -1, 255, -1)

    # Convert the binary text back to BGR format for composition
    binary_text_color = cv2.cvtColor(binary_text, cv2.COLOR_GRAY2BGR)

    return binary_text_color, holes_mask

def add_severe_distortions(image):
    # Combine multiple distortions for more severe effects
    # Binarize and remove contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    image = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply Gaussian Blur
    image = cv2.GaussianBlur(image, (15, 15), 0)
    
    # Apply heavy warp transformation
    h, w = image.shape[:2]
    src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst_points = src_points + np.random.normal(0, 50, src_points.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    image = cv2.warpPerspective(image, M, (w, h))

    # Add Gaussian noise
    noise = np.random.normal(0, 60, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)

    # Apply limited rotation with cropping to avoid black areas
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    image = crop_to_content(image)

    # Apply random zoom-in
    zoom_factor = random.uniform(1.2, 1.5)
    image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    if zoom_factor > 1:
        start_x = (image.shape[1] - w) // 2
        start_y = (image.shape[0] - h) // 2
        image = image[start_y:start_y + h, start_x:start_x + w]

    # Ensure the image is not too light by adjusting brightness and contrast
    alpha = 1.2  # Contrast control (lower value for less contrast)
    beta = -30   # Brightness control (negative value to darken)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Add black noise to the background
    image = add_black_noise_to_background(image)

    return image

def crop_to_content(image):
    """Crop the image to the content to remove black borders after rotation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return image[y:y + h, x:x + w]

def preprocess_image_remove_max_noise(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    small_noise_threshold = 30  
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > small_noise_threshold]
    
    mask = np.zeros_like(opening)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    processed_image = cv2.bitwise_and(gray_image, mask)
    return processed_image

def erode_text(image, min_iterations=2, max_iterations=5):
    iterations = random.randint(min_iterations, max_iterations)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return eroded

def add_noise_to_text(image, noise_level=0.05):
    noisy_image = image.copy()
    noise = np.random.normal(loc=0, scale=255 * noise_level, size=image.shape).astype(np.uint8)
    noisy_image = cv2.add(noisy_image, noise)
    return noisy_image

def process_image_and_remove_small_contours(image, min_contour_area=150, threshold_value=180, use_adaptive_threshold=False):
    # Remove border cropping and use the entire image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = preprocess_image_remove_max_noise(gray_image)
    
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    if use_adaptive_threshold:
        img_binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        _, img_binary = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((2, 2), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray_image)
    if contours and hierarchy is not None:
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_contour_area and hierarchy[0][idx][3] == -1:
                cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
    
    extracted_image = cv2.bitwise_and(image, image, mask=mask)
    
    kernel = np.ones((4, 4), np.uint8)
    connected_image = cv2.dilate(extracted_image, kernel, iterations=1)
    
    _, final_binary_image = cv2.threshold(connected_image, 128, 255, cv2.THRESH_BINARY)
    
    return final_binary_image

def add_random_white_border(image, min_thickness=1, max_thickness=10):
    """
    Add a random white border around the image.
    The border may appear on random edges.
    """
    top_border = random.randint(min_thickness, max_thickness) if random.choice([True, False]) else 0
    bottom_border = random.randint(min_thickness, max_thickness) if random.choice([True, False]) else 0
    left_border = random.randint(min_thickness, max_thickness) if random.choice([True, False]) else 0
    right_border = random.randint(min_thickness, max_thickness) if random.choice([True, False]) else 0

    white = [255, 255, 255]
    bordered_image = cv2.copyMakeBorder(
        image,
        top_border,
        bottom_border,
        left_border,
        right_border,
        cv2.BORDER_CONSTANT,
        value=white
    )

    return bordered_image

def generate_composition(image_path, init_width, init_height, letter_to_generate, num_letters):
    base_image = cv2.imread(image_path)
    if base_image is None:
        raise Exception(f"Failed to load image from {image_path}")

    compositions = []
    for _ in range(num_letters):
        x = random.randint(0, base_image.shape[1] - init_width)
        y = random.randint(0, base_image.shape[0] - init_height)
        background = base_image[y:y + init_height, x:x + init_width]
        
        background = add_severe_distortions(background)
        background = process_image_and_remove_small_contours(background)

        text_image, holes_mask = draw_text(np.zeros_like(background), letter_to_generate, font_size=380)
        text_image = erode_text(text_image, min_iterations=2, max_iterations=7)
        text_image = add_noise_to_text(text_image, noise_level=0.3)

        composition_type = random.choice(["text_only"])
        if composition_type == "text_only":
            final_image = text_image
        elif composition_type == "image_background":
            final_image = draw_text_with_border(background, letter_to_generate, font_size=380, border_thickness=5)
        elif composition_type == "background_holes":
            final_image = np.zeros_like(background)
            final_image[holes_mask == 255] = background[holes_mask == 255]
            final_image[holes_mask == 0] = text_image[holes_mask == 0]

        # Apply random white border around the image
        final_image = add_random_white_border(final_image)

        alpha = 1.2
        beta = -20
        adjusted = cv2.convertScaleAbs(final_image, alpha=alpha, beta=beta)

        gray_final = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        _, harsh_binarized = cv2.threshold(gray_final, 200, 255, cv2.THRESH_BINARY)

        compositions.append(harsh_binarized)

    return compositions

def main():
    base_dir = 'data/simple_caps'
    img_path = 'data/extracted_object_32811_017.jpg_4.png'
    init_width, init_height = 308, 366

    num_letters_dict = {}
    default_num_letters = 1000

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        folder_path = os.path.join(base_dir, letter)
        os.makedirs(folder_path, exist_ok=True)

        num_letters = num_letters_dict.get(letter, default_num_letters)

        compositions = generate_composition(img_path, init_width, init_height, letter, num_letters)

        for idx, comp in enumerate(compositions):
            plt.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
            plt.title(f"Composition {letter}_{idx}")
            plt.axis('off')
            #plt.show()
            file_name = f"{letter}_{idx}.png"
            file_path = os.path.join(folder_path, file_name)
            cv2.imwrite(file_path, comp)

if __name__ == "__main__":
    main()
