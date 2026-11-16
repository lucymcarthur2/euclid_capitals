# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:33:52 2024

@author: Lucy McArthur
"""

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

BACKGROUND_IMAGE_PATH = r'data/extracted_object_32811_017.jpg_4.png'

def add_background(image, background_image_path):
    if not os.path.isfile(background_image_path):
        raise FileNotFoundError(f"The background image '{background_image_path}' was not found.")
    
    # Load the background image
    background = cv2.imread(background_image_path)
    if background is None:
        return image  # Return original image if background loading fails

    # Convert to grayscale
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_background, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, hierarchy = cv2.findContours(adaptive_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(gray_background)

    # Draw the largest contours based on area, excluding inner contours
    min_contour_area = 1000  # Minimum contour area to keep
    if contours and hierarchy is not None:
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_contour_area and hierarchy[0][idx][3] == -1:  # Not an inner contour
                cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

    # Bitwise-and to extract relevant parts of the background
    extracted_background = cv2.bitwise_and(background, background, mask=mask)

    # Determine random position for the overlay
    max_x = max(0, extracted_background.shape[1] - image.shape[1])
    max_y = max(0, extracted_background.shape[0] - image.shape[0])
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Crop the extracted background to fit the image size
    cropped_background = extracted_background[y:y+image.shape[0], x:x+image.shape[1]]

    # Overlay the processed background onto the image
    overlay = np.where(cropped_background != 0, cropped_background, image)
    
    return overlay

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def draw_text(image, text, font_size=380, border_size=5):
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

    # Draw border by drawing the text multiple times slightly offset
    for dx in range(-border_size, border_size + 1):
        for dy in range(-border_size, border_size + 1):
            if dx != 0 or dy != 0:  # Skip the center
                draw.text((text_x + dx, text_y + dy), text, font=font, fill="black")

    # Draw the main text on top of the border
    draw.text((text_x, text_y), text, font=font, fill="white")

    # Convert to OpenCV format
    text_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return text_image

def generate_composition(init_width, init_height, letter_to_generate, num_letters):
    compositions = []
    for _ in range(num_letters):
        background = np.zeros((init_height, init_width, 3), dtype=np.uint8)
        
        # Decide whether to use the background image
        use_background = random.random() < 1
        if use_background:
            background = add_background(background, BACKGROUND_IMAGE_PATH)

        text_image = draw_text(background, letter_to_generate, font_size=250)
        compositions.append(text_image)

    return compositions

def main():
    base_dir = 'data/simple_caps'
    init_width, init_height = 308, 366

    num_letters_dict = {}
    default_num_letters = 1

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        folder_path = os.path.join(base_dir, letter)
        os.makedirs(folder_path, exist_ok=True)

        num_letters = num_letters_dict.get(letter, default_num_letters)

        compositions = generate_composition(init_width, init_height, letter, num_letters)

        for idx, comp in enumerate(compositions):
            plt.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
            plt.title(f"Composition {letter}_{idx}")
            plt.axis('off')
            plt.show()
            file_name = f"{letter}_{idx}.png"
            file_path = os.path.join(folder_path, file_name)
            #cv2.imwrite(file_path, comp)

if __name__ == "__main__":
    main()
