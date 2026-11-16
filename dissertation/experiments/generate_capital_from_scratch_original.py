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

FONT_PATHS = [
    "times.ttf",
    "calibri.ttf",
    "cambria.ttc",
    "georgia.ttf",
    "verdana.ttf",
    # Ensure these paths are correct for your system
]

def draw_text_with_border(image, text, font_size=380, border_thickness=5):
    font_path = random.choice(FONT_PATHS)
    font = ImageFont.truetype(font_path, font_size)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Adjust font size to fit text within the image
    text_width, text_height = draw.textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size -= 10
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
        if font_size <= 10:  # Prevents font size from becoming too small
            break

    # Randomize position while keeping text within the image
    text_x = random.randint(0, max(pil_image.width - text_width, 0))
    text_y = random.randint(0, max(pil_image.height - text_height, 0))

    draw.text((text_x, text_y), text, font=font, fill="white", stroke_width=border_thickness, stroke_fill="black")

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_text(image, text, font_size=380):
    font_path = random.choice(FONT_PATHS)
    font = ImageFont.truetype(font_path, font_size)
    
    pil_image = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0, 0, 0))
    draw = ImageDraw.Draw(pil_image)

    # Adjust font size to fit text within the image
    text_width, text_height = draw.textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size -= 10
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
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

def generate_composition(image_path, init_width, init_height, letter_to_generate, num_letters):
    base_image = cv2.imread(image_path)
    if base_image is None:
        raise Exception(f"Failed to load image from {image_path}")

    compositions = []
    for _ in range(num_letters):
        x = random.randint(0, base_image.shape[1] - init_width)
        y = random.randint(0, base_image.shape[0] - init_height)
        background = base_image[y:y + init_height, x:x + init_width]

        text_image, holes_mask = draw_text(np.zeros_like(background), letter_to_generate, font_size=380)

        composition_type = random.choice(["text_only", "image_background", "background_holes"])
        if composition_type == "text_only":
            final_image = text_image
        elif composition_type == "image_background":
            final_image = draw_text_with_border(background, letter_to_generate, font_size=380, border_thickness=5)
        elif composition_type == "background_holes":
            final_image = np.zeros_like(background)
            final_image[holes_mask == 255] = background[holes_mask == 255]
            final_image[holes_mask == 0] = text_image[holes_mask == 0]

        # Increase contrast
        alpha = 1.5 # Contrast control
        beta = 0    # Brightness control
        adjusted = cv2.convertScaleAbs(final_image, alpha=alpha, beta=beta)

        # Apply thresholding
        gray_final = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        #_, thresh_final = cv2.threshold(gray_final, 10, 255, cv2.THRESH_BINARY)
        #thresh_final_color = cv2.cvtColor(thresh_final, cv2.COLOR_GRAY2BGR)

        compositions.append(gray_final)

    return compositions



def main():
    base_dir = 'data/manual_capitals'
    img_path = 'data/initial_capitals_improved/extracted_object_32811_017.jpg_4.png'
    init_width, init_height = 308, 366

    # Define specific numbers for certain classes, default for others
    num_letters_dict = {}  # Example: 100 for 'I', 200 for 'S'
    default_num_letters = 300  # Default value for other classes

    for letter in 'ABCDEFGHILMNOPQRST':
        folder_path = os.path.join(base_dir, letter)
        os.makedirs(folder_path, exist_ok=True)

        # Get the specific number of images for the class, or use the default
        num_letters = num_letters_dict.get(letter, default_num_letters)

        compositions = generate_composition(img_path, init_width, init_height, letter, num_letters)

        for idx, comp in enumerate(compositions):
            file_name = f"{letter}_{idx}.png"
            file_path = os.path.join(folder_path, file_name)
            cv2.imwrite(file_path, comp)

if __name__ == "__main__":
    main()

