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
import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt




FONT_PATHS = [
    "times.ttf",
    #"calibri.ttf",
    "cambria.ttc",
    "georgia.ttf",
    #"verdana.ttf",
    "BASKVILL.TTF",
    r'C:\Windows\Fonts\trebuc.ttf',
    r'C:\Windows\Fonts\centaur.ttf',
    # Ensure these paths are correct for your system
]


def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def draw_helix_curve_with_random_side_offset(image, curve_thickness=10):
    height, width = image.shape[:2]
    center_x = width // 2

    # Random amplitude between a specified range, e.g., 1/8 to 1/4 of the minimum dimension
    amplitude = random.randint(min(width, height) // 3, min(width, height) // 1)

    frequency = 2 * np.pi / (height / 2)  # Frequency of the wave

    # Random phase offset
    side_offset = np.random.uniform(0, np.pi)

    for y in range(height):
        x1 = int(center_x + amplitude * np.sin(frequency * y))
        x2 = int(center_x + amplitude * np.sin(frequency * y + side_offset))  # Adding random side offset
        cv2.line(image, (x1, y), (x1, y), (255, 255, 255), curve_thickness)
        cv2.line(image, (x2, y), (x2, y), (255, 255, 255), curve_thickness)

    return image




def process_image_and_remove_small_contours(image, threshold_value=180, min_contour_area=1000, use_adaptive_threshold=False, border_size=10):
    # Crop the image by border size from each border
    cropped_image = image[border_size:-border_size, border_size:-border_size]

    # Convert to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    if use_adaptive_threshold:
        # Adaptive thresholding
        img_binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        # Fixed thresholding with the specified threshold value
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
            if area > min_contour_area and hierarchy[0][idx][3] == -1:  # Check if not an inner contour
                cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

    # Bitwise-and to extract the relevant parts of the image
    extracted_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    # Final conversion to binary
    _, final_binary_image = cv2.threshold(extracted_image, 128, 255, cv2.THRESH_BINARY)

    # Add a black border around the image if needed
    final_image_with_border = cv2.copyMakeBorder(final_binary_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return final_image_with_border


def draw_text(image, text, font_size_range=(380, 420)):
    font_path = random.choice(FONT_PATHS)
    
    # Choose a random font size within the specified range
    font_size = random.randint(*font_size_range)
    font = ImageFont.truetype(font_path, font_size)
    
    pil_image = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0, 0, 0))
    draw = ImageDraw.Draw(pil_image)

    # Adjust font size to fit text within the image
    text_width, text_height = textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size = random.randint(*font_size_range)  # Retry with a different random size
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = textsize(text, font=font)
        if font_size <= font_size_range[0]:  # Prevents font size from becoming too small
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

def draw_text_with_border(image, text, font_size_range=(380, 420), border_thickness=6):
    font_path = random.choice(FONT_PATHS)
    font_size = random.randint(*font_size_range)
    font = ImageFont.truetype(font_path, font_size)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Adjust font size to fit text within the image
    text_width, text_height = textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size = random.randint(*font_size_range)  # Retry with a different random size
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = textsize(text, font=font)
        if font_size <= font_size_range[0]:  # Prevents font size from becoming too small
            break

    # Randomize position while keeping text within the image
    text_x = random.randint(0, max(pil_image.width - text_width, 0))
    text_y = random.randint(0, max(pil_image.height - text_height, 0))

    draw.text((text_x, text_y), text, font=font, fill="white", stroke_width=border_thickness, stroke_fill="black")

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)



def add_black_squares(image, square_size_range=(70, 100), num_squares_range=(1, 3)):
    square_size = random.randint(*square_size_range)
    num_squares = random.randint(*num_squares_range)
    for _ in range(num_squares):
        x = random.randint(0, image.shape[1] - square_size)
        y = random.randint(0, image.shape[0] - square_size)
        image[y:y + square_size, x:x + square_size] = 0
    return image

def generate_composition(image_path, init_width, init_height, letter_to_generate, num_letters, line_thickness=10):
    base_image = cv2.imread(image_path)
    
    if base_image is None:
        raise Exception(f"Failed to load image from {image_path}")

    compositions = []
    for _ in range(num_letters):
        x = random.randint(0, base_image.shape[1] - init_width)
        y = random.randint(0, base_image.shape[0] - init_height)
        background = base_image[y:y + init_height, x:x + init_width]
        text_image, holes_mask = draw_text(np.zeros_like(background), letter_to_generate, font_size_range=(380, 470))

        composition_type = random.choice(["image_background", "background_holes"])
        if composition_type == "text_only":
            final_image = draw_text_with_border(np.zeros_like(background), letter_to_generate, font_size_range=(380, 470), border_thickness=5)
            final_image = add_black_squares(final_image)
        elif composition_type == "image_background":
            if random.choice([True, False]):  # 50% chance to apply processing
                background = process_image_and_remove_small_contours(background)
            final_image = draw_text_with_border(background, letter_to_generate)
        else:  # For background_holes, skip helix curve
            final_image = np.zeros_like(background)
            final_image[holes_mask == 255] = background[holes_mask == 255]
            final_image[holes_mask == 0] = text_image[holes_mask == 0]
            compositions.append(final_image)
            continue

        # Increase contrast
        alpha = 1.5  # Contrast control
        beta = 0     # Brightness control
        adjusted = cv2.convertScaleAbs(final_image, alpha=alpha, beta=beta)

        # Apply thresholding
        gray_final = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

        # Random chance (5%) to draw the helix curve
        #if random.random() < 0.0:
        #    gray_final = draw_helix_curve_with_random_side_offset(gray_final)

        compositions.append(gray_final)

    return compositions





def main():
    base_dir = 'data/manual_capitals'
    img_path = 'data/extracted_object_32811_017.jpg_4.png'
    init_width, init_height = 308, 366

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        folder_path = os.path.join(base_dir, letter)
        os.makedirs(folder_path, exist_ok=True)

        compositions = generate_composition(img_path, init_width, init_height, letter, 5000)

        for idx, comp in enumerate(compositions):
            # Display the image using Matplotlib
            plt.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
            plt.title(f"Composition {letter}_{idx}")
            plt.axis('off')
            plt.show()
            file_name = f"{letter}_{idx}.png"
            file_path = os.path.join(folder_path, file_name)
            #cv2.imwrite(file_path, comp)

if __name__ == "__main__":
    main()


