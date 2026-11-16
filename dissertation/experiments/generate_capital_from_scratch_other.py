
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
    r'fonts\RECOLETA-REGULARDEMO.otf',
    r'fonts\LEMOSTY.otf'
    
]

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


def draw_text(image, text, font_size_range=(380, 420), max_rotation=5):
    font_path = random.choice(FONT_PATHS)
    font_size = random.randint(*font_size_range)
    font = ImageFont.truetype(font_path, font_size)
    
    pil_image = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0, 0, 0))
    draw = ImageDraw.Draw(pil_image)

    # Find the optimal font size that fits the image
    text_width, text_height = draw.textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)

    # Center the text
    x = (pil_image.width - text_width) // 2
    y = (pil_image.height - text_height) // 2
    draw.text((x, y), text, font=font, fill="white")

    text_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray_text = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    _, binary_text = cv2.threshold(gray_text, 10, 255, cv2.THRESH_BINARY)

    # Random rotation
    #rotation = random.uniform(-max_rotation, max_rotation)
    #pil_image = pil_image.rotate(rotation, expand=True)
    
    # Creating a mask for holes
    contours, _ = cv2.findContours(binary_text, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes_mask = np.zeros_like(binary_text)
    for i, contour in enumerate(contours):
        if i % 2 == 1:  # Creating holes in every alternate contour
            cv2.drawContours(holes_mask, [contour], -1, 255, -1)

    binary_text_color = cv2.cvtColor(binary_text, cv2.COLOR_GRAY2BGR)

    return binary_text_color, holes_mask


def draw_text_with_border(image, text, font_size_range=(380, 420), border_thickness=6, max_rotation=5):
    font_path = random.choice(FONT_PATHS)
    font_size = random.randint(*font_size_range)
    font = ImageFont.truetype(font_path, font_size)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Find the optimal font size that fits the image
    text_width, text_height = draw.textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)

    # Center the text
    x = (pil_image.width - text_width) // 2
    y = (pil_image.height - text_height) // 2
    draw.text((x, y), text, font=font, fill="white", stroke_width=border_thickness, stroke_fill="black")

    # Random rotation
    rotation = random.uniform(-max_rotation, max_rotation)
    pil_image = pil_image.rotate(rotation, expand=True)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)




def add_black_squares(image, square_size_range=(70, 100), num_squares_range=(1, 3)):
    square_size = random.randint(*square_size_range)
    num_squares = random.randint(*num_squares_range)
    for _ in range(num_squares):
        x = random.randint(0, image.shape[1] - square_size)
        y = random.randint(0, image.shape[0] - square_size)
        image[y:y + square_size, x:x + square_size] = 0
    return image

def generate_composition(image_path, init_width, init_height, letter_to_generate, num_letters, generate_letter=True):
    base_image = cv2.imread(image_path)
    
    if base_image is None:
        raise Exception(f"Failed to load image from {image_path}")

    compositions = []
    for _ in range(num_letters):
        x = random.randint(0, base_image.shape[1] - init_width)
        y = random.randint(0, base_image.shape[0] - init_height)
        background = base_image[y:y + init_height, x:x + init_width]

        if generate_letter:
            text_image, holes_mask = draw_text(np.zeros_like(background), letter_to_generate, font_size_range=(380, 470))

            composition_type = random.choice(["text_only", "image_background", "background_holes"])
            if composition_type == "text_only":
                final_image = draw_text_with_border(np.zeros_like(background), letter_to_generate, font_size_range=(380, 470), border_thickness=5)
                final_image = add_black_squares(final_image)
            elif composition_type == "image_background":
                if random.choice([True, False]):
                    background = process_image_and_remove_small_contours(background)
                final_image = draw_text_with_border(background, letter_to_generate)
            elif composition_type == "background_holes":
                final_image = np.zeros_like(background)
                final_image[holes_mask == 255] = background[holes_mask == 255]
                final_image[holes_mask == 0] = text_image[holes_mask == 0]
        else:
            final_image = background

        # Increase contrast
        alpha = 1.5  # Contrast control
        beta = 0     # Brightness control
        adjusted = cv2.convertScaleAbs(final_image, alpha=alpha, beta=beta)

        gray_final = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

        compositions.append(gray_final)

    return compositions

def main():
    base_dir = 'data/manual_capitals'
    img_path = 'data/initial_capitals_improved/extracted_object_32811_017.jpg_4.png'
    init_width, init_height = 308, 366

    # Create 'Other' directory
    other_folder_path = os.path.join(base_dir, 'Other')
    os.makedirs(other_folder_path, exist_ok=True)
    num_other_images = 500

    other_compositions = generate_composition(img_path, init_width, init_height, '', num_other_images, generate_letter=False)
    for idx, comp in enumerate(other_compositions):
        file_name = f"Other_{idx}.png"
        file_path = os.path.join(other_folder_path, file_name)
        cv2.imwrite(file_path, comp)

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        folder_path = os.path.join(base_dir, letter)
        os.makedirs(folder_path, exist_ok=True)
        num_letters = 600 if letter in ['T', 'N', 'L', 'O', 'P', 'S'] else 300
        compositions = generate_composition(img_path, init_width, init_height, letter, num_letters)
        for idx, comp in enumerate(compositions):
            file_name = f"{letter}_{idx}.png"
            file_path = os.path.join(folder_path, file_name)
            cv2.imwrite(file_path, comp)

if __name__ == "__main__":
    main()