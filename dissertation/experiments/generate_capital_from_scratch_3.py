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
    "BASKVILL.TTF",
    r'C:\Windows\Fonts\trebuc.ttf',
    r'C:\Windows\Fonts\centaur.ttf',
    r'C:\Windows\Fonts\trebuc.ttf'
    # Ensure these paths are correct for your system
]

def threshold_and_remove_small_contours(image, min_contour_area=50):
    # Convert to grayscale and apply threshold to remove grey
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)  # Keeping only white and black

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for small white contours
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            cv2.drawContours(mask, [contour], -1, 0, -1)  # Fill small contours with black

    # Apply the mask
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def draw_text(image, text, font_size_range=(380, 420)):
    font_path = random.choice(FONT_PATHS)
    
    # Choose a random font size within the specified range
    font_size = random.randint(*font_size_range)
    font = ImageFont.truetype(font_path, font_size)
    
    pil_image = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0, 0, 0))
    draw = ImageDraw.Draw(pil_image)

    # Adjust font size to fit text within the image
    text_width, text_height = draw.textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size = random.randint(*font_size_range)  # Retry with a different random size
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
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

def draw_text_with_border(image, text, font_size_range=(380, 420), border_thickness=2):
    font_path = random.choice(FONT_PATHS)
    font_size = random.randint(*font_size_range)
    font = ImageFont.truetype(font_path, font_size)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Adjust font size to fit text within the image
    text_width, text_height = draw.textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size = random.randint(*font_size_range)  # Retry with a different random size
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
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

def generate_composition(image_path, init_width, init_height, letter_to_generate, num_letters):
    base_image = cv2.imread(image_path)
    
    if base_image is None:
        raise Exception(f"Failed to load image from {image_path}")

    compositions = []
    for _ in range(num_letters):
        x = random.randint(0, base_image.shape[1] - init_width)
        y = random.randint(0, base_image.shape[0] - init_height)
        background = base_image[y:y + init_height, x:x + init_width]
        text_image, holes_mask = draw_text(np.zeros_like(background), letter_to_generate, font_size_range=(380, 470))

        composition_type = random.choice(["text_only", "image_background", "background_holes"]) #
        if composition_type == "text_only":
            final_image = draw_text_with_border(np.zeros_like(background), letter_to_generate, font_size_range=(380, 470), border_thickness=5)
            final_image = add_black_squares(final_image)
        elif composition_type == "image_background":
            processed_background = threshold_and_remove_small_contours(background)
            final_image = draw_text_with_border(processed_background, letter_to_generate, font_size_range=(380, 470), border_thickness=5)
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

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        folder_path = os.path.join(base_dir, letter)
        os.makedirs(folder_path, exist_ok=True)

        compositions = generate_composition(img_path, init_width, init_height, letter, 300)

        for idx, comp in enumerate(compositions):
            file_name = f"{letter}_{idx}.png"
            file_path = os.path.join(folder_path, file_name)
            cv2.imwrite(file_path, comp)

if __name__ == "__main__":
    main()