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

def draw_helix_curve_with_random_side_offset(image, curve_thickness=10):
    height, width = image.shape[:2]
    center_x = width // 2

    amplitude = random.randint(min(width, height) // 3, min(width, height) // 1)
    frequency = 2 * np.pi / (height / 2)  # Frequency of the wave
    side_offset = np.random.uniform(0, np.pi)

    for y in range(height):
        x1 = int(center_x + amplitude * np.sin(frequency * y))
        x2 = int(center_x + amplitude * np.sin(frequency * y + side_offset))
        cv2.line(image, (x1, y), (x1, y), (255, 255, 255), curve_thickness)
        cv2.line(image, (x2, y), (x2, y), (255, 255, 255), curve_thickness)

    return image

def process_image_and_remove_small_contours(image, threshold_value=180, min_contour_area=1000, use_adaptive_threshold=False, border_size=10):
    cropped_image = image[border_size:-border_size, border_size:-border_size]
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    if use_adaptive_threshold:
        img_binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        _, img_binary = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
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

def draw_text(image, text, font_size=250):
    font_path = random.choice(FONT_PATHS)
    font = ImageFont.truetype(font_path, font_size)
    pil_image = Image.new('RGB', (image.shape[1], image.shape[0]), color=(0, 0, 0))
    draw = ImageDraw.Draw(pil_image)

    text_width, text_height = textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size -= 10
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = textsize(text, font=font)
        if font_size <= 10:
            break

    text_x = random.randint(0, max(pil_image.width - text_width, 0))
    text_y = random.randint(0, max(pil_image.height - text_height, 0))

    draw.text((text_x, text_y), text, font=font, fill="white")
    text_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray_text = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
    _, binary_text = cv2.threshold(gray_text, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_text, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes_mask = np.zeros_like(binary_text)
    for i, contour in enumerate(contours):
        if i % 2 == 1:
            cv2.drawContours(holes_mask, [contour], -1, 255, -1)

    binary_text_color = cv2.cvtColor(binary_text, cv2.COLOR_GRAY2BGR)
    return binary_text_color, holes_mask

def draw_text_with_border(image, text, font_size=250, border_thickness=5):
    font_path = random.choice(FONT_PATHS)
    font = ImageFont.truetype(font_path, font_size)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    text_width, text_height = textsize(text, font=font)
    while text_width > pil_image.width or text_height > pil_image.height:
        font_size -= 10
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = textsize(text, font=font)
        if font_size <= 10:
            break

    text_x = random.randint(0, max(pil_image.width - text_width, 0))
    text_y = random.randint(0, max(pil_image.height - text_height, 0))

    draw.text((text_x, text_y), text, font=font, fill="white", stroke_width=border_thickness, stroke_fill="black")

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def generate_composition(image_path, init_width, init_height, letter_to_generate, num_letters, line_thickness=10):
    base_image = cv2.imread(image_path)
    
    if base_image is None:
        raise Exception(f"Failed to load image from {image_path}")

    compositions = []
    for _ in range(num_letters):
        x = random.randint(0, base_image.shape[1] - init_width)
        y = random.randint(0, base_image.shape[0] - init_height)
        background = base_image[y:y + init_height, x:x + init_width]

        # Randomly decide whether to use the background
        use_background = random.random() < 1  # 50% chance to use the background
        if use_background:
            text_image, holes_mask = draw_text(np.zeros_like(background), letter_to_generate, font_size=250)

            composition_type = random.choice(["image_background"])
            if composition_type == "text_only":
                final_image = draw_text_with_border(np.zeros_like(background), letter_to_generate, font_size=250, border_thickness=5)
            elif composition_type == "image_background":
                if random.choice([True, True]):  # 100% chance to apply processing
                    background = process_image_and_remove_small_contours(background)
                final_image = draw_text_with_border(background, letter_to_generate, font_size=250)
            else:
                final_image = np.zeros_like(background)
                final_image[holes_mask == 255] = background[holes_mask == 255]
                final_image[holes_mask == 0] = text_image[holes_mask == 0]
                compositions.append(final_image)
                continue
        else:
            final_image = draw_text_with_border(np.zeros_like(background), letter_to_generate, font_size=250, border_thickness=5)

        alpha = 1.5
        beta = 0
        adjusted = cv2.convertScaleAbs(final_image, alpha=alpha, beta=beta)
        gray_final = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        compositions.append(gray_final)

    return compositions

def main():
    base_dir = 'data/simple_caps'
    img_path = 'data/extracted_object_32811_017.jpg_4.png'
    init_width, init_height = 308, 366

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        folder_path = os.path.join(base_dir, letter)
        os.makedirs(folder_path, exist_ok=True)

        compositions = generate_composition(img_path, init_width, init_height, letter, 500)

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
