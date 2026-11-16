# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:27:52 2024

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

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

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

    # Draw text
    draw.text((text_x, text_y), text, font=font, fill="white")

    # Convert to OpenCV format
    text_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return text_image

def generate_composition(image_path, init_width, init_height, letter_to_generate, num_letters):
    compositions = []
    for _ in range(num_letters):
        background = np.zeros((init_height, init_width, 3), dtype=np.uint8)
        text_image = draw_text(background, letter_to_generate, font_size=380)
        compositions.append(text_image)

    return compositions

def main():
    base_dir = 'data/simple_caps'
    init_width, init_height = 308, 366

    num_letters_dict = {}
    default_num_letters = 1000

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        folder_path = os.path.join(base_dir, letter)
        os.makedirs(folder_path, exist_ok=True)

        num_letters = num_letters_dict.get(letter, default_num_letters)

        compositions = generate_composition(None, init_width, init_height, letter, num_letters)

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
