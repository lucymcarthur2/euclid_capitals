import os
from PIL import Image
import shutil
import numpy as np

# Define the image directory and create output directories
image_dir = r"data\initial_capitals"
specific_dir = os.path.join(image_dir, "specific")
other_dir = os.path.join(image_dir, "other")
os.makedirs(specific_dir, exist_ok=True)
os.makedirs(other_dir, exist_ok=True)

# Define the specific dimensions and a percentage threshold for tolerance
specific_width, specific_height = 296, 343
threshold = 0.20  # Images within 10% of the specific dimensions will be considered
#image _150 S with 175 x 191     242 x 303
# Process the images
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_dir, filename)
        with Image.open(img_path) as img:
            width, height = img.size

            # If the image dimensions are within the tolerance level of the specific dimensions
            if (specific_width * (1 - threshold) <= width <= specific_width * (1 + threshold) and
                specific_height * (1 - threshold) <= height <= specific_height * (1 + threshold)):
                dest_dir = specific_dir
            else:
                dest_dir = other_dir

            dest_path = os.path.join(dest_dir, filename)
            try:
                shutil.copy(img_path, dest_path)
                print(f"Copied {img_path} to {dest_path}")
            except Exception as e:
                print(f"Error copying {img_path} to {dest_path}: {e}")
