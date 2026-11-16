import os
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

# Define the path to the directory containing the images
path = r"data/initial_capitals/specific"

# Define the path for the new directory to save the processed images
new_directory = os.path.join(path, "processed")

# Create the new directory if it doesn't exist
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

# Iterate through all files in the directory
for filename in os.listdir(path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image file
        image_path = os.path.join(path, filename)
        image = Image.open(image_path)
        
        # Increase contrast
        image = image.convert("RGB")
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Convert to grayscale
        image = image.convert("L")
        
        # Quarter the size
        image = image.resize((int(image.width / 4), int(image.height / 4)))
        
        # Apply threshold
        threshold_value = 128
        image = image.point(lambda p: p > threshold_value and 255)
        
        # Smooth the image with Gaussian blur
        image = image.filter(ImageFilter.GaussianBlur(radius=0.1))
        
        # Invert the image
        image = ImageOps.invert(image)
        
        # Save the processed image in the new directory
        new_filename = os.path.splitext(filename)[0] + "_processed.png"
        new_path = os.path.join(new_directory, new_filename)
        image.save(new_path)
        
        # Close the image file
        image.close()
