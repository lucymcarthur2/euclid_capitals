import os
import shutil
from PIL import Image

def calculate_average_dimensions(directory):
    dimensions = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                with Image.open(os.path.join(directory, filename)) as img:
                    dimensions.append(img.size)
            except Exception as e:
                print(f"Error opening '{filename}': {e}")

    if not dimensions:
        return None, None

    avg_width = sum(d[0] for d in dimensions) / len(dimensions)
    avg_height = sum(d[1] for d in dimensions) / len(dimensions)

    filtered_dimensions = [d for d in dimensions if (0.5 * avg_width < d[0] < 1.5 * avg_width) and (0.5 * avg_height < d[1] < 1.5 * avg_height)]

    if not filtered_dimensions:
        return avg_width, avg_height

    final_avg_width = sum(d[0] for d in filtered_dimensions) / len(filtered_dimensions)
    final_avg_height = sum(d[1] for d in filtered_dimensions) / len(filtered_dimensions)

    return final_avg_width, final_avg_height

EXTRACTED_OBJECTS_DIR = r"data\compilatio\extracted\all"
JUNK_DIR = os.path.join(EXTRACTED_OBJECTS_DIR, 'junk')
CAPITALS_DIR = os.path.join(EXTRACTED_OBJECTS_DIR, 'capitals')

os.makedirs(JUNK_DIR, exist_ok=True)
os.makedirs(CAPITALS_DIR, exist_ok=True)

avg_width, avg_height = calculate_average_dimensions(EXTRACTED_OBJECTS_DIR)

width_lower_threshold = 0.7 * avg_width
width_upper_threshold = 1.35 * avg_width
height_lower_threshold = 0.7 * avg_height
height_upper_threshold = 1.35 * avg_height

print(width_lower_threshold, width_upper_threshold)
print(height_lower_threshold, height_upper_threshold)

for filename in os.listdir(EXTRACTED_OBJECTS_DIR):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(EXTRACTED_OBJECTS_DIR, filename)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if (width_lower_threshold <= width <= width_upper_threshold) and \
                   (height_lower_threshold <= height <= height_upper_threshold):
                    shutil.copy(image_path, CAPITALS_DIR)
                else:
                    shutil.copy(image_path, JUNK_DIR)
        except Exception as e:
            print(f"Error processing '{filename}': {e}")

print("Image sorting completed.")
