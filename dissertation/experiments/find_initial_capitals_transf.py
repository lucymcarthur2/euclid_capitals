import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import xml.etree.ElementTree as ET
import matplotlib.patches as patches

def check_directory(directory=r"data\tiff_cropped"):
    tiff_files = [f for f in os.listdir(directory) if f.endswith('.tif') or f.endswith('.tiff')]
    if not tiff_files:
        raise ValueError("No TIFF files found in the directory.")
    with Image.open(os.path.join(directory, tiff_files[0])) as img:
        width, height = img.size
    for file in tiff_files:
        with Image.open(os.path.join(directory, file)) as img:
            assert img.size[0] == width and img.size[1] == height, f"Image {file} is not the same size as the first image."
    print("All images are the same size.")

def move_annotation(image_path, xml_path, new_image_path, new_xml_path, image_number):
    img = Image.open(image_path)
    tree = ET.parse(xml_path)

    width, height = img.size
    img = img.resize((width // 4, height // 4))
    width, height = img.size
    img = img.convert('L')
    fig, ax = plt.subplots(1)

    existing_bboxes = []

    num_bboxes = random.randint(1, 4)  # Randomize the number of bounding boxes capped at 4
    if num_bboxes == 1:
        box = random.choice(tree.findall('.//object'))  # Randomly choose a bounding box
        # Get the bounding box coordinates
        xmin = int(float(box.find('bndbox/xmin').text)) // 4
        ymin = int(float(box.find('bndbox/ymin').text)) // 4
        xmax = int(float(box.find('bndbox/xmax').text)) // 4
        ymax = int(float(box.find('bndbox/ymax').text)) // 4
        existing_bboxes.append([xmin, ymin, xmax, ymax])
        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        
        # Random rotation and resizing
        angle = random.uniform(-5, 5)
        cropped_img = cropped_img.rotate(angle, fillcolor='white')
        scale = random.uniform(0.8, 1.2)
        new_size = (int(cropped_img.width * scale), int(cropped_img.height * scale))
        cropped_img = cropped_img.resize(new_size)
        
        img.paste(cropped_img, (xmin, ymin, xmax, ymax))
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    img.save(new_image_path)
    tree.write(new_xml_path)
    ax.imshow(img, cmap='gray')
    plt.show()


#32811_079.jpg  // labelled_xml_dup.xml
#32811_287.jpg // labelled_2.xml
xml_dir = r"data\xmls"
mock_labels_dir = r"data\mock_labels"
image_path = r"data\jpegs_cropped\for_labelling.jpg"
xml_path = r"data\labelled_xml_dup.xml"
for i in range(300):
    new_image_path = os.path.join(mock_labels_dir, f"new_image_{i}.jpg")
    new_xml_path = os.path.join(xml_dir, f"new_annotation_{i}.xml")
    move_annotation(image_path, xml_path, new_image_path, new_xml_path, i)
