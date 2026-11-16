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
        img.paste(cropped_img, (xmin, ymin, xmax, ymax))
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    elif num_bboxes > 1:
        available_bboxes = tree.findall('.//object')
        random.shuffle(available_bboxes)
        for i in range(min(num_bboxes, len(available_bboxes))):
            box = available_bboxes[i]  # Select the next available bounding box

            # Get the bounding box coordinates
            xmin = int(float(box.find('bndbox/xmin').text)) // 4
            ymin = int(float(box.find('bndbox/ymin').text)) // 4
            xmax = int(float(box.find('bndbox/xmax').text)) // 4
            ymax = int(float(box.find('bndbox/ymax').text)) // 4

            # Compute the width and height of the bounding box
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            # Find a new position for the bounding box
            for _ in range(1000):  # Try up to 1000 times
                # Generate random coordinates for the new position. Make sure new_xmin starts at least 100 pixels from the left side of the image.
                # And ends within the left 3/4 of the image.
                new_xmin = random.randint(100, (3 * width // 4) - bbox_width - 1)
                new_ymin = random.randint(0, height - bbox_height - 1)
                new_xmax = new_xmin + bbox_width
                new_ymax = new_ymin + bbox_height

                # Generate a bounding box for the new position
                new_bbox = [new_xmin, new_ymin, new_xmax, new_ymax]

                # Check if the bounding box overlaps with or is too close to any existing bounding box
                if any(
                    (
                        abs(new_xmin - xmin) < bbox_width + 50 and abs(new_ymin - ymin) < bbox_height + 50
                    )
                    for xmin, ymin, xmax, ymax in existing_bboxes
                ):
                    continue  # Try again with a new position

                # If we get here, the new position is valid
                break
            else:
                # If we didn't break from the loop, we couldn't find a valid position
                print(
                    f"Could not find a valid position for the bounding box in {image_path} after 1000 attempts. Skipping this bounding box."
                )
                continue  # Skip this bounding box and move on to the next one
            existing_bboxes.append(new_bbox)
            cropped_img = img.crop((xmin, ymin, xmax, ymax))
            img.paste(cropped_img, (new_xmin, new_ymin, new_xmax, new_ymax))
            rect = patches.Rectangle(
                (new_xmin, new_ymin), bbox_width, bbox_height, linewidth=1, edgecolor='r', facecolor='none'
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
for i in range(1000):
    new_image_path = os.path.join(mock_labels_dir, f"new_image_{i}.jpg")
    new_xml_path = os.path.join(xml_dir, f"new_annotation_{i}.xml")
    move_annotation(image_path, xml_path, new_image_path, new_xml_path, i)
