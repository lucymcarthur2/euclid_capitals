import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np


# Parameters
x_coordinate = 30  # Set the desired x coordinate for the superimposed images
resize_range = (0.8, 1.2)  # Adjust the range for resizing (width and height) as needed


#.280.jpg , x=310
#.273.jpg , x=30
# Paths to necessary files and directories
false_positive_image_path = r"data\jpegs_cropped\32811_025.jpg"
false_positive_xml_path = r"data\squished_capitals.xml"
blank_image_path = r"data\jpegs_cropped\32811_273.jpg"

output_image_dir = r"data\mock_labels"
output_xml_dir = r"data\xmls"

# Open necessary images
false_positive_image = Image.open(false_positive_image_path)
blank_image = Image.open(blank_image_path)

width, height = false_positive_image.size
false_positive_image = false_positive_image.resize((width // 4, height // 4))
blank_image = blank_image.resize((width // 4, height // 4))
false_positive_image = false_positive_image.convert('L')
blank_image = blank_image.convert('L')

# Parse false positive XML to get bounding boxes
tree = ET.parse(false_positive_xml_path)
root = tree.getroot()
bounding_boxes = []
for obj in root.findall('object'):
    bbox = obj.find('bndbox')
    xmin = int(float(bbox.find('xmin').text)) // 4
    ymin = int(float(bbox.find('ymin').text))  // 4
    xmax = int(float(bbox.find('xmax').text))  // 4
    ymax = int(float(bbox.find('ymax').text))  // 4
    bounding_boxes.append((xmin, ymin, xmax, ymax))

def is_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

# Create 200 new images
for i in range(200):
    new_image = blank_image.copy()
    new_boxes = []
    annotation = ET.Element('annotation')

    for bbox in bounding_boxes:
        # Extract false positive area
        false_positive_area = false_positive_image.crop(bbox)
        w, h = false_positive_area.size

        # Randomly select position in blank image
        pos_y = None
        for _ in range(1000):
            max_y = new_image.height - h
            new_pos_y = random.randint(0, max_y)

            # Check if bounding box overlaps with any existing box
            if not any(is_overlap((x_coordinate, new_pos_y, w, h), box) for box in new_boxes):
                pos_y = new_pos_y
                new_boxes.append((x_coordinate, pos_y, w, h))
                break

        if pos_y is None:
            continue  # Couldn't find a non-overlapping position within 1000 attempts

        # Resize and rotate the false positive area
        rotate_deg = random.uniform(0, 5)
        width_resize_ratio = height_resize_ratio = random.uniform(*resize_range)  # Adjust these values to your preference
        false_positive_area = false_positive_area.resize((int(w * width_resize_ratio), int(h * height_resize_ratio)), Image.ANTIALIAS)
        
        # Convert image to have an alpha channel and rotate
        false_positive_area = false_positive_area.convert('RGBA')
        false_positive_area = false_positive_area.rotate(rotate_deg, expand=1)
        
        # Create a mask using the alpha channel
        mask = Image.new('L', false_positive_area.size)
        mask.paste(false_positive_area.split()[3], (0,0))  # Paste the alpha channel to the new mask
        
        # Convert rotated image back to RGB
        false_positive_area = false_positive_area.convert('RGB')
        
        # Apply mask to new_image
        new_image.paste(false_positive_area, (x_coordinate, pos_y), mask)

        # Create 'object' element and append it to 'annotation'
        object_element = ET.SubElement(annotation, 'object')

        # Add 'name' element (class label)
        name = ET.SubElement(object_element, 'name')
        name.text = 'false_positive'  # Replace with your actual class label

        # Add 'pose', 'truncated', 'difficult' elements (typically not used, but required by the schema)
        pose = ET.SubElement(object_element, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object_element, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(object_element, 'difficult')
        difficult.text = '0'

        # Add 'bndbox' element with 'xmin', 'ymin', 'xmax', 'ymax' elements
        bndbox = ET.SubElement(object_element, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(x_coordinate)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(pos_y)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(x_coordinate + false_positive_area.width)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(pos_y + false_positive_area.height)

    # Save new image
    new_image_path = os.path.join(output_image_dir, f'aligned_{i}_capital_2.jpg')
    new_image.save(new_image_path)

    # Display the new image
    plt.imshow(new_image, cmap='gray')
    plt.show()

    # Create corresponding XML
    folder = ET.SubElement(annotation, 'folder')
    folder.text = output_image_dir

    filename = ET.SubElement(annotation, 'filename')
    filename.text = f'blank_{i}.jpg'

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    width_node = ET.SubElement(size, 'width')
    width_node.text = str(new_image.width)

    height_node = ET.SubElement(size, 'height')
    height_node.text = str(new_image.height)

    depth_node = ET.SubElement(size, 'depth')
    depth_node.text = '3'

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    tree = ET.ElementTree(annotation)

    #Save new XML
    new_xml_path = os.path.join(output_xml_dir, f'aligned_{i}_capital_2.xml')
    tree.write(new_xml_path)
