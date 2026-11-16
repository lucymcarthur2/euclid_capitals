import os
import random
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt

# Paths to necessary files and directories
false_positive_image_path = r"data\jpegs_cropped\32811_065.jpg"
false_positive_xml_path = r"data\false_positives.xml"
blank_image_path = r"data\jpegs_cropped\32811_014.jpg"

output_image_dir = r"data\mock_labels"
output_xml_dir = r"data\xmls"



# Open necessary images
false_positive_image = Image.open(false_positive_image_path).convert('L')  # convert to grayscale
blank_image = Image.open(blank_image_path).convert('L')  # convert to grayscale

# Resize images
width, height = false_positive_image.size
false_positive_image = false_positive_image.resize((width // 4, height // 4))  # you can change the size to your preference
blank_image = blank_image.resize((width // 4, height // 4))  # you can change the size to your preference



# Parse false positive XML to get bounding boxes
tree = ET.parse(false_positive_xml_path)
root = tree.getroot()
bounding_boxes = []
for obj in root.findall('object'):
    bbox = obj.find('bndbox')
    xmin = int(float(bbox.find('xmin').text)) //4
    ymin = int(float(bbox.find('ymin').text)) //4 
    xmax = int(float(bbox.find('xmax').text)) //4
    ymax = int(float(bbox.find('ymax').text)) //4
    bounding_boxes.append((xmin, ymin, xmax, ymax))

def is_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

# Create 500 new images
for i in range(500):
    new_image = blank_image.copy().convert('RGBA')  # convert to RGBA
    new_boxes = []
    annotation = ET.Element('annotation')

    for bbox in bounding_boxes:
        # Extract false positive area
        false_positive_area = false_positive_image.crop(bbox).convert('RGBA')  # convert to RGBA
        w, h = false_positive_area.size

        # Randomly select rotation angle and resize factor
        rotation_angle = random.uniform(0, 360)  # Angle in degrees
        resize_factor = random.uniform(0.5, 1.5)  # Resize between 50% and 150% of original size

        # Rotate and resize false positive area
        false_positive_area = false_positive_area.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)
        false_positive_area = false_positive_area.resize((int(w * resize_factor), int(h * resize_factor)), Image.ANTIALIAS)

        w, h = false_positive_area.size  # Update the width and height after resizing

        # Randomly select position in blank image
        pos_x = pos_y = None
        for _ in range(1000):
            max_x = new_image.width - w
            max_y = new_image.height - h
            new_pos_x = random.randint(90, max_x)  # Minimum 300 pixels from the left
            new_pos_y = random.randint(30, max_y)  # Minimum 100 pixels from the top

            # Check if bounding box overlaps with any existing box
            if not any(is_overlap((new_pos_x, new_pos_y, w, h), box) for box in new_boxes):
                pos_x = new_pos_x
                pos_y = new_pos_y
                new_boxes.append((pos_x, pos_y, w, h))
                break

        if pos_x is None:
            continue  # Couldn't find a non-overlapping position within 1000 attempts

        # Paste rotated and resized false positive area onto blank image
        new_image.paste(false_positive_area, (pos_x, pos_y), false_positive_area)  # the third argument is used as a mask

        # Create 'object' element and append it to 'annotation'
        object_element = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object_element, 'name')
        name.text = 'false_positive'  # Replace with your actual class label
        pose = ET.SubElement(object_element, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object_element, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(object_element, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(object_element, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(pos_x)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(pos_y)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(pos_x + false_positive_area.width)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(pos_y + false_positive_area.height)

    # Save new image
    # Convert new_image back to 'L' before saving
    new_image = new_image.convert('L')
    new_image_path = os.path.join(output_image_dir, f'blank_{i}.jpg')
    #new_image.save(new_image_path)

    # Display the new image
    plt.imshow(new_image, cmap = 'gray')
    plt.show()

    # Create the rest of the XML
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

    # Save new XML
    #new_xml_path = os.path.join(output_xml_dir, f'blank_{i}.xml')
    #tree.write(new_xml_path)
