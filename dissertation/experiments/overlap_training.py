import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# Paths to necessary files and directories
false_positive_image_path = r"data\jpegs_cropped\32811_025.jpg"
false_positive_xml_path = r"data\squished_capitals.xml"
blank_image_path = r"data\jpegs_cropped\32811_234.jpg"

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
    ymin = int(float(bbox.find('ymin').text)) // 4
    xmax = int(float(bbox.find('xmax').text)) // 4
    ymax = int(float(bbox.find('ymax').text)) // 4
    bounding_boxes.append((xmin, ymin, xmax, ymax))

# Create 200 new images
for i in range(500):
    new_image = blank_image.copy()
    new_boxes = []
    annotation = ET.Element('annotation')

    num_bboxes = random.randint(1, 4)  # Randomize the number of bounding boxes between 1 and 4

    for _ in range(num_bboxes):
        bbox = random.choice(bounding_boxes)  # Randomly choose a bounding box
        # Extract false positive area
        false_positive_area = false_positive_image.crop(bbox)
        w, h = false_positive_area.size

        # Randomly select position in blank image
        pos_x = 420 - w
        pos_y = random.randint(100, 800 - h)

        new_boxes.append((pos_x, pos_y, w, h))

        # Resize the false positive area
        resize_ratio = random.uniform(0.9, 1)  # Adjust this value to your preference
        resized_area = false_positive_area.resize((int(w * resize_ratio), int(h * resize_ratio)), Image.ANTIALIAS)

        # Rotate the false positive area
        rotate_deg = random.uniform(0, 5)
        rotated_area = resized_area.rotate(rotate_deg, resample=Image.BILINEAR, expand=True)

        # Create a mask using the alpha channel of the rotated area
        mask = rotated_area.split()[-1]

        # Apply random opacity to the rotated area
        #opacity = 255
        #rotated_area.putalpha(opacity)

        # Apply mask to new_image
        new_image.paste(rotated_area, (pos_x, pos_y), mask)

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
        xmin.text = str(pos_x)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(pos_y)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(pos_x + rotated_area.width)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(pos_y + rotated_area.height)

    # Save new image
    #new_image_path = os.path.join(output_image_dir, f'overlap_{i}_capital_2.jpg')
    #new_image.save(new_image_path)

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

    # Save new XML
    #new_xml_path = os.path.join(output_xml_dir, f'overlap_{i}_capital_2.xml')
    #tree.write(new_xml_path)
