import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from PIL import ImageEnhance

#32811_025.jpg \\ squished_capitals.xml
#32811_079.jpg  // labelled_xml_dup.xml
# blank rhs 32811_249.jpg , 37 fixed x pos, 32811_264 for lhs, 320 x pos
# Paths to necessary files and directories
false_positive_image_path = r"data\jpegs_cropped\32811_025.jpg"
false_positive_xml_path = r"data\squished_capitals.xml"
blank_image_path = r"data\jpegs_cropped\32811_264.jpg"

output_image_dir = r"data\mock_labels"
output_xml_dir = r"data\xmls"

# Open necessary images
false_positive_image = Image.open(false_positive_image_path)
blank_image = Image.open(blank_image_path)

width, height = false_positive_image.size
false_positive_image = false_positive_image.resize((width // 4, height // 4))
blank_image = blank_image.resize((width // 4, height // 4))
false_positive_image = false_positive_image.convert('RGBA')
blank_image = blank_image.convert('RGBA')

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

# Function to check overlap
def does_overlap(box1, box2, overlap=20):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if x1 > x2 + w2 - overlap or x1 + w1 < x2 + overlap or y1 > y2 + h2 - overlap or y1 + h1 < y2 + overlap:
        return False
    return True


#  Create 500 new images
for i in range(1000):
    new_image = blank_image.copy()
    annotation = ET.Element('annotation')

    num_bboxes = random.randint(1, 4)  # Randomize the number of bounding boxes between 1 and 4

    used_positions = []  # Store positions of already placed bounding boxes

    fixed_pos_x = 320 #340  # Fixed x coordinate for all bounding boxes

    for _ in range(num_bboxes):
        bbox = random.choice(bounding_boxes)  # Randomly choose a bounding box
        false_positive_area = false_positive_image.crop(bbox).copy()  # Extract false positive area
        w, h = false_positive_area.size

        # Randomly select position in blank image
        pos_x = fixed_pos_x  # Use the fixed x coordinate
        pos_y = random.randint(0, 800 - h)

        # Check if new position overlaps with existing images
        while any(does_overlap((pos_x, pos_y, w, h), used_position, overlap=10) for used_position in used_positions):

            pos_y = random.randint(0, 800 - h)

        used_positions.append((pos_x, pos_y, w, h))

        # Resize the false positive area
        resize_ratio = random.uniform(0.9, 1)  # Adjust this value to your preference
        resized_area = false_positive_area.resize((int(w * resize_ratio), int(h * resize_ratio)), Image.ANTIALIAS)

        # Rotate the false positive area
        rotate_deg = random.uniform(0, 5)
        rotated_area = resized_area.rotate(rotate_deg, resample=Image.BILINEAR, expand=True)
        
        # Apply opacity to the rotated area
        rotated_area_with_opacity = Image.new('RGBA', rotated_area.size)
        
        # Generate a random opacity once for the entire image
        opacity = random.uniform(0.6, 1)
        
        for x in range(rotated_area.width):
            for y in range(rotated_area.height):
                r, g, b, a = rotated_area.getpixel((x, y))
                rotated_area_with_opacity.putpixel((x, y), (r, g, b, int(a * opacity)))

        
        # Composite the rotated image onto the base image
        new_image.alpha_composite(rotated_area_with_opacity, (pos_x, pos_y))

    # Convert to greyscale
    new_image = new_image.convert("L")

    # Save new image
    new_image_path = os.path.join(output_image_dir, f'overlap_{i}_capital_aligned_lhs.jpg')
    new_image.save(new_image_path)

    # Display the new image
    plt.imshow(new_image, cmap='gray')
    plt.show()
    
    # Create corresponding XML
    folder = ET.SubElement(annotation, 'folder')
    folder.text = output_image_dir

    filename = ET.SubElement(annotation, 'filename')
    filename.text = f'overlap_{i}_capital.png'

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
    new_xml_path = os.path.join(output_xml_dir, f'overlap_{i}_capital_aligned_lhs.xml')
    tree.write(new_xml_path)
