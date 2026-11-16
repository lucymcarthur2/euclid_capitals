import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
import shutil

# Paths to necessary directories
negative_image_dir = r"data\negative_samples"
output_image_dir = r"data\mock_labels_2"
output_xml_dir = r"data\xmls_2"

def load_negative_samples(negative_image_dir):
    negative_images = []
    filenames = []
    for filename in os.listdir(negative_image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            negative_images.append(os.path.join(negative_image_dir, filename))
            filenames.append(filename)
    return negative_images, filenames

negative_samples, filenames = load_negative_samples(negative_image_dir)

# Dimensions to match positive training samples
target_width, target_height = 812, 1078

for i, (image_path, filename) in enumerate(zip(negative_samples, filenames)):
    new_image_name = f"negative_{i}.jpg"
    new_image_path = os.path.join(output_image_dir, new_image_name)
    
    # Open image and resize it to match the training image size
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure image is in RGB mode
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)  # Resize to match training size
        img.save(new_image_path)

    # Create a corresponding XML annotation file with no ROIs
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = os.path.basename(output_image_dir)
    filename_xml = ET.SubElement(annotation, 'filename')
    filename_xml.text = new_image_name
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    width_node = ET.SubElement(size, 'width')
    width_node.text = str(target_width)
    height_node = ET.SubElement(size, 'height')
    height_node.text = str(target_height)
    depth_node = ET.SubElement(size, 'depth')
    depth_node.text = '3'  # RGB image depth
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    # Save the XML file with the same base name as the image
    new_xml_name = f"negative_{i}.xml"
    new_xml_path = os.path.join(output_xml_dir, new_xml_name)
    tree = ET.ElementTree(annotation)
    tree.write(new_xml_path, encoding="utf-8", xml_declaration=True)

    # Convert XML tree to string and print it (optional)
    generated_xml_content = ET.tostring(annotation, encoding="utf-8").decode("utf-8")
    print(f"Generated XML for {new_image_name}:\n{generated_xml_content}\n")
