import fitz  # PyMuPDF
from PIL import Image
import os
import io

def pdf_to_jpgs(pdf_path, rotation_angle=0, dpi=300, jpeg_quality=95):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Extract directory and base name for the PDF path
    base_name = os.path.basename(pdf_path)
    output_folder_name = os.path.splitext(base_name)[0]
    directory = os.path.dirname(pdf_path)
    output_folder_path = os.path.join(directory, output_folder_name)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # Iterate over each page in the PDF
    for page_number in range(len(pdf_document)):
        # Get the page
        page = pdf_document[page_number]
        
        # Render page to an image with higher resolution
        pix = page.get_pixmap(dpi=dpi)
        
        # Store the image as a PNG in memory
        img_data = pix.tobytes("png")
        
        # Open the image using PIL
        image = Image.open(io.BytesIO(img_data))
        
        # Rotate the image if a rotation angle is specified
        if rotation_angle != 0:
            image = image.rotate(rotation_angle, expand=True)
        
        # Save the image to the output folder as JPG with higher quality
        image_jpg_path = os.path.join(output_folder_path, f"page_{page_number + 1}.jpg")
        image.save(image_jpg_path, "JPEG", quality=jpeg_quality)
    
    # Close the PDF document
    pdf_document.close()
    print(f"Converted PDF pages to JPGs in folder: {output_folder_path}")
# Replace with your desired rotation angle in degrees




# Example usage:
pdf_path = r'data\ptolemy.pdf'  # Replace with your PDF file path
output_folder = r'data\ptolemy'  # Replace with your desired output folder path
rotation_angle = 0  # Replace with your desired rotation angle in degrees

pdf_to_jpgs(pdf_path, rotation_angle)
