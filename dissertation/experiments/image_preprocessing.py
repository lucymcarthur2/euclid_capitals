import os
import zipfile
from PIL import Image, UnidentifiedImageError

# Define source directory and zip file
src_dir = r"data"
zip_file_name = 'tiffs.zip'
dst_dir = r"data\tiff_resized"

# Extract the zip file
with zipfile.ZipFile(os.path.join(src_dir, zip_file_name), 'r') as zip_ref:
    zip_ref.extractall(src_dir)

# Define the directory with the extracted tiffs
tiff_dir = os.path.join(src_dir, 'tiffs')

# Make sure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# Get a list of all TIFF files in the tiff directory
tiff_files = [f for f in os.listdir(tiff_dir) if f.endswith('.tif') or f.endswith('.tiff')]

# Ensure there are TIFF files in the source directory
if not tiff_files:
    raise ValueError("No TIFF files found in the source directory.")

# Loop through all TIFF files
for file in tiff_files:
    try:
        # Open each image
        with Image.open(os.path.join(tiff_dir, file)) as img:
            # Downsize the image
            new_size = (img.size[0] // 4, img.size[1] // 4)  # 1/4 of original size
            img_resized = img.resize(new_size)

            # Convert the image to grayscale
            img_gray = img_resized.convert('L')

            # Save the resized grayscale image to the destination directory
            #img_gray.save(os.path.join(dst_dir, file))
    except UnidentifiedImageError:
        print(f"Skipping invalid image file: {file}")


# Define the source directory and target directory
src_dir = r"data\tiff_resized"
target_dir = r"data\jpegs_resized"

# Create target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Get a list of all TIFF files in the directory
tiff_files = [f for f in os.listdir(src_dir) if f.endswith('.tif') or f.endswith('.tiff')]

# Loop over the TIFF files
for tiff_file in tiff_files:
    # Open the TIFF file with PIL
    img = Image.open(os.path.join(src_dir, tiff_file))
    
    # Define the name and path for the new JPEG file
    jpeg_file = tiff_file.rsplit(".", 1)[0] + '.jpg'
    jpeg_path = os.path.join(target_dir, jpeg_file)
    
    # Save the image as a JPEG file
    #img.save(jpeg_path, 'JPEG')

print("Conversion completed!")
