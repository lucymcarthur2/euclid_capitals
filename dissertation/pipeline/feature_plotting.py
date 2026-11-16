import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def align_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    aligned_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    return aligned_img1, img2

def visualize_orb_matches(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    return Image.fromarray(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))

def overlay_with_white_background(img1, img2):
    mask1 = img1 < 255
    mask2 = img2 < 255
    
    overlay = np.ones((img2.shape[0], img2.shape[1], 3), dtype=np.uint8) * 255
    overlay[mask1] = [255, 0, 127]  # Pink for the aligned first image
    overlay[mask2] = [0, 255, 255]  # Cyan for the original second image
    
    return overlay

def display_with_matplotlib(title, image):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    
def compute_difference(img1, img2):
    """Compute the absolute difference between two images."""
    diff = cv2.absdiff(img1, img2)
    return diff



# Paths to your images
img1_path = "data\\extracted_diagrams\\extracted_object_32811_241.jpg_0.png"
img2_path = "data\\extracted_diagrams\\extracted_object_32811_242.jpg_0.png"

aligned_img1, original_img2 = align_images(img1_path, img2_path)
display_with_matplotlib("ORB Feature Matches", visualize_orb_matches(aligned_img1, original_img2))
display_with_matplotlib("Overlay With White Background", overlay_with_white_background(aligned_img1, original_img2))

save_path = "data\\"


# Display and save the ORB Feature Matches
orb_matches_img = visualize_orb_matches(aligned_img1, original_img2)
display_with_matplotlib("ORB Feature Matches", orb_matches_img)


# Display and save the Overlay With White Background
overlay_img = overlay_with_white_background(aligned_img1, original_img2)
display_with_matplotlib("Overlayed Images", overlay_img)

# Save the ORB Feature Matches (using PIL's save method)
orb_matches_img.save(save_path + "ORB_feature_matches.png")

# Compute, display, and save the difference between the two images.
diff_img = compute_difference(aligned_img1, original_img2)
display_with_matplotlib("Difference Between Images", diff_img)
cv2.imwrite(save_path + "difference_between_images.png", diff_img)

# Save the Overlay With White Background (using OpenCV's imwrite)
cv2.imwrite(save_path + "overlay_with_white_background.png", cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
