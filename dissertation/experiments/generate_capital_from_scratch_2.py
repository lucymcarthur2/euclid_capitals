
import cv2
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont
import scipy
def remove_background(image, threshold=128):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

from PIL import ImageOps

def random_rotation(image, max_angle=5):
    """Applies a random rotation to the image."""
    angle = random.uniform(-max_angle, max_angle)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    return rotated

def random_skew(image, max_skew=0.2):
    """Applies a random skew to the image."""
    rows, cols, _ = image.shape
    x1 = cols*random.uniform(0, max_skew)
    y1 = rows*random.uniform(0, max_skew)
    x2 = cols - cols*random.uniform(0, max_skew)
    y2 = rows*random.uniform(0, max_skew)
    x3 = cols*random.uniform(0, max_skew)
    y3 = rows - rows*random.uniform(0, max_skew)
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
    pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
    M = cv2.getAffineTransform(pts1, pts2)
    skewed = cv2.warpAffine(image, M, (cols, rows))
    return skewed


def draw_leaf(img, base_point, end_point):
    """Draw a leaf using Bezier curves."""
    direction = np.array(end_point) - np.array(base_point)
    normal_direction = np.array([-direction[1], direction[0]])
    normalized_normal = normal_direction / np.linalg.norm(normal_direction)
    
    control_point1 = (base_point + normalized_normal * random.uniform(5, 15)).astype(int)
    control_point2 = (end_point + normalized_normal * random.uniform(5, 15)).astype(int)
    
    t_values = np.linspace(0, 1, 100)
    curve1_points = bezier_curve([base_point, control_point1, end_point], t_values)
    curve2_points = bezier_curve([base_point, control_point2, end_point], t_values)
    
    curve_points = np.vstack((curve1_points, curve2_points[::-1]))
    curve_points = curve_points.astype(int)
    cv2.polylines(img, [curve_points], isClosed=True, color=255, thickness=2)



def random_horizontal_squash(image):
    """Randomly compresses the left or right half of the image horizontally."""
    rows, cols, _ = image.shape
    
    # Randomly determine whether to apply the squash
    apply_squash = random.choice([True, False])
    
    # If not applying the squash, simply return the original image
    if not apply_squash:
        return image
    
    # Randomly determine the degree of squash (e.g., between 70% to 100% of the half width)
    compression_factor = random.uniform(0.7, 1.0)
    compressed_width = int(cols // 2 * compression_factor)
    
    # Randomly choose the side to squash
    squash_left = random.choice([True, False])
    
    if squash_left:
        # Squash the left half
        left_half = image[:, :cols//2]
        compressed_left_half = cv2.resize(left_half, (compressed_width, rows))
        squashed_image = np.hstack((compressed_left_half, image[:, cols//2:]))
    else:
        # Squash the right half
        right_half = image[:, cols//2:]
        compressed_right_half = cv2.resize(right_half, (compressed_width, rows))
        squashed_image = np.hstack((image[:, :cols//2], compressed_right_half))
    
    return squashed_image



def add_noise_to_text(image, noise_level):
    row, col, ch = image.shape
    mean = 0
    sigma = noise_level * 255
    gauss = np.random.normal(mean, sigma, (row, col)).astype(np.uint8)
    gauss = np.repeat(gauss[:, :, np.newaxis], 3, axis=2)
    noisy = cv2.add(image, gauss)
    return noisy

FONT_PATHS = [
    "times.ttf",
    "calibri.ttf",
    "cambria.ttc",
    "georgia.ttf"
    # Add more font paths as needed
]

def draw_text_with_border(image, text, font_size=380, border_thickness=10):
    # Choose a random font from the list
    font_path = random.choice(FONT_PATHS)
    font = ImageFont.truetype(font_path, font_size)
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Calculate the text bounding box
    draw = ImageDraw.Draw(pil_image)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2], text_bbox[3]
    
    # Calculate the position to draw the text in the center of the image and adjust upwards by 10 pixels
    center_x = (pil_image.width - text_width) // 2
    center_y = (pil_image.height - text_height) // 2 - 20
    
    # Draw the text with its black border onto the main image
    draw.text((center_x, center_y), text, font=font, fill=(255, 255, 255), stroke_width=border_thickness, stroke_fill=(0, 0, 0))
    
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)




def add_plant_to_canvas(canvas, plant_image, opacity_factor, max_attempts=1000):
    """Adds a plant image to a canvas without overlapping other plants."""
    rows, cols = canvas.shape
    plant_rows, plant_cols = plant_image.shape

    # Crop the plant if it's too large for the canvas
    if plant_rows > rows:
        crop_start_row = (plant_rows - rows) // 2
        crop_end_row = crop_start_row + rows
        plant_image = plant_image[crop_start_row:crop_end_row, :]
        plant_rows = rows

    if plant_cols > cols:
        crop_start_col = (plant_cols - cols) // 2
        crop_end_col = crop_start_col + cols
        plant_image = plant_image[:, crop_start_col:crop_end_col]
        plant_cols = cols

    # List to keep track of placed plants' bounding boxes
    placed_plants = []
    
    for attempt in range(max_attempts):
        x_offset = random.randint(0, cols - plant_cols)
        y_offset = random.randint(0, rows - plant_rows)
        
        # Define the bounding box for the new plant
        new_bbox = (x_offset, y_offset, x_offset + plant_cols, y_offset + plant_rows)
        
        # Check for overlap with already placed plants
        overlap = False
        for placed_bbox in placed_plants:
            if (new_bbox[2] > placed_bbox[0] and new_bbox[0] < placed_bbox[2]) and \
               (new_bbox[3] > placed_bbox[1] and new_bbox[1] < placed_bbox[3]):
                overlap = True
                break
        
        # If no overlap, place the plant and update the bounding boxes list
        if not overlap:
            roi = canvas[y_offset:y_offset+plant_rows, x_offset:x_offset+plant_cols]
            mask = plant_image > 0
            roi_flat = roi[mask].reshape(-1, 1)
            plant_flat = plant_image[mask].reshape(-1, 1)
            combined_values = cv2.addWeighted(roi_flat, 1 - opacity_factor, plant_flat, opacity_factor, 0)
            roi[mask] = combined_values.ravel()
            placed_plants.append(new_bbox)
            break
    
    return canvas


def remove_background(image, threshold=128):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def bezier_curve(points, t_values):
    curve_points = []
    num_points = len(points)
    
    for t in t_values:
        bezier_point = np.array([0.0, 0.0])
        for i in range(num_points):
            coefficient = scipy.special.comb(num_points-1, i) * (1 - t) ** (num_points-1 - i) * t ** i
            
            # Debug lines to print shapes and types
            #print(f"coefficient type: {type(coefficient)}, shape: {np.shape(coefficient)}")
            #print(f"points[i] type: {type(points[i])}, shape: {np.shape(points[i])}")
            #print(f"np.array(points[i]) type: {type(np.array(points[i]))}, shape: {np.shape(np.array(points[i]))}")
            
            bezier_point += coefficient * np.array(points[i])
        curve_points.append(bezier_point)
        
    return np.array(curve_points)



def generate_branch_background(width, height):
    canvas = np.zeros((height, width), dtype=np.uint8)
    occupied_grid = np.zeros((height, width), dtype=np.uint8)

    # Randomly choose a quadrant or half
    choice = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right', 'left-half', 'right-half', 'top-half', 'bottom-half'])

    if choice in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
        sub_width, sub_height = width // 2, height // 2
    else:
        sub_width, sub_height = width, height // 2

    sub_canvas = np.zeros((sub_height, sub_width), dtype=np.uint8)

    def line_overlaps_existing(line_points):
        """Check if the line overlaps with existing content."""
        for point in line_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < occupied_grid.shape[1] and 0 <= y < occupied_grid.shape[0]:
                if occupied_grid[y, x] == 1:
                    return True
            else:
                return True  # Consider out-of-bound points as overlaps to avoid the error
        return False
    # Generate curly branches
    for _ in range(random.randint(2, 5)):
        while True:
            num_control_points = random.randint(6, 8)
            control_points = [np.array([random.randint(0, sub_width), random.randint(0, sub_height)]) for _ in range(num_control_points)]
            t_values = np.linspace(0, 1, 1000)
            curve_points = bezier_curve(control_points, t_values)
            if not line_overlaps_existing(curve_points):
                break
        int_curve_points = [tuple(map(int, point)) for point in curve_points]
        thickness = random.randint(6, 12)
        cv2.polylines(sub_canvas, [np.array(int_curve_points)], isClosed=False, color=255, thickness=thickness)
        for point in int_curve_points:
            occupied_grid[point[1], point[0]] = 1

    # Generate straighter branches
    for _ in range(random.randint(1, 3)):
        while True:
            num_control_points = random.randint(3, 4)
            control_points = [np.array([random.randint(0, sub_width), random.randint(0, sub_height)]) for _ in range(num_control_points)]
            t_values = np.linspace(0, 1, 1000)
            curve_points = bezier_curve(control_points, t_values)
            if not line_overlaps_existing(curve_points):
                break
        int_curve_points = [tuple(map(int, point)) for point in curve_points]
        cv2.polylines(sub_canvas, [np.array(int_curve_points)], isClosed=False, color=255, thickness=5)
        for point in int_curve_points:
            occupied_grid[point[1], point[0]] = 1

    # Scatter leaves in unoccupied spaces
    leaf_count = random.randint(30, 60)
    for _ in range(leaf_count):
        while True:
            x = random.randint(0, sub_width - 1)
            y = random.randint(0, sub_height - 1)
            if occupied_grid[y, x] == 0:
                break
        leaf_angle = random.uniform(0, 2 * np.pi)
        leaf_length = random.randint(5, 15)
        leaf_end = (int(x + leaf_length * np.cos(leaf_angle)),
                    int(y + leaf_length * np.sin(leaf_angle)))
        draw_leaf(sub_canvas, (x, y), leaf_end)




    # Place the sub_canvas into the main canvas and flip as necessary
    if choice == 'top-left':
        canvas[:sub_height, :sub_width] = sub_canvas
        canvas[sub_height:, :sub_width] = np.flipud(sub_canvas)
        canvas[:, sub_width:] = np.fliplr(canvas[:, :sub_width])
    elif choice == 'top-right':
        canvas[:sub_height, width-sub_width:] = sub_canvas
        canvas[sub_height:, width-sub_width:] = np.flipud(sub_canvas)
        canvas[:, :width-sub_width] = np.fliplr(canvas[:, width-sub_width:])
    elif choice == 'bottom-left':
        canvas[height-sub_height:, :sub_width] = sub_canvas
        canvas[:height-sub_height, :sub_width] = np.flipud(sub_canvas)
        canvas[:, sub_width:] = np.fliplr(canvas[:, :sub_width])
    elif choice == 'bottom-right':
        canvas[height-sub_height:, width-sub_width:] = sub_canvas
        canvas[:height-sub_height, width-sub_width:] = np.flipud(sub_canvas)
        canvas[:, :width-sub_width] = np.fliplr(canvas[:, width-sub_width:])
    elif choice == 'left-half':
        sub_canvas = cv2.resize(sub_canvas, (sub_width, height))
        canvas[:, :sub_width] = sub_canvas
        canvas[:, width-sub_width:] = np.fliplr(sub_canvas)
    elif choice == 'right-half':
        sub_canvas = cv2.resize(sub_canvas, (sub_width, height))
        canvas[:, width-sub_width:] = sub_canvas
        canvas[:, :sub_width] = np.fliplr(sub_canvas)
    elif choice == 'top-half':
        sub_canvas_resized = cv2.resize(sub_canvas, (width, height // 2))
        canvas[:height//2, :] = sub_canvas_resized
        canvas[height//2:, :] = np.flipud(sub_canvas_resized)
    elif choice == 'bottom-half':
        sub_canvas_resized = cv2.resize(sub_canvas, (width, height // 2))
        canvas[height//2:, :] = sub_canvas_resized
        canvas[:height//2, :] = np.flipud(sub_canvas_resized)

    return canvas




def generate_composition(image_path, init_width, init_height, 
                         letter_to_generate, num_letters, min_blur=1, max_blur=7):
    # Load the base image from which to crop pieces
    base_image = cv2.imread(image_path)
    base_height, base_width = base_image.shape[:2]

    compositions = []
    for _ in range(num_letters):
        # Decide randomly whether to use a cropped image or a black background
        use_black_background = random.choice([True, False])
        if use_black_background:
            # Create a black background
            crop = np.zeros((init_height, init_width, 3), dtype=np.uint8)
        else:
            # Randomly select a portion of the image to crop
            x = random.randint(0, base_width - init_width)
            y = random.randint(0, base_height - init_height)
            crop = base_image[y:y+init_height, x:x+init_width]

        # Randomly choose blur and noise levels
        blur_level = random.randint(min_blur, max_blur)
        blur_level = blur_level + 1 if blur_level % 2 == 0 else blur_level
        noise_level = random.uniform(0, 0.001)

        # Generate the text with border
        scaled_font_size = 380
        image_with_text = draw_text_with_border(crop, letter_to_generate, font_size=scaled_font_size)

        # Add noise and blur to the image with text
        image_with_noise = add_noise_to_text(image_with_text, noise_level)
        image_blurred = cv2.GaussianBlur(image_with_noise, (blur_level, blur_level), 0)

        compositions.append(image_blurred)

    return compositions

def main():
    base_dir = 'data/manual_capitals'
    img_path = 'data/initial_capitals_improved/extracted_object_32811_017.jpg_4.png'
    init_width, init_height = 308, 366
    use_plant_background = True  # Set this to False to use mosaic background

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        folder_path = os.path.join(base_dir, letter)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        compositions = generate_composition(image_path = img_path, init_width = init_width,
                                            init_height = init_height,
                                            letter_to_generate = letter, num_letters = 50,
                                            min_blur=1, max_blur=13)

        for idx, comp in enumerate(compositions):
            file_name = f"{letter}_{idx}.png"
            file_path = os.path.join(folder_path, file_name)
            cv2.imwrite(file_path, comp)

if __name__ == "__main__":
    main()