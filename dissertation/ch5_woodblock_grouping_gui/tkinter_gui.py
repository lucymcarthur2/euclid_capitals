import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageTk
import torch
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import zipfile
import io
import numpy as np
import tensorflow as tf
import torchvision
import cv2
import threading
import queue
import sys

# ScrollableFrame Class with grid layout support
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, columns=1, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.columns = columns

        canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def add_widget(self, widget, row=None, column=None, **grid_options):
        if row is None or column is None:
            children = self.scrollable_frame.winfo_children()
            row = len(children) // self.columns
            column = len(children) % self.columns
        widget.grid(row=row, column=column, padx=10, pady=10, **grid_options)

    def clear(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

class DetectionModel:
    def __init__(self, model_path, logger=None):
        self.logger = logger or (lambda msg: None)
        self.initial_log_messages = []
        self.initial_log_messages.append("Initializing DetectionModel")
        self.model = self._get_model()
        self._load_model(model_path)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])

    def _get_model(self):
        self.initial_log_messages.append("Getting detection model")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2  # 1 class (your object) + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _load_model(self, model_path):
        self.initial_log_messages.append(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def predict(self, image, threshold=0.5):
        self.logger("Predicting image")
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(image_tensor)
        boxes = prediction[0]['boxes'][prediction[0]['scores'] > threshold].cpu().numpy()
        scores = prediction[0]['scores'][prediction[0]['scores'] > threshold].cpu().numpy()
        return boxes, scores

# Classification Model Class
class ClassificationModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def classify(self, image):
        img_array = tf.keras.preprocessing.image.img_to_array(image.resize((128, 128)))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image
        predictions = self.model.predict(img_array)
        top_3_indices = predictions[0].argsort()[-3:][::-1]
        top_3_labels = [(chr(65 + idx), predictions[0][idx]) for idx in top_3_indices]
        return top_3_labels

# Redirect print to the log text widget
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, message)
        self.text_widget.configure(state='disabled')
        self.text_widget.see(tk.END)

    def flush(self):
        pass

# Simple GUI with Tkinter
class Application(tk.Tk):
    def __init__(self, detection_model_path, classification_model_path):
        super().__init__()
        self.detection_model = DetectionModel(detection_model_path, self.log)
        self.classification_model = ClassificationModel(classification_model_path)
        self.title("Image Detector")
        self.geometry("1200x900")

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(expand=1, fill="both")

        self.extraction_frame = ttk.Frame(self.tabs)
        self.classification_frame = ttk.Frame(self.tabs)
        self.grouping_frame = ttk.Frame(self.tabs)  # Add the Grouping tab
        self.log_frame = ttk.Frame(self.tabs)  # Add the Log tab

        self.tabs.add(self.extraction_frame, text="Extraction")
        self.tabs.add(self.classification_frame, text="Classification")
        self.tabs.add(self.grouping_frame, text="Grouping")  # Add the Grouping tab
        self.tabs.add(self.log_frame, text="Log")  # Add the Log tab

        self.init_extraction_tab()
        self.init_classification_tab()
        self.init_grouping_tab()  # Initialize the Grouping tab
        self.init_log_tab()  # Initialize the Log tab

        self.images = []  # List to hold uploaded images and their file paths
        self.cropped_images = []  # List to hold cropped bounding box images
        self.extracted_images = []  # List to hold extracted images for classification and grouping
        self.image_references = []  # Keep references to image objects to prevent garbage collection

        self.grouping_images = []  # Initialize the list to hold images for grouping
        self.use_extracted_for_grouping = False  # Flag to track if extracted images should be used for grouping

        # Redirect print statements to the log tab
        self.log_queue = queue.Queue()
        sys.stdout = StdoutRedirector(self.log_text)

        # Start a thread to periodically process log messages
        self.after(100, self.process_log_queue)

        # Process initial log messages from the detection model
        for msg in self.detection_model.initial_log_messages:
            self.log(msg)

    def process_log_queue(self):
        while not self.log_queue.empty():
            message = self.log_queue.get_nowait()
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, message + '\n')
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)
        self.after(100, self.process_log_queue)

    def log(self, message):
        self.log_queue.put(message)

    def init_extraction_tab(self):
        self.upload_button = tk.Button(self.extraction_frame, text="Upload Images", command=self.upload_images)
        self.upload_button.pack(pady=10)
    
        self.process_button = tk.Button(self.extraction_frame, text="Process Images", command=self.start_processing_images)
        self.process_button.pack(pady=10)
    
        self.extract_button = tk.Button(self.extraction_frame, text="Extract", command=self.start_extracting_bounding_boxes)
        self.extract_button.pack(pady=10)
    
        self.download_button = tk.Button(self.extraction_frame, text="Download", command=self.download_cropped_images)
        self.download_button.pack(pady=10)
    
        self.threshold_label = tk.Label(self.extraction_frame, text="Probability Threshold:")
        self.threshold_label.pack(pady=5)
    
        self.threshold_entry = tk.Entry(self.extraction_frame)
        self.threshold_entry.insert(0, "0.5")  # Default value
        self.threshold_entry.pack(pady=5)
    
        self.scrollable_frame = ScrollableFrame(self.extraction_frame, columns=5)  # Set the number of columns to 5
        self.scrollable_frame.pack(fill=tk.BOTH, expand=True)
    
        self.progress_bar = ttk.Progressbar(self.extraction_frame, mode='indeterminate')
        self.progress_bar.pack(pady=10)


    def init_classification_tab(self):
        self.upload_button = tk.Button(self.classification_frame, text="Upload Images", command=self.upload_classification_images)
        self.upload_button.pack(pady=10)

        self.use_extracted_button = tk.Button(self.classification_frame, text="Use Extracted Images", command=self.use_extracted_images)
        self.use_extracted_button.pack(pady=10)

        self.classify_button = tk.Button(self.classification_frame, text="Classify Images", command=self.start_classifying_images)
        self.classify_button.pack(pady=10)

        self.classification_scrollable_frame = ScrollableFrame(self.classification_frame, columns=3)
        self.classification_scrollable_frame.pack(fill=tk.BOTH, expand=True)

        self.progress_bar_classification = ttk.Progressbar(self.classification_frame, mode='indeterminate')
        self.progress_bar_classification.pack(pady=10)

    def init_grouping_tab(self):
        self.upload_button_grouping = tk.Button(self.grouping_frame, text="Upload Images", command=self.upload_grouping_images)
        self.upload_button_grouping.pack(pady=10)

        self.use_extracted_button_grouping = tk.Button(self.grouping_frame, text="Use Extracted Images", command=self.use_extracted_images_for_grouping)
        self.use_extracted_button_grouping.pack(pady=10)

        # Input fields for grouping parameters
        self.min_matches_label = tk.Label(self.grouping_frame, text="Minimum Matches:")
        self.min_matches_label.pack(pady=5)
        self.min_matches_entry = tk.Entry(self.grouping_frame)
        self.min_matches_entry.insert(0, "20")
        self.min_matches_entry.pack(pady=5)

        self.gradient_threshold_label = tk.Label(self.grouping_frame, text="Gradient Threshold:")
        self.gradient_threshold_label.pack(pady=5)
        self.gradient_threshold_entry = tk.Entry(self.grouping_frame)
        self.gradient_threshold_entry.insert(0, "0.3")
        self.gradient_threshold_entry.pack(pady=5)

        self.similarity_percentage_label = tk.Label(self.grouping_frame, text="Similarity Percentage Threshold:")
        self.similarity_percentage_label.pack(pady=5)
        self.similarity_percentage_entry = tk.Entry(self.grouping_frame)
        self.similarity_percentage_entry.insert(0, "85")
        self.similarity_percentage_entry.pack(pady=5)

        self.logging_var = tk.BooleanVar()
        self.logging_checkbox = tk.Checkbutton(self.grouping_frame, text="Enable Logging", variable=self.logging_var)
        self.logging_checkbox.pack(pady=5)

        self.group_button = tk.Button(self.grouping_frame, text="Group", command=self.start_grouping)
        self.group_button.pack(pady=10)

        self.grouping_scrollable_frame = ScrollableFrame(self.grouping_frame, columns=7)
        self.grouping_scrollable_frame.pack(fill=tk.BOTH, expand=True)

        self.progress_bar_grouping = ttk.Progressbar(self.grouping_frame, mode='indeterminate')
        self.progress_bar_grouping.pack(pady=10)

    def init_log_tab(self):
        self.log_text = tk.Text(self.log_frame, state='disabled', wrap='word')
        self.log_text.pack(expand=True, fill='both')

    def upload_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_paths:
            self.images = [(Image.open(file_path).convert("RGB"), file_path) for file_path in file_paths]
            self.use_extracted_for_grouping = False  # Set the flag to indicate these are newly uploaded images
            messagebox.showinfo("Success", f"Uploaded {len(file_paths)} images.")
            self.log(f"Uploaded images: {file_paths}")

    def start_processing_images(self):
        threading.Thread(target=self.process_images).start()

    def process_images(self):
        if not self.images:
            messagebox.showwarning("No Images", "Please upload images before processing.")
            return

        try:
            threshold = float(self.threshold_entry.get())
        except ValueError:
            messagebox.showwarning("Invalid Threshold", "Please enter a valid number for the probability threshold.")
            return

        self.progress_bar.start()
        self.upload_button.config(state='disabled')
        self.process_button.config(state='disabled')
        self.extract_button.config(state='disabled')
        self.download_button.config(state='disabled')

        self.scrollable_frame.clear()  # Clear previous images

        for idx, (image, image_path) in enumerate(self.images):
            self.log(f"Processing image {idx + 1} out of {len(self.images)}: {os.path.basename(image_path)}")
            boxes, scores = self.detection_model.predict(image, threshold)
            self.display_image_with_boxes(image.copy(), boxes, scores)  # Display image with boxes, use a copy to avoid altering original
            self.update_idletasks()  # Update the GUI to show the processed image immediately

        self.progress_bar.stop()
        self.upload_button.config(state='normal')
        self.process_button.config(state='normal')
        self.extract_button.config(state='normal')
        self.download_button.config(state='normal')

        messagebox.showinfo("Processing Completed", "All images have been processed.")
        self.log("Processing completed for all images.")

    def start_extracting_bounding_boxes(self):
        threading.Thread(target=self.extract_bounding_boxes).start()

    def extract_bounding_boxes(self):
        if not self.images:
            messagebox.showwarning("No Images", "Please upload and process images before extracting.")
            return

        try:
            threshold = float(self.threshold_entry.get())
        except ValueError:
            messagebox.showwarning("Invalid Threshold", "Please enter a valid number for the probability threshold.")
            return

        self.progress_bar.start()
        self.upload_button.config(state='disabled')
        self.process_button.config(state='disabled')
        self.extract_button.config(state='disabled')
        self.download_button.config(state='disabled')

        self.scrollable_frame.clear()  # Clear previous images
        self.cropped_images = []  # Clear previously cropped images
        self.extracted_images = []  # Clear previously extracted images

        for idx, (image, image_path) in enumerate(self.images):
            self.log(f"Extracting bounding boxes from image {idx + 1} out of {len(self.images)}: {os.path.basename(image_path)}")
            boxes, _ = self.detection_model.predict(image, threshold)
            for i, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = map(int, box)
                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                cropped_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i+1}.jpg"
                self.cropped_images.append((cropped_image, cropped_image_name))
                self.extracted_images.append((cropped_image, cropped_image_name))
                self.display_cropped_image(cropped_image)
                self.update_idletasks()  # Update the GUI to show the cropped image immediately

        self.progress_bar.stop()
        self.upload_button.config(state='normal')
        self.process_button.config(state='normal')
        self.extract_button.config(state='normal')
        self.download_button.config(state='normal')

        messagebox.showinfo("Extraction Completed", "All bounding boxes have been extracted.")
        self.log("Bounding box extraction completed for all images.")

    def display_image_with_boxes(self, image, boxes, scores):
        draw = ImageDraw.Draw(image)
        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red", width=5)  # Set the bounding box width to 5 pixels
            draw.text((xmin, ymin - 10), f"{score:.2f}", fill="yellow")  # Display the probability above the box

        # Resize the image to make it smaller for display while maintaining aspect ratio
        base_width = 350
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image = image.resize((base_width, h_size), Image.LANCZOS)

        img = ImageTk.PhotoImage(image)
        img_label = tk.Label(self.scrollable_frame.scrollable_frame, image=img)
        img_label.image = img  # Keep a reference to avoid garbage collection
        self.image_references.append(img)  # Keep reference to avoid garbage collection

        self.scrollable_frame.add_widget(img_label)  # Add the image to the scrollable grid

    def display_cropped_image(self, image):
    # Set the base width for resizing the image
        base_width = 100  # You can adjust this value to make the image smaller or larger
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        
        # Resize the image to the specified width while maintaining aspect ratio
        image_resized = image.resize((base_width, h_size), Image.LANCZOS)
    
        img = ImageTk.PhotoImage(image_resized)
        img_label = tk.Label(self.scrollable_frame.scrollable_frame, image=img)
        img_label.image = img  # Keep a reference to avoid garbage collection
        self.image_references.append(img)  # Keep reference to avoid garbage collection
    
        self.scrollable_frame.add_widget(img_label)  # Add the cropped image to the scrollable grid
 # Add the cropped image to the scrollable grid

    def download_cropped_images(self):
        if not self.cropped_images:
            messagebox.showwarning("No Images", "No cropped images to download. Please extract images first.")
            return

        with zipfile.ZipFile("extracted_images.zip", "w") as zipf:
            for cropped_image, image_name in self.cropped_images:
                img_byte_arr = io.BytesIO()
                cropped_image.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                zipf.writestr(image_name, img_byte_arr.read())

        save_path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile="extracted_images.zip", filetypes=[("Zip files", "*.zip")])
        if save_path:
            with open("extracted_images.zip", "rb") as src_file:
                with open(save_path, "wb") as dest_file:
                    dest_file.write(src_file.read())

        os.remove("extracted_images.zip")  # Clean up the temporary zip file
        messagebox.showinfo("Download Completed", "The extracted images have been downloaded.")
        self.log("Downloaded extracted images as a zip file.")

    def upload_classification_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_paths:
            self.extracted_images = [(Image.open(file_path).convert("RGB"), file_path) for file_path in file_paths]
            self.use_extracted_for_grouping = False  # Set the flag to indicate these are newly uploaded images
            messagebox.showinfo("Success", f"Uploaded {len(file_paths)} images.")
            self.log(f"Uploaded images for classification: {file_paths}")
            self.display_classification_images()

    def use_extracted_images(self):
        if not self.extracted_images:
            messagebox.showwarning("No Extracted Images", "Please extract images first.")
            return
        self.use_extracted_for_grouping = True  # Set the flag to use extracted images for grouping
        self.log("Using extracted images for classification.")
        self.display_classification_images()

    def display_classification_images(self):
        self.classification_scrollable_frame.clear()
        for image, image_path in self.extracted_images:
            self.display_image_for_classification(image, image_path)

    def display_image_for_classification(self, image, image_name):
        base_width = 200
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image = image.resize((base_width, h_size), Image.LANCZOS)

        img = ImageTk.PhotoImage(image)
        img_label = tk.Label(self.classification_scrollable_frame.scrollable_frame, image=img)
        img_label.image = img  # Keep a reference to avoid garbage collection
        self.image_references.append(img)  # Keep reference to avoid garbage collection

        self.classification_scrollable_frame.add_widget(img_label)  # Add the image to the scrollable grid

    def start_classifying_images(self):
        threading.Thread(target=self.classify_images).start()

    def classify_images(self):
        if not self.extracted_images:
            messagebox.showwarning("No Extracted Images", "Please extract or upload images first.")
            return

        self.progress_bar_classification.start()
        self.upload_button.config(state='disabled')
        self.use_extracted_button.config(state='disabled')
        self.classify_button.config(state='disabled')

        self.classification_scrollable_frame.clear()

        for idx, (image, image_path) in enumerate(self.extracted_images):
            self.log(f"Classifying image {idx + 1} out of {len(self.extracted_images)}: {os.path.basename(image_path)}")
            top_3_labels = self.classification_model.classify(image)
            self.display_classified_image(image, top_3_labels)

        self.progress_bar_classification.stop()
        self.upload_button.config(state='normal')
        self.use_extracted_button.config(state='normal')
        self.classify_button.config(state='normal')

    def display_classified_image(self, image, top_3_labels):
        base_width = 200
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image_resized = image.resize((base_width, h_size), Image.LANCZOS)

        img = ImageTk.PhotoImage(image_resized)
        img_label = tk.Label(self.classification_scrollable_frame.scrollable_frame, image=img)
        img_label.image = img  # Keep a reference to avoid garbage collection
        self.image_references.append(img)  # Keep reference to avoid garbage collection

        label_text = "\n".join([f"{label}: {prob:.2f}" for label, prob in top_3_labels])
        prediction_label = tk.Label(self.classification_scrollable_frame.scrollable_frame, text=label_text, justify=tk.LEFT)

        self.classification_scrollable_frame.add_widget(img_label)
        self.classification_scrollable_frame.add_widget(prediction_label)

    def upload_grouping_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_paths:
            self.grouping_images = [(Image.open(file_path).convert("RGB"), file_path) for file_path in file_paths]
            self.use_extracted_for_grouping = False  # Set the flag to indicate these are newly uploaded images
            messagebox.showinfo("Success", f"Uploaded {len(file_paths)} images.")
            self.log(f"Uploaded images for grouping: {file_paths}")
            self.display_grouping_images()  # Display the uploaded images in the grouping tab



    def use_extracted_images_for_grouping(self):
        if not self.extracted_images:
            messagebox.showwarning("No Extracted Images", "Please extract images first.")
            return
        self.use_extracted_for_grouping = True  # Set the flag to use extracted images for grouping
        self.log("Using extracted images for grouping.")
        self.display_grouping_images()

    def display_grouping_images(self):
        self.grouping_scrollable_frame.clear()
        for image, image_path in self.extracted_images:
            self.display_image_for_grouping(image, image_path)

    def display_image_for_grouping(self, image, image_name):
        base_width = 200
        w_percent = (base_width / float(image.size[0]))
        h_size = int((float(image.size[1]) * float(w_percent)))
        image = image.resize((base_width, h_size), Image.LANCZOS)

        img = ImageTk.PhotoImage(image)
        img_label = tk.Label(self.grouping_scrollable_frame.scrollable_frame, image=img)
        img_label.image = img  # Keep a reference to avoid garbage collection
        self.image_references.append(img)  # Keep reference to avoid garbage collection

        self.grouping_scrollable_frame.add_widget(img_label)  # Add the image to the scrollable grid

    def start_grouping(self):
        # Start a new thread for grouping to avoid freezing the GUI
        threading.Thread(target=self.group_images).start()

    def group_images(self):
        if not self.extracted_images and not self.grouping_images:
            messagebox.showwarning("No Images", "Please extract or upload images first.")
            return
    
        self.grouping_scrollable_frame.clear()  # Clear previous images immediately
    
        self.progress_bar_grouping.start()
        self.upload_button_grouping.config(state='disabled')
        self.use_extracted_button_grouping.config(state='disabled')
        self.group_button.config(state='disabled')
    
        # Retrieve the parameters for grouping
        try:
            min_matches = int(self.min_matches_entry.get())
            gradient_threshold = float(self.gradient_threshold_entry.get())
            similarity_percentage_threshold = float(self.similarity_percentage_entry.get())
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please ensure all grouping parameters are correctly set.")
            self.progress_bar_grouping.stop()
            self.upload_button_grouping.config(state='normal')
            self.use_extracted_button_grouping.config(state='normal')
            self.group_button.config(state='normal')
            return
    
        # Get the logging state from the checkbox
        logging = self.logging_var.get()
    
        # Determine the images based on whether extracted images should be used
        if self.use_extracted_for_grouping:
            images_to_group = self.cropped_images  # Use the cropped images with paths
            self.log("Grouping using extracted images.")
        else:
            images_to_group = self.grouping_images  # Use newly uploaded images with paths
            self.log("Grouping using newly uploaded images for grouping.")
    
        self.log(f"Images used for grouping: {[img_path for _, img_path in images_to_group]}")
    
        orb = cv2.ORB_create(nfeatures=1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
        groups, image_orientations = self.classify_and_group_images(images_to_group, orb, bf, min_matches, gradient_threshold, similarity_percentage_threshold, logging)
    
        self.progress_bar_grouping.stop()
        self.upload_button_grouping.config(state='normal')
        self.use_extracted_button_grouping.config(state='normal')
        self.group_button.config(state='normal')
    
        # Display the groups
        self.display_image_groups(groups, image_orientations)







    def flip_image(self, image):
        """Flip the image."""
        return cv2.rotate(image, cv2.ROTATE_180)

    def preprocess_image(self, image_path):
        """Preprocess the image by applying Gaussian blur."""
        img = cv2.imread(image_path, 0)  # Read in grayscale mode
        return cv2.GaussianBlur(img, (5, 5), 0)

    def resize_images_to_same_dimensions(self, img1, img2):
        """Resize both images to match the dimensions of the larger image while maintaining aspect ratios."""
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]
        target_height = max(height1, height2)
        target_width = max(width1, width2)
        aspect_ratio1 = width1 / height1
        aspect_ratio2 = width2 / height2
        if height1 != target_height or width1 != target_width:
            new_width1 = int(target_height * aspect_ratio1)
            new_height1 = int(new_width1 / aspect_ratio1)
            img1 = cv2.resize(img1, (new_width1, new_height1))
        if height2 != target_height or width2 != target_width:
            new_width2 = int(target_height * aspect_ratio2)
            new_height2 = int(new_width2 / aspect_ratio2)
            img2 = cv2.resize(img2, (new_width2, new_height2))
        return img1, img2

    def calculate_gradient(self, centroid1, centroid2, img1_width):
        """Calculate the gradient of the line connecting centroid1 and centroid2."""
        adjusted_centroid2_x = centroid2[0] + img1_width
        delta_x = adjusted_centroid2_x - centroid1[0]
        delta_y = centroid2[1] - centroid1[1]
        return np.inf if delta_x == 0 else delta_y / delta_x

    def check_similarity(self, kp1, kp2, good_matches, img1_width, gradient_threshold, similarity_percentage_threshold):
        """Check the similarity between keypoints based on gradient and return similarity percentage."""
        gradients = [self.calculate_gradient(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt, img1_width) for m in good_matches]
        gradients_no_inf = [g for g in gradients if g != np.inf and g != -np.inf]
        count_lines_with_gradient_less_than = sum(abs(g) < gradient_threshold for g in gradients_no_inf)
        percentage = (count_lines_with_gradient_less_than / len(gradients_no_inf)) * 100 if gradients_no_inf else 0
        return percentage >= similarity_percentage_threshold, percentage

    def compare_images(self, img1, img2, orb, bf, min_matches, gradient_threshold, similarity_percentage_threshold, log=False):
        """Compare two images and return if they are similar based on keypoints and descriptors."""
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.73 * n.distance]
        if len(good_matches) < min_matches:
            if log:
                self.log("Not enough matches found.")
            return False, [], 0

        is_similar, similarity_percentage = self.check_similarity(kp1, kp2, good_matches, img1.shape[1], gradient_threshold, similarity_percentage_threshold)
        if log:
            self.log(f"Similarity: {similarity_percentage}%, Similar: {'Yes' if is_similar else 'No'}")
        return is_similar, good_matches, similarity_percentage

    def classify_and_group_images(self, images_to_group, orb, bf, min_matches, gradient_threshold, similarity_percentage_threshold, logging):
        """Classify and group images based on similarity, including consideration for flipping images."""
        groups = []
        image_orientations = {}  # Use paths as keys instead
    
        for img, img_path in images_to_group:
            img_gray = np.array(img.convert('L'))
            img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)  # Preprocess the image
            best_group = None
            highest_similarity = 0
    
            if logging:
                self.log(f"Analyzing Image: {os.path.basename(img_path)}")
    
            for group in groups:
                for rep_img, rep_img_path in group[:5]:
                    rep_img_gray = np.array(rep_img.convert('L'))
                    rep_img_gray = cv2.GaussianBlur(rep_img_gray, (5, 5), 0)  # Preprocess the representative image
    
                    # Check if the representative image should be flipped
                    if image_orientations.get(rep_img_path, False):
                        rep_img_gray = self.flip_image(rep_img_gray)
    
                    img_resized, rep_img_resized = self.resize_images_to_same_dimensions(img_gray, rep_img_gray)
    
                    # Compare in original orientation
                    if logging:
                        self.log(f"Comparing {os.path.basename(img_path)} with {os.path.basename(rep_img_path)}")
                    is_similar, _, similarity_percentage = self.compare_images(img_resized, rep_img_resized, orb, bf, min_matches, gradient_threshold, similarity_percentage_threshold, logging)
                    if is_similar and similarity_percentage > highest_similarity:
                        best_group = group
                        highest_similarity = similarity_percentage
                        image_orientations[img_path] = False
    
                    # Compare in flipped orientation
                    is_similar_flipped, _, similarity_percentage_flipped = self.compare_images(self.flip_image(img_resized), rep_img_resized, orb, bf, min_matches, gradient_threshold, similarity_percentage_threshold, logging)
                    if is_similar_flipped and similarity_percentage_flipped > highest_similarity:
                        best_group = group
                        highest_similarity = similarity_percentage_flipped
                        image_orientations[img_path] = True
    
            # Log the decision
            if best_group is not None:
                if logging:
                    flip_status = "Flipped" if image_orientations.get(img_path, False) else "Not Flipped"
                    self.log(f"Added to group {groups.index(best_group) + 1} ({flip_status})")
                best_group.append((img, img_path))
            else:
                if logging:
                    self.log("Forms a new group")
                groups.append([(img, img_path)])
    
        return groups, image_orientations



    def display_image_groups(self, groups, image_orientations):
        """Display grouped images within the Tkinter GUI."""
        self.grouping_scrollable_frame.clear()  # Clear previous groups

        num_groups = len(groups)
        largest_group_size = max(len(group) for group in groups)
        mean_group_size = sum(len(group) for group in groups) / num_groups

        summary_label = tk.Label(self.grouping_scrollable_frame.scrollable_frame, text=f"Number of groups: {num_groups}\nLargest group size: {largest_group_size}\nMean group size: {mean_group_size:.2f}", font=("Helvetica", 14))
        self.grouping_scrollable_frame.add_widget(summary_label, row=0, column=0, columnspan=self.grouping_scrollable_frame.columns)

        for i, group in enumerate(groups):
            group_row = i * 3 + 1  # Each group takes 3 rows (title, images, labels) + 1 for the summary
            group_label = tk.Label(self.grouping_scrollable_frame.scrollable_frame, text=f"Group {i+1}:", font=("Helvetica", 16, "bold"))
            self.grouping_scrollable_frame.add_widget(group_label, row=group_row, column=0, columnspan=self.grouping_scrollable_frame.columns)

            for j, (image, img_path) in enumerate(group):
                img = np.array(image.convert('RGB'))
                if image_orientations.get(img_path, False):
                    img = self.flip_image(img)
                img = Image.fromarray(img)

                # Resize the image for display
                base_width = 200
                w_percent = (base_width / float(img.size[0]))
                h_size = int((float(img.size[1]) * float(w_percent)))
                img = img.resize((base_width, h_size), Image.LANCZOS)

                img_tk = ImageTk.PhotoImage(img)
                img_label = tk.Label(self.grouping_scrollable_frame.scrollable_frame, image=img_tk)
                img_label.image = img_tk  # Keep a reference to avoid garbage collection
                self.image_references.append(img_tk)  # Keep reference to avoid garbage collection

                simplified_title = os.path.basename(img_path)
                flip_status = "Flipped" if image_orientations.get(img_path, False) else "Not Flipped"
                img_title = tk.Label(self.grouping_scrollable_frame.scrollable_frame, text=f"{simplified_title} ({flip_status})")

                self.grouping_scrollable_frame.add_widget(img_label, row=group_row + 1, column=j)
                self.grouping_scrollable_frame.add_widget(img_title, row=group_row + 2, column=j)

        messagebox.showinfo("Grouping Completed", "Grouping of images has been completed successfully.")

# Model paths
detection_model_path = 'negatives.pth'
classification_model_path = r"data\models\best_1000_images.h5"

# Run the application
app = Application(detection_model_path, classification_model_path)
app.mainloop()
