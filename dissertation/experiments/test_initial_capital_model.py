# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 08:42:21 2023

@author: Lucy McArthur
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 18:20:33 2023

@author: Lucy McArthur
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define the image directory
test_image_dir = r"data\initial_capitals\specific\test"

# Define target size for resizing
target_size = (72, 87)  # height first, then width

# Load the model
model = load_model('my_model.h5')

# Define labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'I', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']

# Loop through the directory and make predictions
for img_file in os.listdir(test_image_dir):
    img_file_path = os.path.join(test_image_dir, img_file)
    if os.path.isfile(img_file_path):
        img = cv2.imread(img_file_path, 1)  # read the image in color mode
        img = cv2.resize(img, target_size[::-1])  # cv2.resize takes width first, then height
        img = img.astype('float32') / 255
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class = labels[np.argmax(prediction)]

        print(f"The image {img_file} is predicted to be of class: {predicted_class}")
