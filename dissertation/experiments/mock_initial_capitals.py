import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Paths
base_path = r"data\initial_capitals\specific\train unprocessed"
augmented_path = r"data\initial_capitals\specific\train augmented"

# Define ImageDataGenerator for augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.05, height_shift_range=0.05,
                             shear_range=0.01, zoom_range=0.1, fill_mode='constant', cval=0,
                             horizontal_flip=False)

folders = os.listdir(base_path)

for folder in folders:
    images = os.listdir(os.path.join(base_path, folder))
    save_path = os.path.join(augmented_path, folder)
    os.makedirs(save_path, exist_ok=True)
    
    for image in images:
        img = load_img(os.path.join(base_path, folder, image))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix=image, save_format='jpeg'):
            i += 1
            if i > 100:  # create 20 augmented images per base image
                break
