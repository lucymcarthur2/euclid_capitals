# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:06:51 2023

@author: Lucy McArthur
"""


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# Define parameters
img_height, img_width = 72, 87
batch_size = 16

# Define train_dir
train_dir = r"data\initial_capitals\specific\train"

# Define the data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)  # validation split of 30%

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # not 'binary'
    subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_dir,  # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # not 'binary'
    subset='validation')  # set as validation data

# Create a new model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(15, activation='softmax')  # 14 classes instead of 26
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=20)

# Save the model
model.save('my_model.h5')