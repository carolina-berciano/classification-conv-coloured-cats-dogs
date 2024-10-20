# import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Download the dataset on disk
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# Download a folder into disk and unzip it. get_file returns the path to the downloaded folder: /home/carolina/.keras/datasets
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

# store full path including folder name: /home/carolina/.keras/datasets/cats_and_dogs_filtered
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
# store full path including train data name: /home/carolina/.keras/datasets/cats_and_dogs_filtered/train
train_dir = os.path.join(base_dir, 'train')
# store full path including validation data name: /home/carolina/.keras/datasets/cats_and_dogs_filtered/validation
validation_dir = os.path.join(base_dir, 'validation')

# /home/carolina/.keras/datasets/cats_and_dogs_filtered/train/cats
train_cats_dir = os.path.join(train_dir, 'cats')
# /home/carolina/.keras/datasets/cats_and_dogs_filtered/train/dogs
train_dogs_dir = os.path.join(train_dir, 'dogs')
# /home/carolina/.keras/datasets/cats_and_dogs_filtered/validation/cats
validation_cats_dir = os.path.join(validation_dir, 'cats')
# /home/carolina/.keras/datasets/cats_and_dogs_filtered/validation/dogs
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Explore the data
num_cats_tr = len(os.listdir(train_cats_dir))  # number of elements in the train cats directory
num_dogs_tr = len(os.listdir(train_dogs_dir))  # number of elements in the train dogs directory
num_cats_val = len(os.listdir(validation_cats_dir))  # number of elements in the validation cats directory
num_dogs_val = len(os.listdir(validation_dogs_dir))  # number of elements in the validation dogs directory
total_train = num_cats_tr + num_dogs_tr  # total number of train elements
total_val = num_cats_val + num_dogs_val  # total number of train elements

# Set reusable params
BATCH_SIZE = 100
IMG_SHAPE = 150

# Prepare data
# resize is change image size, say from 1234x1234 to 5x5 / rescale is change image px values, say from 248 to 0.8
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)  # apply image augmentation to avoid overfitting
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                           class_mode='binary')

validation_image_generator = ImageDataGenerator(rescale=1./255) # data (image) augmentation not applicable to validation
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                              class_mode='binary')


# Create model
# 1. Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),  # prevent overfitting by setting 50% of the values coming to this layer to 0
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# get to know the model structure
print(model.summary())

# train the model
# use model_fit instead of fit because the data comes from the ImageDataGenerator object
EPOCHS = 60
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

# when working with binary classification problems
# alternative approach:
# define model last layer -> tf.keras.layers.Dense(1, activation='sigmoid')
# use loss='binary_crossentropy' when compile

# Visualize loss and accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()
