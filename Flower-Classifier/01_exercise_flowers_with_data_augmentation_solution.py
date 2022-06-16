# Import packages
import os
import numpy as np
import glob
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tsds
from keras import layers

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Data loading

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

train_roses = os.path.join(train_dir, 'roses')
train_daisy = os.path.join(train_dir, 'daisy')
train_dandelion = os.path.join(train_dir, 'dandelion')
train_sunflowers = os.path.join(train_dir, 'sunflowers')
train_tulips = os.path.join(train_dir, 'tulips')

val_roses = os.path.join(val_dir, 'roses')
val_daisy = os.path.join(val_dir, 'daisy')
val_dandelion = os.path.join(val_dir, 'dandelion')
val_sunflowers = os.path.join(val_dir, 'sunflowers')
val_tulips = os.path.join(val_dir, 'tulips')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} Images".format(cl, len(images)))
    train, val = images[:round(len(images) * 0.8)], images[round(len(images) * 0.8):]

    # for t in train:
    #     if not os.path.exists(os.path.join(base_dir, 'train', cl)):
    #         os.makedirs(os.path.join(base_dir, 'train', cl))
    #     shutil.move(t, os.path.join(base_dir, 'train', cl))
    #
    # for v in val:
    #     if not os.path.exists(os.path.join(base_dir, 'val', cl)):
    #         os.makedirs(os.path.join(base_dir, 'val', cl))
    #     shutil.move(v, os.path.join(base_dir, 'val', cl))

# Understanding our data

# num_roses_tr = len(os.listdir(train_roses))
# num_daisy_tr = len(os.listdir(train_daisy))
# num_dandelion_tr = len(os.listdir(train_dandelion))
# num_sunflowers_tr = len(os.listdir(train_sunflowers))
# num_tulips_tr = len(os.listdir(train_tulips))
#
# num_roses_val = len(os.listdir(val_roses))
# num_daisy_val = len(os.listdir(val_daisy))
# num_dandelion_val = len(os.listdir(val_dandelion))
# num_sunflowers_val = len(os.listdir(val_sunflowers))
# num_tulips_val = len(os.listdir(val_tulips))

# total_train = num_roses_tr + num_daisy_tr + num_dandelion_tr + num_sunflowers_tr + num_tulips_tr
# total_val = num_roses_val + num_daisy_val + num_dandelion_val + num_sunflowers_val + num_tulips_val

print(base_dir)

# print('total training of rose images:', num_roses_tr)
# print('total training of daisy images:', num_daisy_tr)
# print('total training of dandelion images:', num_dandelion_tr)
# print('total training of sunflowers images:', num_sunflowers_tr)
# print('total training of tulips images: ', num_tulips_tr)
#
# print('total validation of rose images:', num_roses_val)
# print('total validation of daisy images:', num_daisy_val)
# print('total validation of dandelion images:', num_dandelion_val)
# print('total validation of sunflowers images:', num_sunflowers_val)
# print('total validation of tulips images: ', num_tulips_val)
#
# print("Total training images:", total_train)
# print("Total validation images:", total_val)

# Setting model params
batch_size = 100
IMAGE_SHAPE = 150

image_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=45, zoom_range=0.15, horizontal_flip=True,
                               width_shift_range=.2, height_shift_range=.2, shear_range=0.15, fill_mode='nearest')

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True,
                                               target_size=(IMAGE_SHAPE, IMAGE_SHAPE), class_mode='sparse')

image_gen_val = ImageDataGenerator(rescale=1. / 255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size, directory=val_dir,
                                                 target_size=(IMAGE_SHAPE, IMAGE_SHAPE), class_mode='sparse')


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 80
history = model.fit(train_data_gen,
                    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size)))
                    )
# Visualizing results of the training
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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
