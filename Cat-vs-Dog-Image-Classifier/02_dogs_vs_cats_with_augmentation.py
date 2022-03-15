# Importing packages
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Loading the dataset
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

zip_dir_base = os.path.dirname(zip_dir)
print(zip_dir_base)

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

# Understanding our data
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# Setting the Model Parameters
BATCH_SIZE = 100  # Number of training examples to process before updating our models variables
IMG_SHAPE = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels


# Data Augmentation
# Overfitting often occurs when we have a small number of training examples. One way to fix this
# problem is to augment our dataset so that it has sufficient number and variety of training examples. Data
# augmentation takes the approach of generating more training data from existing training samples, by augmenting the
# samples through random transformations that yield believable-looking images. The goal is that at training time,
# your model will never see the exact same picture twice. This exposes the model to more aspects of the data,
# allowing it to generalize better.

# In tf.keras we can implement this using the same ImageDataGenerator class we used before. We can simply pass
# different transformations we would want to our dataset as a form of arguments, and it will take care of applying it
# to the dataset during our training process.

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# Flipping the image horizontally
# Rotating the image @ 45 degrees
# Zooming the image
image_gen_train = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                     shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE, directory=train_dir, shuffle=True,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='binary')

# To see the transformation in action, let's take one sample image from our training set and repeat it five times.
# The augmentation will be randomly applied (or not) to each repetition.
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)

# Creating Validation Data generator Generally, we only apply data augmentation to our training examples,
# since the original images should be representative of what our model needs to manage. So, in this case we are only
# rescaling our validation images and converting them into batches using ImageDataGenerator.
image_gen_train = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE, directory=validation_dir,
                                                   target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='binary')

# Model creation
# The model consists of four convolution blocks with a max pool layer in each of them. Then we have a
# fully connected layer with 512 units, with a relu activation function. The model will output class probabilities
# for two classes — dogs and cats — using softmax.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

# Compile the model As usual, we will use the adam optimizer. Since we output a softmax categorization,
# we'll use sparse_categorical_crossentropy as the loss function. We would also like to look at training and
# validation accuracy on each epoch as we train our network, so we are passing in the metrics argument.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model Summary
# Let's look at all the layers of our network using summary method.
model.summary()

# Train the model
# It's time we train our network.
# Since our batches are coming from a generator (ImageDataGenerator), we'll use fit_generator instead of fit.
EPOCHS = 100
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

# Visualizing results of the training
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
