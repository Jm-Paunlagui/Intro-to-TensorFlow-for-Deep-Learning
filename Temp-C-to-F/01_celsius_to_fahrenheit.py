# Import dependencies
import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Set up training data As we saw before, supervised Machine Learning is all about figuring out an algorithm given a
# set of inputs and outputs. Since the task in this Codelab is to create a model that can give the temperature in
# Fahrenheit when given the degrees in Celsius, we create two lists celsius_q and fahrenheit_a that we can use to
# train our model.
celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i, c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

# Create the model
# We'll call the layer l0 and create it by instantiating tf.keras.layers.Dense with the following configuration:
#
# input_shape=[1] — This specifies that the input to this layer is a single value. That is, the shape is a
# one-dimensional array with one member. Since this is the first (and only) layer, that input shape is the input
# shape of the entire model. The single value is a floating point number, representing degrees Celsius.
#
# units=1 — This specifies the number of neurons in the layer. The number of neurons defines how many internal
# variables the layer has to try to learn how to solve the problem (more later). Since this is the final layer,
# it is also the size of the model's output — a single float value representing degrees Fahrenheit. (In a
# multi-layered network, the size and shape of the layer would need to match the input_shape of the next layer.)
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Assemble layers into the model Once layers are defined, they need to be assembled into a model. The Sequential
# model definition takes a list of layers as an argument, specifying the calculation order from the input to the
# output.
# This model has just a single layer, l0.
model = tf.keras.Sequential([l0])

# Note
# You will often see the layers defined inside the model definition, rather than beforehand:
# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(units=1, input_shape=[1])
# ])

# Compile the model, with loss and optimizer functions
# Before training, the model has to be compiled. When compiled for training, the model is given:

# Loss function — A way of measuring how far off predictions are from the desired outcome. (The measured difference
# is called the "loss".)

# Optimizer function — A way of adjusting internal values in order to reduce the loss.

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.1), metrics=['mae'])

# Train the model
# Train the model by calling the fit method.

# During training, the model takes in Celsius values, performs a calculation using the current internal variables (
# called "weights") and outputs values which are meant to be the Fahrenheit equivalent. Since the weights are
# initially set randomly, the output will not be close to the correct value. The difference between the actual output
# and the desired output is calculated using the loss function, and the optimizer function directs how the weights
# should be adjusted.

# This cycle of calculate, compare, adjust is controlled by the fit method. The first argument is the inputs,
# the second argument is the desired outputs. The epochs argument specifies how many times this cycle should be run,
# and the verbose argument controls how much output the method produces.

history = model.fit(celsius_q, fahrenheit_a, epochs=1000, verbose=1)
print("Finished training the model")
print(model.predict([100.0]))
model.save('tempConverterModel.h5')

plt.xlabel('Epoch Number')
plt.ylabel("Loss/Accuracy Magnitude")
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.savefig('result.png')
