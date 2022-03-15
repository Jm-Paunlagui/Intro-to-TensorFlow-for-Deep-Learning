import tensorflow as tf

# First, we will train a model to forecast the next step given the previous 20 steps, therefore, we need to create a
# dataset of 20-step windows for training.

dataset = tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy())

def window_dataset(series, window_size, batch_size=32,
                   shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


for x, y in dataset:
    print("x =", x.numpy())
    print("y =", y.numpy())
