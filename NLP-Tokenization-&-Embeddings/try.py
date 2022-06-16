# Import all the libraries needed
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizer_v2.adam import Adam

# dataset
dataset = pd.read_csv('D:/ProjectsRepository/EvaluationResultsSentimentAnalysis/newsph-nli/train.csv')
# validation = pd.read_csv('newsph-nli/valid.csv')

sentences = dataset['s1'].tolist()
labels = dataset['label'].tolist()

# Print some example sentences and labels
for x in range(2):
    print(sentences[x])
    print(labels[x])
    print("\n")


# Separate out the sentences and labels into training and test sets
training_size = int(len(sentences) * 0.7)

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Make labels into numpy arrays for use with the network later
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# Tokenize the dataset (with tweaks!)
# vocab_size = 1000
# embedding_dim = 16

vocab_size = 2500
embedding_dim = 64
max_length = 300
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Check that the tokenizer works appropriately
num = 5
print(sentences[num])
encoded = tokenizer.encode(sentences[num])
print(encoded)

# Separately print out each subword, decoded
for i in encoded:
    print(tokenizer.decode([i]))

for i, sentence in enumerate(sentences):
    sentences[i] = tokenizer.encode(sentence)

# Check the sentences are appropriately replaced
print(sentences[5])

tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,
                                                       return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
lr = 3e-4
num_epochs = 10
opt = Adam(learning_rate=lr, decay=lr / num_epochs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_focal_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(training_padded, training_labels_final, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels_final))

loss, accuracy = model.evaluate(testing_padded, testing_labels_final)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
