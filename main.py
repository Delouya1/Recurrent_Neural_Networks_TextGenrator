import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, LSTM

filepath = tf.keras.utils.get_file('shakespeare.txt', origin='https://storage.googleapis.com/download.tensorflow.org'
                                                             '/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()  # better performance with lower case

text = text[300000:800000]  # cut out the first 300k characters and the last 200k characters to reduce training time

characters = sorted(set(text))  # a set of all the unique characters in the text

char_to_index = dict((c, i) for i, c in enumerate(characters))  # a dictionary mapping unique characters to indices
index_to_char = dict((i, c) for i, c in enumerate(characters))  # a dictionary mapping indices to unique characters

SEQ_LENGTH = 40  # the length of our input sequences
STEP = 3  # how many characters we will move forward with our input sequence with each iteration

# sentences = []
# next_chars = []
#
# for i in range(0, len(text) - SEQ_LENGTH, STEP):  # create our input sequences and output sequences
#     sentences.append(text[i: i + SEQ_LENGTH])
#     next_chars.append(text[i + SEQ_LENGTH])
#
# x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)  # our input data
# y = np.zeros((len(sentences), len(characters)), dtype=bool)  # our target data
#
# for i, sentence in enumerate(sentences):  # one-hot encode our input data and output data
#     for t, char in enumerate(sentence):
#         x[i, t, char_to_index[char]] = 1
#     y[i, char_to_index[next_chars[i]]] = 1

# model = Sequential()
# model.add(LSTM(128, input_shape=(
#     SEQ_LENGTH, len(characters))))  # the first layer in our network needs to know the input shape
# model.add(Dense(len(characters)))  # a dense layer to output a prediction for each unique character
# # as many neurons as characters
# model.add(Activation('softmax'))  # a softmax activation function to turn the outputs into probability-like values
# # and pick the character with the highest probability as our model’s prediction
#
#
# model.compile(loss='categorical_crossentropy', optimizer='adam')  # compile the model
#
# model.fit(x, y, batch_size=256, epochs=4)  # train the model
#
# model.save('textgenerator.model')  # save the model
model = keras.models.load_model('textgenerator.model')  # load the model we saved earlier


def sample(preds, temperature=1.0):  # a function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')  # convert the predictions to a numpy array
    preds = np.log(preds) / temperature  # apply temperature
    exp_preds = np.exp(preds)  # exponentiate the predictions
    preds = exp_preds / np.sum(exp_preds)  # get the probabilities for each prediction
    probas = np.random.multinomial(1, preds, 1)  # sample the index
    return np.argmax(probas)  # return the index of the predicted character


def generate_text(length, temperature):  # a function to generate text
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)  # pick a random seed
    generated = ''  # our output string
    sentence = text[start_index: start_index + SEQ_LENGTH]  # seed with part of the text
    generated += sentence  # add the seed to our output

    for i in range(length):  # iterate over the number of characters we want to generate
        x_pred = np.zeros((1, SEQ_LENGTH, len(characters)))  # initialize the input array as all zeros

        for t, char in enumerate(sentence):  # one-hot encode our new input characters
            x_pred[0, t, char_to_index[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]  # make a prediction
        next_index = sample(preds, temperature)  # sample the output
        next_char = index_to_char[next_index]  # convert the output index to a character

        generated += next_char  # add the character to our generated text
        sentence = sentence[1:] + next_char  # move our input over by one character

    return generated

print('----- diversity: 0.2 -----')
print(generate_text(300, 0.2))  # generate 300 characters with a low temperature
print('----- diversity: 0.5 -----')
print(generate_text(300, 0.5))  # generate 300 characters with a medium temperature
print('----- diversity: 1.0 -----')
print(generate_text(300, 1.0))  # generate 300 characters with a high temperature
