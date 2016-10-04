import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from theano.tensor.shared_randomstreams import RandomStreams
import h5py
import string
import csv


# fix random seed for reproducibility
numpy.random.seed(7)
srng = RandomStreams(7)
seq_length = 150

#reading in the agent_log file and converting it to a list
with open('training.txt', 'rb') as f:
    reader = csv.reader(f)
    agent_data = list(reader)

#setting up an ascii dictionary
dd = list(string.printable)

# Convert for a chracter to an in in the dictionary
char_to_int = dict((c, i) for i, c in enumerate(dd))

#the reverse integer back to character
int_to_char = dict((i, c) for i, c in enumerate(dd))

##all used to determine the padding##

#The maximum length of a word
max_word_length = 10

#The maximum number of words in the sentence
max_words = 15

#setting up the Target value array
target_array = []

#Setting up the Data Array
data_array = []

#Setting up the word padding to ensure that all sentences are 15 words
full_pad = [999,999,999,999,999,999,999,999,999,999,]


for data in agent_data:
    int_array = []
    data_string = data[0].strip().split(" ")
    t = []
    t.append(int(data[2]))
    target_array.append(t)
    for s in data_string:
        temp_array = []
        c_list = list(s) # ['I', 'i', 'K', 'D', 'd', 'g', 'b', '2']
        for c in c_list:
            temp_array.append(char_to_int[c])
        padding = max_word_length - len(temp_array)

        i = 1
        while (i <= padding):
            temp_array.append(999)
            i = i+1
        int_array.append(temp_array)
    word_padding = max_words - len(int_array)
    ii = 1
    while (ii <= word_padding):
        int_array.append(full_pad)
        ii = ii+1
    data_array.append(int_array)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(data_array, (len(data_array), 150, 1))

# normalize
X = X / float(len(dd))

# one hot encode the output variable
y = np_utils.to_categorical(target_array)

# create and fit the model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), dropout_W=0.2, dropout_U=0.2))

model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoint-{epoch:02d}-{acc:.2f}.hdf5',monitor='acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

#writing weights
#model.fit(X, y, nb_epoch=100, batch_size=64, verbose=2, callbacks=[checkpoint])


model.fit(X, y, nb_epoch=500, batch_size=1, verbose=2)
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))


