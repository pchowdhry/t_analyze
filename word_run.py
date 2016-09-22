import numpy
from keras.utils import np_utils
from keras.models import load_model
import string



# fix random seed for reproducibility
#numpy.random.seed(7)
#srng = RandomStreams(7)
#seq_length = 150


agent_data = []
agent_data.append('5K8 36U 0Oc EJJwxg soWMpUU f8XBLSM QHRjivI2nw F-h5TWvrBA ,0Oc,2'.split(","))


#setting up an ascii dictionary
dd = list(string.printable)

# Convert for a chracter to an in in the dictionary
char_to_int = dict((c, i) for i, c in enumerate(dd))

#the reverse integer back to character
int_to_char = dict((i, c) for i, c in enumerate(dd))

#all used to determine the padding

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


model = load_model('checkpoint-93-1.00.hdf5')

ar = 0
pattern = data_array[ar]
x = numpy.reshape(pattern, (1, 150, 1))
x = x / float(len(dd))
prediction = model.predict(x, verbose=0)
index = numpy.argmax(prediction)
if int(agent_data[ar][2]) == int(index):
    print "YES"
else: print 'NOPE'


