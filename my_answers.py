import numpy as np
import string
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import unittest
import numpy.testing as npt



# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    n = 0;
    while n + window_size < len(series):
        X.append(series[n:(n+window_size)])
        y.append(series[n+window_size])
        n += 1
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))
    return model


###  return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?',' ']
    #define set of accepted characters
    valid_chars_set = ''.join(punctuation) + string.ascii_lowercase 
    #convert to lower case and then filter by acceptence criterion
    #text = ''.join(list(filter(lambda x: x in valid_chars_set,map(lambda y:y.lower(),list(text)))))
    text = ''.join(list(filter(lambda x: x in valid_chars_set,map(lambda y:y.lower(),text))))
    return text

### fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    n = 0;
    #split on input-output pairs, ignore tail giving incomplete window input
    while n + window_size < len(text):
        inputs.append(text[n:(n+window_size)])
        outputs.append(text[n+window_size])
        n += step_size
    return inputs,outputs

# build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
    
class AnswerTests(unittest.TestCase):
    def test_window_transform_series(self):
        series=[1.0, 2.0, 3.0, 4.0,5.0]
        input, output = window_transform_series(series,2)
        npt.assert_array_equal(input,np.asarray([[1.0,2.0],[2.0,3.0],[3.0,4.0]]))
        npt.assert_array_equal(output,np.asarray([[3.0],[4.0],[5.0]]))
    def test_window_transform_text(self):
        text = 'to be or not '
        input, output = window_transform_text(text,3,2)
        self.assertEqual(input,['to ',' be','e o','or ',' no'])
        self.assertEqual(output,['b',' ','r','n','t'])
        text = 'ABCDEFG'
        input, output = window_transform_text(text,5,2)
        self.assertEqual(input,['ABCDE'])
        self.assertEqual(output,['F'])
        text = 'ABCDEFGH'
        input, output = window_transform_text(text,5,2)
        self.assertEqual(input,['ABCDE','CDEFG'])
        self.assertEqual(output,['F','H'])

    def test_cleaned_text(self):
        text = 'aaa&&!,.:;? bbb\n\r&~^$@#*()'
        self.assertEqual(cleaned_text(text),'aaa!,.:;? bbb')

if __name__ == '__main__':
    unittest.main()