from __future__ import print_function
import pickle
import numpy as np
from keras.preprocessing import sequence
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def create_ngram_set(input_list, ngram_value=2):
    # """
    # Extract a set of n-grams from a list of integers.
    #
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    # {(4, 9), (4, 1), (1, 4), (9, 4)}
    #
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    # [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    # """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    # >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 2
max_features = 5000
maxlen = 500
batch_size = 128
embedding_dims = 300
epochs = 100

print('Loading data...')

with open("./dialog_1_seg.pkl", "rb") as f:
    dialog_1 = pickle.load(f)
f.close()

with open("./dialog_2_seg.pkl", "rb") as f:
    dialog_2 = pickle.load(f)
f.close()

with open("./label_index_onehot.pkl", "rb") as f:
    label = pickle.load(f)
f.close()

tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(dialog_1)
sequences1 = tokenizer1.texts_to_sequences(dialog_1)

tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(dialog_2)
sequences2 = tokenizer1.texts_to_sequences(dialog_2)
# print(sequences)
# exit()
x_train1, x_test1, y_train1, y_test1 = train_test_split(sequences1, label, test_size=0.3333, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(sequences2, label, test_size=0.3333, random_state=42)
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train1), 'train1 sequences')
print(len(x_test1), 'test sequences')

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train1:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train1 = add_ngram(x_train1, token_indice, ngram_range)
    x_test1 = add_ngram(x_test1, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train1)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test1)), dtype=int)))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train2:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train1 = add_ngram(x_train2, token_indice, ngram_range)
    x_test1 = add_ngram(x_test2, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train2)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test2)), dtype=int)))

print('Pad sequences (samples x time)')
x_train1 = pad_sequences(x_train1, maxlen=maxlen)
x_test1 = pad_sequences(x_test1, maxlen=maxlen)

x_train2 = pad_sequences(x_train2, maxlen=maxlen)
x_test2 = pad_sequences(x_test2, maxlen=maxlen)
print('x_train1 shape:', x_train1.shape)
print('x_test1 shape:', x_test1.shape)
print('x_train2 shape:', x_train2.shape)
print('x_test2 shape:', x_test2.shape)

#fasttext
print('Build model...')

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions

input1 = Input(shape=(maxlen, ))
embedding_1 = Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen)(input1)
dropout1_1 = Dropout(0.1)(embedding_1)
avg_pool1 = GlobalAveragePooling1D()(dropout1_1)

input2 = Input(shape=(maxlen, ))
embedding_2 = Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen)(input2)
dropout2_1 = Dropout(0.1)(embedding_2)
avg_pool2 = GlobalAveragePooling1D()(dropout2_1)
x = concatenate([avg_pool1, avg_pool2])
dropout3 = Dropout(0.1)(x)
output = Dense(37, activation="softmax")(dropout3)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()

early_stop = EarlyStopping(monitor="val_acc", patience=5, verbose=0, mode="auto")
#adam 9064, rmsprop 0.9089
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

model.fit([x_train1, x_train2], y_train1,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([x_test1, x_test2], y_test1), callbacks=[early_stop])
# plot_model(model, to_file='./tmp/log/model.png')


