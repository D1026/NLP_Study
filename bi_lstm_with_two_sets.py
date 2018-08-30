from __future__ import print_function
import numpy as np
import pickle
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

def get_session(gpu_fraction=0.3):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# ktf.set_session(get_session(0.5))  # using 60% of total GPU Memory

# Set parameters:
# ngram_range = 2 will add bi-grams features
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

print('Pad sequences (samples x time)')
x_train1 = pad_sequences(x_train1, maxlen=maxlen)
x_test1 = pad_sequences(x_test1, maxlen=maxlen)

x_train2 = pad_sequences(x_train2, maxlen=maxlen)
x_test2 = pad_sequences(x_test2, maxlen=maxlen)
print('x_train shape:', x_train1.shape)
print('x_test shape:', x_test1.shape)


print('Build model...')

input1 = Input(shape=(maxlen, ))
embedding_1 = Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen)(input1)
dropout1_1 = Dropout(0.1)(embedding_1)
# x1 = Bidirectional(GRU(128, return_sequences=True))(x)
bilstm1 = Bidirectional(LSTM(150, return_sequences=True))(dropout1_1)
# conc = concatenate([x1, x2])
# avg_pool = GlobalAveragePooling1D()(conc)
# max_pool1 = GlobalMaxPooling1D()(x1)
# conc = concatenate([avg_pool, max_pool])
dense1 = Dense(100, activation='relu')(bilstm1)
dropout1_2 = Dropout(0.1)(dense1)

input2 = Input(shape=(maxlen, ))
embedding_2 = Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen)(input2)
dropout2_1 = Dropout(0.1)(embedding_2)
# x1 = Bidirectional(GRU(128, return_sequences=True))(x)
bilstm2 = Bidirectional(LSTM(150, return_sequences=True))(dropout2_1)
# conc = concatenate([x1, x2])
# avg_pool = GlobalAveragePooling1D()(conc)
# max_pool2 = GlobalMaxPooling1D()(x2)
# conc = concatenate([avg_pool, max_pool])
dense2 = Dense(100, activation='relu')(bilstm2)
dropout2_2 = Dropout(0.1)(dense2)

x = concatenate([dropout1_2, dropout2_2])
max_pool = GlobalMaxPooling1D()(x)
# dense3 = Dense(50, activation='relu')(max_pool)
dropout3 = Dropout(0.1)(max_pool)
output = Dense(37, activation="softmax")(dropout3)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit([x_train1, x_train2], y_train1,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([x_test1, x_test2], y_test1))
# plot_model(model, to_file='./tmp/log/model.png')
