# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.layers import GlobalAveragePooling1D
from keras.models import model_from_json

np.random.seed(42)
label_num = 38
vocab_dim = 300
input_length = 500
n_epoch = 50
# [n_symbols, X_train, y_train, X_test, y_test] = pd.read_pickle('temp.pkl')
(X_train, X_test, y_train, y_test) = pd.read_pickle('yw_xxyy.pkl')
print(X_train[0])
print(y_train[0])


# val_dense_1_acc: 0.8992007288537471, 0.9007221248538815, 0.9012578473280258] 512 *0.8 20维
def train_cnn(n_symbols, X_train, y_train, X_test, y_test):
    inputs_a = Input(shape=(input_length,))
    inputs_b = Input(shape=(input_length,))
    x_a = Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=False, input_length=input_length)(inputs_a)
    x_b = Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True, input_length=input_length)(inputs_b)
    x_b = Bidirectional(GRU(500))(x_b)
    x_a = GlobalAveragePooling1D()(x_a)
    x_a = Dropout(0.1)(x_a)
    x_b = Dropout(0.1)(x_b)
    predictions_a = Dense(38, activation='softmax')(x_a)
    predictions_b = Dense(38, activation='softmax')(x_b)
    x_all = Concatenate(axis=-1)([x_a, x_b])
    predictions = Dense(38, activation='softmax')(x_all)
    model = Model(inputs=[inputs_a, inputs_b],
                  outputs=[predictions_a, predictions_b, predictions])
    print(u"训练...")
    batch_size = 64
    # load json and create model
    # loaded_model_json = pd.read_pickle('model/MLP/' + str(20) + '.json')
    # model = model_from_json(loaded_model_json)
    # model.load_weights('model/MLP/' + str(20) + '.weight')

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'], loss_weights=[0.2, 1.0, 0.1])
    model.fit([X_train, X_train], [y_train, y_train, y_train], batch_size=batch_size, epochs=200,
              validation_data=([X_test, X_test], [y_test, y_test, y_test]), verbose=1)

    # a = model.evaluate([X_test, X_test], [y_test, y_test, y_test], verbose=0, batch_size=5000)
    # print(a)
    # model.save_weights('model/MLP/' + str(i) + '.weight')
    # mj = model.to_json()
    # import pandas as pd
    # pd.to_pickle(mj, 'model/MLP/' + str(i) + '.json')


k = X_train.shape[0]
n_symbols = X_train.shape[0] + 1
train_cnn(n_symbols, X_train[:k], y_train[:k], X_test, y_test)
