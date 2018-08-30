import json
import jieba.posseg as ps
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# -----训练数据------
with open('intent.train_data.2w', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
train_samples = []
train_ids = []
for i in lines:
    train_ids.append(i.split('\t')[0])
    train_samples.append(json.loads(i.split('\t')[1]))

# ------预测数据------
with open('intent_data.test_B.5k.raw.0827', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
testB_samples = []
testB_ids = []
for i in lines:
    testB_ids.append(i.split('\t')[0])
    testB_samples.append(json.loads(i.split('\t')[1]))
# ---训练数据----
# 拆解 句子、意图、槽位三项
train_sts = []
train_ints = []
train_slos = []
for sp in train_samples:
    int = []
    slo = []
    train_sts.append(sp['sentence'])
    if len(sp['intents']) > 0:
        for it in sp['intents']:
            int.append(it['action']['value'] + '*' + it['target']['value'])
    if len(sp['slots']) > 0:
        for sl in sp['slots']:
            slo.append(sl['key'] + '*' + sl['value'])
    train_ints.append(int)
    train_slos.append(slo)
# test
print('sts数量： '+str(len(train_sts)))
print('ints长度： '+str(len(train_ints)))
print('slos长度： '+str(len(train_slos)))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_ints)
z = mlb.fit_transform(train_slos)
# test
print(mlb.classes_)
print(z[3277])
print('y的维度：' + str(len(y[0])))     # 67
print('z的维度：' + str(len(z[0])))     # 119
# -------  label: y,z 处理完毕 -------
# ----预测数据-----
testB_sts = []
for i in testB_samples:
    testB_sts.append(i['sentence'])
print('预测数据数量：'+str(len(testB_sts)))
# ----
sts = train_sts.extend(testB_sts)
# -----
seg_sts = []
seg_fls = []
for i in sts:
    wp = ps.cut(i)
    lw = []
    lf = []
    for w, f in wp:
        lw.append(w)
        lf.append(f)
    seg_sts.append(' '.join(lw))
    seg_fls.append(' '.join(lf))
print(seg_sts[0])
print('所有sentences数量：' + str(len(seg_sts)))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer1 = Tokenizer(num_words=30000)
tokenizer1.fit_on_texts(seg_sts)
x1 = tokenizer1.texts_to_sequences(seg_sts)
tokenizer2 = Tokenizer(num_words=32)
tokenizer2.fit_on_texts(seg_fls)
x2 = tokenizer2.texts_to_sequences(seg_sts)

print(x1[0])
print(len(x1))

x1 = pad_sequences(x1, maxlen=10, truncating='pre')
print(x1[0])
predict_x1 = x1[20001:]
x1 = x1[:20001]
x2 = pad_sequences(x2, maxlen=10, truncating='pre')
predict_x2 = x2[20001:]
x2 = x2[:20001]
# --------- x 序列化完成 ------
x1_train, x1_test, x2_train, x2_test, y_train, y_test, z_train, z_test = train_test_split(x1, x2, y, z, test_size=0.2, random_state=77)
#
print('x2_train: '+ str(len(x2_train)))
print('y_train: '+ str(len(y_train)))
print('z_train: '+ str(len(z_train)))

# import pickle
# with open('xxyyzz.pkl', 'wb') as f:
#     pickle.dump((x_train, x_test, y_train, y_test, z_train, z_test), f)

# ------------ LSTM + fasttext ------------
from keras.layers import *
from keras.models import *
from keras.callbacks import *
label_num = [13, 40, 7, 74]
vocab_dim = 300
pos_dim = 30
vocabulary_size = 30000
pos_size = 32


def build_input(input_dim, output_dim, shape):
    inputs = Input(shape=shape)
    x = Embedding(output_dim=output_dim, input_dim=input_dim + 1, mask_zero=False,
                    input_length=shape[0])(inputs)

    return inputs, x


def concat_output(x_right, x_left, vocab_dimension, pos_dimension):
    x = Concatenate()([x_right, x_left])
    x = Dropout(0.5)(x)
    x = AveragePooling1D(pool_size=1)(x)
    x = Reshape((-1, vocab_dimension + pos_dimension))(x)
    print(x.shape)
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(300))(x)

    return x


inputs_w, x_w = build_input(vocabulary_size, vocab_dim, x1_train[0].shape)
inputs_f, x_f = build_input(pos_size, pos_dim, x2_train[0].shape)
x = concat_output(x_w, x_f, vocab_dim, pos_dim)

predict_y = Dense(67, activation='softmax', name="intend")(x)
predict_z = Dense(119, activation='softmax', name="slots")(x)

model = Model(inputs=[inputs_w, inputs_f],
              outputs=[predict_y, predict_z])

print("训练...")

batch_size = 64
tensorboard = TensorBoard(log_dir="./log2/2")
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'], loss_weights=[0.2, 1.0, 0.5, 0.1])
# validation_data=([X_test, P_test], [action_test, target_test, key_test, value_test]),
model.fit([x1_train, x2_train], [y_train, z_train], batch_size=batch_size, epochs=64,
           verbose=1, callbacks=[tensorboard])

predict_label = model.predict(x=[x1_test, x2_test])

predict_y = predict_label[0]
predict_z = predict_label[1]



# ---------------- lstm -----------------
import pickle
with open('xxyyzz.pkl', 'rb') as f:
    x_train, x_test, y_train, y_test, z_train, z_test = pickle.load(f)

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional

print('Build model...')
model = Sequential()
model.add(Embedding(30000, 128))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(67, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          validation_data=(x_test, y_test))