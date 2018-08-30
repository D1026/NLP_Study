import json
import jieba
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
for i in sts:
    wl = jieba.lcut(i)
    seg_sts.append(' '.join(wl))
print(seg_sts[0])
print('所有sentences数量：' + str(len(seg_sts)))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(seg_sts)
x = tokenizer.texts_to_sequences(seg_sts)

print(x[0])
print(len(x))

x = pad_sequences(x, maxlen=10, truncating='pre')
print(x[0])
predict_x = x[20001:]
x = x[:20001]
# --------- x 序列化完成 ------
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, random_state=77)
#
print('x_train: '+ str(len(x_train)))
print('y_train: '+ str(len(y_train)))
print('z_train: '+ str(len(z_train)))

import pickle
with open('xxyyzz.pkl', 'wb') as f:
    pickle.dump((x_train, x_test, y_train, y_test, z_train, z_test), f)

# ------------ LSTM ------------
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