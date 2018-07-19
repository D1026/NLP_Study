# from keras.models import Sequential
# from keras.layers import Dense, Activation
# import keras
#
# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=100))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # 生成虚拟数据
# import numpy as np
# data = np.random.random((1000, 100))
# labels = np.random.randint(10, size=(1000, 1))
#
# # 将标签转换为分类的 one-hot 编码
# one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
#
# # 训练模型，以 32 个样本为一个 batch 进行迭代
# model.fit(data, one_hot_labels, epochs=10, batch_size=32)
import jieba
import word2vec

# 45来电原因
y_class = ['投诉（含抱怨）网络问题', '投诉（含抱怨）营销问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', \
           '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', ]

with open('callreason.train.fj_and_sh.2w', 'r', encoding='UTF-8') as train_txt:
    content = train_txt.read()
call_list = content.split('\n\n')
print(call_list[1])
for i in call_list[1].split('\t'):
    print(i)

x_train = []
y_train = []
# 遍历每个来电数据，提取
for ele in call_list:
    if ele == '':
        continue
    sents = ele.split('\n')
    y_str = sents[0].split('\t')[1:]    # 两个元素或一个 一级分类 二级分类
    x_str = []  # 多条对话
    for i in sents[1:]:
        x_str.append(i.split('\t')[1])
    x_train.append(x_str)
    y_train.append(y_str)

# 分词
X_train = []
w_str = ''
jieba.suggest_freq('兆', tune=True)
jieba.suggest_freq('块', tune=True)
jieba.suggest_freq('流量', tune=True)
for x in x_train:
    x_split = []
    for s in x:
        line = jieba.lcut(s)
        # 删除信息量几乎为0的词
        for i in line:
            if i in ('你好', '您好', '请讲', '请说'):
                line.remove(i)
        if len(line) > 0:
            x_split.extend(line)

        word_str = ' '.join(x_split)
        l_str = ' '.join(line) + '\n'
        w_str = w_str + l_str
    # w_str = w_str + '$'
    X_train.append(' '.join(x_split))

# with open('seg', 'w', encoding='UTF-8') as fw:
#     fw.write(w_str)

# word2vec.word2vec('seg', 'vec.bin', size=10, verbose=True)
# model = word2vec.load('vec.bin')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(nb_words=20000)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)

print(sequences[0])
print(len(sequences))

x_data = pad_sequences(sequences, maxlen=500, truncating='pre')

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

print('Build model...')
model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(LSTM(32, activation='relu', dropout=0.3, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))