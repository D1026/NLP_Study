import jieba
import word2vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# 45来电原因
y_class = ['投诉（含抱怨）网络问题', '投诉（含抱怨）营销问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）服务问题', '投诉（含抱怨）业务使用问题', \
           '投诉（含抱怨）业务办理问题', '投诉（含抱怨）业务规定不满', '投诉（含抱怨）不知情定制问题', '投诉（含抱怨）信息安全问题', '投诉（含抱怨）电商售后问题', \
           '办理开通', '办理取消', '办理变更', '办理下载/设置', '办理转户', '办理打印/邮寄', '办理重置/修改/补发', \
           '办理缴费', '办理移机/装机/拆机', '办理停复机', '办理补换卡', '办理入网', '办理销户/重开', \
           '咨询（含查询）产品/业务功能', '咨询（含查询）账户信息', '咨询（含查询）业务资费', '咨询（含查询）业务订购信息查询', '咨询（含查询）使用方式', '咨询（含查询）办理方式', '咨询（含查询）业务规定', \
           '咨询（含查询）号码状态', '咨询（含查询）用户资料', '咨询（含查询）服务渠道信息', '咨询（含查询）工单处理结果', '咨询（含查询）电商货品信息', '咨询（含查询）营销活动信息', '咨询（含查询）宽带覆盖范围', \
           '表扬及建议表扬', '表扬及建议建议', '特殊来电无声电话', '特殊来电骚扰电话', '转归属地10086', '非移动业务', 'c', '非来电']

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
    y_str = ''.join(y_str)
    x_str = []  # 多条对话
    for i in sents[1:]:
        x_str.append(i.split('\t')[1])
    x_train.append(x_str)
    y_train.append(y_str)
# -----
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(y_train)
# convert integers to dummy variables (one hot encoding)
y = np_utils.to_categorical(encoded_Y)

#手工生成多分类标签
# y = [[0 for i in range(45)] for j in range(len(x_train))]
# for i in range(len(y_train)):
#     for j in range(len(y_class)):
#         #
#         print('i = '+str(i)+'   j= '+str(j)+'   yti = '+y_train[i] + '  ycj= ' + y_class[j])
#         if y_train[i] == y_class[j]:
#             print('匹配')
#             y[i][j] = 1
#             break
#     print(y[i])
# #
# y = np.array(y)
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
#
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.33333, random_state=77)
#
import pickle
with open('xy.pkl', 'wb') as f:
    pickle.dump((x_data, y), f)

# LSTM 目前最高准确率 0.4528
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

print('Build model...')
model = Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(38, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
# model.load_weights('weights', by_name=False)
model.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=128)
# 保存权重
model.save_weights('weights')

print('Test score:', score)
print('Test accuracy:', acc)