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

# 45来电原因
y_class = ['投诉（含抱怨）网络问题', '投诉（含抱怨）营销问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', \
           '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', '投诉（含抱怨）费用问题', ]

with open('callreason.train.fj_and_sh.2w', 'r', encoding='UTF-8') as train_txt:
    content = train_txt.read()
call_list = content.split('\n\n')
# print(call_list[1])

x_train = []
y_train = []

for ele in call_list:
    if ele == '':
        continue
    sents = ele.split('\n')