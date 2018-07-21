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
# print(call_list[1])
# for i in call_list[1].split('\t'):
#     print(i)

x_train = []
y_train = []
# 遍历每个来电数据，提取
for ele in call_list:
    if ele == '':
        continue
    sents = ele.split('\n')
    y_str = sents[0].split('\t')[1:]    # 两个元素或一个 一级分类 二级分类
    y_str = ''.join(y_str)
    # 找出空白标签
    if y_str == '':
        print(sents[0])

    x_str = []  # 多条对话
    for i in sents[1:]:
        x_str.append(i.split('\t')[1])
    x_train.append(x_str)
    y_train.append(y_str)
# -----
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(y_train)
#
print(len(encoder.classes_))
# convert integers to dummy variables (one hot encoding)
y = np_utils.to_categorical(encoded_Y)
# y = encoded_Y

# 分词
X_train = []
w_str = ''
jieba.suggest_freq('兆', tune=True)
jieba.suggest_freq('块', tune=True)
jieba.suggest_freq('流量', tune=True)
#
stop_list = {}.fromkeys([line.strip() for line in open('stopwords.txt', encoding='UTF-8')])
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

        # word_str = ' '.join(x_split)
        # l_str = ' '.join(line) + '\n'
        # w_str = w_str + l_str
    # w_str = w_str + '$'
    x_s = [word for word in x_split if word not in stop_list]
    X_train.append(' '.join(x_s))

# ---------分词去停用词完成：X_train, y ---------------

# 词序列模型 x_data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(nb_words=30000)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)

print(sequences[0])
print(len(sequences))

x_data = pad_sequences(sequences, maxlen=600, truncating='pre')
#
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.33333, random_state=77)
#
print(np.array(x_train).shape, 'x_train维度')
print(np.array(y_train).shape, 'y_train维度')
print(np.array(x_test).shape, 'x_test维度')
print(np.array(y_test).shape, 'y_test维度')
# import pickle
# with open('xxyy.pkl', 'wb') as f:
#     pickle.dump((x_train, x_test, y_train, y_test), f)

# ------------- 词袋模型 ----------
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.feature_selection import SelectKBest, chi2
#
# count_vec = CountVectorizer(ngram_range=(1, 5), token_pattern=r'\b\w+\b', min_df=1)
# document_term_matrix = count_vec.fit_transform(X_train)
# vocabulary = count_vec.vocabulary_  # 得到词汇表
# tf_idf_transformer = TfidfTransformer()
# tf_idf_matrix = tf_idf_transformer.fit_transform(document_term_matrix)
#
# x_train, x_test, y_train, y_test = train_test_split(tf_idf_matrix, y, test_size=0.33333, random_state=77)
#
# sel = SelectKBest(chi2, k=50000)
# x_train = sel.fit_transform(x_train, y_train)
# x_test = sel.transform(x_test)

# LSTM 目前最高准确率 0.4642
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

print('Build model...')
model = Sequential()
model.add(Embedding(30000, 128))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(37, activation='softmax'))

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
# model.save_weights('weights')

print('Test score:', score)
print('Test accuracy:', acc)

# ---------------- xgb -------------------
# import xgboost as xgb
#
# train_data = x_train
# test_data = x_test
# xgb_train = xgb.DMatrix(train_data, label=y_train)
# xgb_test = xgb.DMatrix(test_data, label=y_test)
# params = {
#     'booster': 'gbtree',
#     'objective': 'multi:softprob',  # 多分类的问题
#     # 'objective': 'binary:logistic',
#     # 'num_class': 38,  # 类别数，与 multisoftmax 并用
#     'num_class': 38,
#     'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
#     'max_depth': 16,  # 构建树的深度，越大越容易过拟合
#     'lambda': 3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#     'subsample': 0.7,  # 随机采样训练样本
#     'colsample_bytree': 0.7,  # 生成树时进行的列采样
#     'min_child_weight': 1,
#     # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#     # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#     # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
#     'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
#     'eta': 0.05,  # 如同学习率
#     'seed': 1000,
#
#     'nthread': 6,  # cpu 线程数
#     'eval_metric': 'merror'
#     }
#
# plst = list(params.items())
# num_rounds = 10000  # 迭代次数model
# watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
# # 训练模型
# model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)