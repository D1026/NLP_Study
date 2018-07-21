# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten
from keras.layers import Input, Dense
from keras.models import Model
import datetime
nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
print('\n', nowTime,)
import numpy as np
import pandas as pd
import xgboost as xgb

if __name__ == "__main__":
    p = pd.read_pickle('p.pkl')
    n = pd.read_pickle('n.pkl')
    # 论文 融合模型 7w正样本 7w负样本 数据已经清洗 分词 去 停用词 ；test_size=0.33333, random_state=42， 目前准确率 最高 0.9058 等你来挑战！
    # 测试 7000 正样本 7000 负样本, 随意试试 rf 0.8695 xgb 0.884 mlp 0.885
    p = p[:70000]
    n = n[:70000]
    print(p[0])
    count_vec = bigram_vectorizer = CountVectorizer(ngram_range=(1, 5), token_pattern=r'\b\w+\b', min_df=1)
    document_term_matrix = count_vec.fit_transform([" ".join(s) for s in p] + [" ".join(s) for s in n])
    vocabulary = count_vec.vocabulary_  # 得到词汇表
    tf_idf_transformer = TfidfTransformer()
    tf_idf_matrix = tf_idf_transformer.fit_transform(document_term_matrix)
    # 所要预测的真实标记
    labels = [1 for i in range(len(p))] + [0 for i in range(len(n))]
    # 训练集测试集切割
    X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, labels, test_size=0.33333, random_state=42)
    sel = SelectKBest(chi2, k=50000)
    X_train = sel.fit_transform(X_train, y_train)
    X_test = sel.transform(X_test)
    # 稀疏矩阵拼接 vstack hstack
    # 稀疏矩阵生成xgboost的dmatrix的时候,train 和 test数据集的非空行数不一致导致特征数不对
    # 如果被迫还原成完整的非空数据集，使用6W数据集5000特征时候，内存溢出 所以强制增加一行非空特征
    v = np.ones((X_train.shape[0], 1))
    X_train = hstack((X_train, v), format='csr')
    v = np.ones((X_test.shape[0], 1))
    X_test = hstack((X_test, v), format='csr')
    print(v.shape)
    print(X_train.shape)
    print(X_test.shape)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print('\n', nowTime, )
    # 随机森林 还可以尝试其他sklearn算法
    clf = RandomForestClassifier(n_estimators=300, n_jobs=4)
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    print(accuracy_score(y_test, y_pre))
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print('\n', nowTime, )

    # **********************************************************************************************************

    train_data = X_train
    test_data = X_test
    xgb_train = xgb.DMatrix(train_data, label=y_train)
    xgb_test = xgb.DMatrix(test_data, label=y_test)
    params = {
        'booster': 'gbtree',
        # 'objective': 'multi:softmax',  # 多分类的问题
        'objective': 'binary:logistic',
        'um_class': 2,  # 类别数，与 multisoftmax 并用
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 12,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        'min_child_weight': 1,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.05,  # 如同学习率
        'seed': 1000,

        'nthread': 6,  # cpu 线程数
        'eval_metric': 'error'
    }

    plst = list(params.items())
    num_rounds = 10000  # 迭代次数model
    watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
    # 训练模型
    model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print('xgb\n', nowTime, )

    # **********************************************************************************************************
    label_num = 2
    vocab_dim = 10
    input_length = 45
    n_epoch = 10
    labels = [1 for i in range(len(p))] + [0 for i in range(len(n))]
    X_train, X_test, y_train, y_test = train_test_split(p + n, labels, test_size=0.33333, random_state=42)
    count_vec = CountVectorizer(stop_words=None, min_df=0, token_pattern=r'\S+')
    document_term_matrix = count_vec.fit_transform([" ".join(s) for s in p + n])
    vocabulary = count_vec.vocabulary_  # 得到词汇表
    X_tr = []
    X_te = []
    out = []
    for i in X_train:
        c = []
        for t in i:
            if t in vocabulary:
                c.append(vocabulary[t])
            else:
                out.append(t)
        X_tr.append(c)
    for i in X_test:
        c = []
        for t in i:
            if t in vocabulary:
                c.append(vocabulary[t])
            else:
                out.append(t)
        X_te.append(c)
    X_train = X_tr
    X_test = X_te
    X_train = sequence.pad_sequences(X_train, 45)
    X_test = sequence.pad_sequences(X_test, 45)
    inputs_a = Input(shape=(input_length,))
    n_symbols = len(vocabulary)
    x_a = Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=False, input_length=input_length)(inputs_a)
    x_a = Flatten()(x_a)
    x_a = Dense(10)(x_a)
    predictions_a = Dense(1, activation='sigmoid')(x_a)
    model = Model(inputs=inputs_a,
                  outputs=predictions_a)
    print(u"训练...")
    batch_size = 128

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=5,
              validation_data=(X_test, y_test), verbose=1)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print('end\n', nowTime, )
