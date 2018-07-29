import json
import jieba
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

with open('intent.train_data.2w', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
samples = []
for i in lines:
    samples.append(json.loads(i.split('\t')[1]))

# 拆解 句子、意图、槽位三项
sts = []
ints = []
slos = []
for sp in samples:
    int = []
    slo = []
    sts.append(sp['sentence'])
    if len(sp['intents']) > 0:
        for it in sp['intents']:
            int.append(it['action']['value'] + '*' + it['target']['value'])
    if len(sp['slots']) > 0:
        for sl in sp['slots']:
            slo.append(sl['key'] + '*' + sl['value'])
    ints.append(int)
    slos.append(slo)
# test
print('sts数量： '+str(len(sts)))
print('ints长度： '+str(len(ints)))
print('slos长度： '+str(len(slos)))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(ints)
z = mlb.fit_transform(slos)
# test
print(mlb.classes_)
print(z[3277])
# -------  label: y,z 处理完毕 -------
seg_sts = []
for i in sts:
    wl = jieba.lcut(i)
    seg_sts.append(' '.join(wl))
print(seg_sts[0])

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(seg_sts)
x = tokenizer.texts_to_sequences(seg_sts)

print(x[0])
print(len(x))

x = pad_sequences(x, maxlen=10, truncating='pre')
print(x[0])
# --------- x 序列化完成 ------
