import json
import jieba

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
            int.append(it['action']['value'] + it['target']['value'])
    if len(sp['slots']) > 0:
        for sl in sp['slots']:
            slo.append(sl['key'] + sl['value'])
    ints.append(int)
    slos.append(slo)
# test
print('sts数量： '+str(len(sts)))
print('ints长度： '+str(len(ints)))
print('slos长度： '+str(len(slos)))