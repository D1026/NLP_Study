import json
from jieba_fast import posseg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

with open('intent.train_data.2w', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
samples = []
for i in lines:
    samples.append(json.loads(i.split('\t')[1]))

# 拆解 句子、意图、槽位三项
sentences = []
intent_action = []
intent_target = []
slots_key = []
slots_value = []
for sp in samples:
    int_action = []
    int_target = []
    slot_key = []
    slot_value = []
    sentences.append(sp['sentence'])
    if len(sp['intents']) > 0:
        for it in sp['intents']:
            int_action.append(it['action']['value'])
            int_target.append(it['target']['value'])

    if len(sp['slots']) > 0:
        for sl in sp['slots']:
            slot_key.append(sl['key'])
            slot_value.append(sl['value'])
    intent_action.append(int_action)
    intent_target.append(int_target)
    slots_key.append(slot_key)
    slots_value.append(slot_value)


# 意图动作和意图目标
intent_action1 = []
intent_action2 = []

intent_target1 = []
intent_target2 = []


for i in range(len(intent_action)):
    if len(intent_action[i]) == 0:
        intent_action1.append("")
        # intent_action2.append("")

    elif len(intent_action[i]) == 1:
        intent_action1.append(intent_action[i][0])
        # intent_action1.append("")

    elif len(intent_action[i]) == 2:
        intent_action1.append(intent_action[i][0])
        # intent_action1.append(intent_action[i][1])
    else:
        pass

for i in range(len(intent_target)):
    if len(intent_target[i]) == 0:
        intent_target1.append("")
        # intent_target2.append("")

    elif len(intent_target[i]) == 1:
        intent_target1.append(intent_target[i][0])
        # intent_target2.append("")

    elif len(intent_target[i]) == 2:
        intent_target1.append(intent_target[i][0])
        # intent_target2.append(intent_target[i][1])
    else:
        pass

# 槽位
slots_key1 = []
slots_key2 = []

slots_value1 = []
slots_value2 = []

for i in range(len(slots_key)):
    if len(slots_key[i]) == 0:
        slots_key1.append("")
        # intent_action2.append("")
    else:
        slots_key1.append(slots_key[i][0])
    # elif len(slots_key[i]) == 1:
    #     slots_key1.append(slots_key[i][0])
    #     # intent_action1.append("")
    #
    # elif len(slots_key[i]) == 2:
    #     slots_key1.append(slots_key[i][0])
    #     # intent_action1.append(intent_action[i][1])
    # else:
    #     pass

for i in range(len(slots_value)):
    if len(slots_value[i]) == 0:
        slots_value1.append("")
        # intent_target2.append("")
    else:
        slots_value1.append(slots_value[i][0])
# elif len(slots_value[i]) == 1:
    #     slots_value1.append(slots_value[i][0])
    #     # intent_target2.append("")
    #
    # elif len(intent_target[i]) == 2:
    #     slots_value1.append(slots_value[i][0])
    #     # intent_target2.append(intent_target[i][1])
    # else:
    #     pass

# intent_actions = []
# intent_targets = []

# intent_actions = intent_action1 + intent_action2
# intent_targets = intent_target1 + intent_target2

# 意图动作和意图目标编码
intent_actions = intent_action1
intent_targets = intent_target1

for i in range(len(intent_actions)):
    if intent_actions[i] == '':
        intent_actions[i] = "10086"

for i in range(len(intent_targets)):
    if intent_targets[i] == '':
        intent_targets[i] = "10086"


encoder = LabelEncoder()
encoded_actions = encoder.fit_transform(intent_actions)
actions_categories = np_utils.to_categorical(encoded_actions)
action_class = encoder.classes_

encoder = LabelEncoder()
encoded_targets = encoder.fit_transform(intent_targets)
targets_categories = np_utils.to_categorical(encoded_targets)
target_class = encoder.classes_

# 槽位编码
slots_keys = slots_key1
slots_values = slots_value1

for i in range(len(slots_keys)):
    if slots_keys[i] == '':
        slots_keys[i] = "10010"

for i in range(len(slots_values)):
    if slots_values[i] == '':
        slots_values[i] = "10010"


encoder = LabelEncoder()
encoded_keys = encoder.fit_transform(slots_keys)
keys_categories = np_utils.to_categorical(encoded_keys)
keys_class = encoder.classes_

encoder = LabelEncoder()
encoded_values = encoder.fit_transform(slots_values)
values_categories = np_utils.to_categorical(encoded_values)
values_class = encoder.classes_

# 意图动作和意图目标真实标签和编码映射
action_dict = {}
for i in action_class:
    action_dict[i] = None

for i in range(len(intent_actions)):
    for k in action_dict.keys():
        if intent_actions[i] == k:
            action_dict[k] = actions_categories[i]

target_dict = {}
for i in target_class:
    target_dict[i] = None

for i in range(len(intent_targets)):
    for k in target_dict.keys():
        if intent_targets[i] == k:
            target_dict[k] = targets_categories[i]

# 意图动作和意图目标真实标签和编码映射
keys_dict = {}
for i in keys_class:
    keys_dict[i] = None

for i in range(len(slots_keys)):
    for k in keys_dict.keys():
        if slots_keys[i] == k:
            keys_dict[k] = keys_categories[i]

values_dict = {}
for i in values_class:
    values_dict[i] = None

for i in range(len(slots_values)):
    for k in values_dict.keys():
        if slots_values[i] == k:
            values_dict[k] = values_categories[i]
# -------  label: y,z 处理完毕 -------
train_text = []
train_positive = []
for i in sentences:
    words = []
    pos_seg = []
    line = posseg.lcut(i)
    for k in line:
        words.append(k.word)
        pos_seg.append(k.flag)
    if len(words) != len(pos_seg):
        print("error")
        break

    if len(words) > 0:
        train_text.append(words)
        train_positive.append(pos_seg)


with open('intent_data.testA_5K', 'r', encoding='UTF-8') as f:
    lines = f.readlines()

sentences_test = []
for i in lines:
    sentences_test.append(json.loads(i.split('\t')[1])["sentence"])


test_text = []
test_positive = []
for i in sentences_test:
    words = []
    pos_seg = []
    line = posseg.lcut(i)
    for k in line:
        words.append(k.word)
        pos_seg.append(k.flag)
    if len(words) != len(pos_seg):
        print("error")
        break

    if len(words) > 0:
        test_text.append(words)
        test_positive.append(pos_seg)

segment_text = train_text + test_text
segment_positive = train_positive + test_positive

tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(segment_text)
text = tokenizer.texts_to_sequences(segment_text)
text_symbols = len(tokenizer.word_index) + 1
print(text_symbols)
text = pad_sequences(text, truncating='pre')


tokenizer = Tokenizer()
tokenizer.fit_on_texts(segment_positive)
positive = tokenizer.texts_to_sequences(segment_positive)
positive_symbols = len(tokenizer.word_index) + 1
print(positive_symbols)
positive = pad_sequences(positive, truncating='pre')

text_train = text[0:len(train_text)]
positive_train = positive[0:len(train_positive)]

text_test = text[len(train_text):]
positive_test = positive[len(train_positive):]
print(len(text_train))
print(len(test_text))
# action1 = actions_categories[0:20002]
# action2 = actions_categories[20002:]
#
# target1 = targets_categories[0:20002]
# target2 = targets_categories[20002:]

# action1 = encoded_actions[0:20002]
# action2 = encoded_actions[20002:]
#
# target1 = encoded_targets[0:20002]
# target2 = encoded_targets[20002:]

import pickle
with open('xxxxyyzz.pkl', 'wb') as f:
    pickle.dump((text_train, positive_train, text_test, positive_test, actions_categories, targets_categories, keys_categories, values_categories), f)

with open("yyzz_dict.pkl", "wb") as f:
    pickle.dump((action_dict, target_dict, keys_dict, values_dict), f)
