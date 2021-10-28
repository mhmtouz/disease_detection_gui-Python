import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import tflearn
import random
import json
import string
import unicodedata
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
nltk.download('punkt')
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

def remove_punctuation(text):
    return text.translate(tbl)

stemmer = LancasterStemmer()
data = None

with open('dataS.json',encoding="utf-8-sig") as json_data:
    data = json.load(json_data)

categories = list(data.keys())
words = []
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        each_sentence = remove_punctuation(each_sentence)
        w = nltk.word_tokenize(each_sentence)
        words.extend(w)
        docs.append((w, each_category))

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

training = []
output = []
output_empty = [0] * len(categories)


for doc in docs:
    bow = []
    token_words = doc[0]
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1
    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=100, batch_size=128, show_metric=True)
model.save('model.tflearn')