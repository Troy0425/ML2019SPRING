from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
import pandas as pd
import numpy as np
import sys
import jieba
from gensim.models import word2vec
import matplotlib.pyplot as plt

jieba.load_userdict(sys.argv[4])
def process(inputdata):
	outputdata = []
	len_data = len(inputdata)
	for i in range(len_data):
		line = []
		words = jieba.cut(inputdata[i], cut_all=False)
		for word in words:
			line.append(word)
		outputdata.append((line))
	return (outputdata)
train_x = pd.read_csv(sys.argv[1], header=0)
train_y = pd.read_csv(sys.argv[2], header=0)
test_x = pd.read_csv(sys.argv[3], header=0)
train_x = np.array(train_x["comment"])
train_y = np.array(train_y["label"])
test_x = np.array(test_x["comment"])
train_x = process(train_x)
test_x = process(test_x)

dimension = 150
length = 80
#word2vecmodel = word2vec.Word2Vec(list(train_x) + list(test_x), size=dimension, iter=10, min_count=10)
#word2vecmodel.save("word2vec.model")
word2vecmodel = word2vec.Word2Vec.load("word2vec.model")
def transform(data_x):
	data_x_vec = np.zeros((len(data_x), length,dimension))
	for i in range(len(data_x)):
		count = 0
		for j in range(len(data_x[i])):
			word = data_x[i][j]
			if word in word2vecmodel.wv.vocab:
				vec = word2vecmodel.wv[word]
				data_x_vec[i][count] = vec
				count += 1
			if(count >= length):
				break
	return data_x_vec
train_x_vec = transform(train_x)
test_x_vec = transform(test_x)

model = Sequential()
model.add(Dense(units = 200, activation = 'relu', input_shape = (length, dimension)))
model.add(Dropout(0.3))
model.add(Dense(units = 200, activation='relu'))
model.add(Dropout(0.3))
#model.add(LSTM(units = 50, return_sequences = False, input_shape = (length, dimension)))
model.add(GRU(units = 100))
#model.add(Dense(units = 300, activation='relu'))
model.add(Dense(units = 1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(train_x_vec, train_y, epochs = 30, batch_size = 1024, shuffle=True)
model.save(sys.argv[5])

