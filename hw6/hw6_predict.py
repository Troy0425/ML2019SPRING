from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
import numpy as np
import sys
import jieba
from gensim.models import word2vec

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
"""train_x = pd.read_csv(sys.argv[1], header=0)
train_y = pd.read_csv(sys.argv[2], header=0)"""
test_x = pd.read_csv(sys.argv[3], header=0)
test_x = np.array(test_x["comment"])
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
test_x_vec = transform(test_x)
predict = []
model1 = load_model('LSTM_model_1')
model2 = load_model('LSTM_model_2')
model3 = load_model('LSTM_model_3')
model4 = load_model('CNN_LSTM_model_1')
model5 = load_model('CNN_LSTM_model_2')
model6 = load_model('CNN_LSTM_model_3')
model7 = load_model('GRU_model_1')
model8 = load_model('GRU_model_2')
model9 = load_model('GRU_model_3')
predict1 = model1.predict(test_x_vec)
predict2 = model2.predict(test_x_vec)
predict3 = model3.predict(test_x_vec)
predict4 = model4.predict(test_x_vec)
predict5 = model5.predict(test_x_vec)
predict6 = model6.predict(test_x_vec)
predict7 = model7.predict(test_x_vec)
predict8 = model8.predict(test_x_vec)
predict9 = model9.predict(test_x_vec)
fp = open(sys.argv[5], 'w')
fp.write("id,label\n")
for i in range(20000):
	ans = 0
	count = 0.
	count += predict1[i]
	count += predict2[i]
	count += predict3[i]
	count += predict4[i]
	count += predict5[i]
	count += predict6[i]
	count += predict7[i]
	count += predict8[i]
	count += predict9[i]
	if count >= 4.5:
		ans = 1
	else:
		ans = 0
	fp.write("%d,%d\n" %(i ,ans))
fp.close()

