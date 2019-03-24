import numpy as np
import math
import pandas as pd
import sys
train = pd.read_csv(sys.argv[1], header=0)
train_x = pd.read_csv(sys.argv[3], header=0)
train_y = pd.read_csv(sys.argv[4], header=0)
test_x = pd.read_csv(sys.argv[5], header=0)

train_x = np.array(train_x)
mean = np.mean(train_x, axis = 0)
std = np.std(train_x, axis = 0)
for x in range(train_x.shape[0]):
	for y in range(train_x.shape[1]):
		if std[y] != 0:
			train_x[x][y] = (train_x[x][y] - mean[y]) / std[y]

test_x = np.array(test_x)
for x in range(test_x.shape[0]):
	for y in range(test_x.shape[1]):
		if std[y] != 0:
			test_x[x][y] = (test_x[x][y] - mean[y]) / std[y]


w = np.zeros(train_x.shape[1]).reshape(train_x.shape[1], 1)
lr = 0.1
lda = 0
iteration = 1000
fx = np.zeros(train_x.shape[1]).reshape(train_x.shape[1], 1)
xTx = np.dot(train_x.transpose(), train_x)
xTy = np.dot(train_x.transpose(), train_y)
xT = train_x.transpose()
n = train_x.shape[0]
def sigmoid(z):
	z = (1. / (1. + np.exp(-z)))
	return z

def Ein(w):
	count = 0
	y = np.dot(train_x, w)
	for i in range(n):
		y[i] = sigmoid(y[i])
	for i in range(n):
		if y[i] >= 0.5:
			y[i] = 1
		else:
			y[i] = 0
		if y[i] == train_y[i]:
			count += 1
	return count / train_x.shape[0]
prev_grad = float()
for i in range(iteration):
	#if i % 100 == 0:
		#print(i, Ein(w))
	fx = np.dot(train_x, w) # n * 1
	for j in range(n):
		fx[j] = sigmoid(fx[j])
	gradient = -1 * (xTy - np.dot(xT, fx))
	prev_grad += gradient ** 2
	ada = np.sqrt(prev_grad)
	w -= lr * gradient / (ada + 0.00000001)

count = 0
y = np.dot(test_x, w)
for i in range(test_x.shape[0]):
	y[i] = sigmoid(y[i])
for i in range(test_x.shape[0]):
	if y[i] >= 0.5:
		y[i] = 1
	else:
		y[i] = 0

fp = open(sys.argv[6], "w")
fp.write("id,label\n")
for i in range(test_x.shape[0]):
	fp.write("%d,%d\n" %(i + 1,y[i]))
fp.close()
