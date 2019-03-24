import numpy as np
import math
import pandas as pd
import sys
from numpy.linalg import inv
train = pd.read_csv(sys.argv[1], header=0)
train_x = pd.read_csv(sys.argv[3], header=0)
train_y = pd.read_csv(sys.argv[4], header=0)
test_x = pd.read_csv(sys.argv[5], header=0)

train_x = np.array(train_x)
test_x = np.array(test_x)
mean = np.mean(train_x, axis = 0)
std = np.std(train_x, axis = 0)
for x in range(train_x.shape[0]):
	for y in range(train_x.shape[1]):
		if std[y] != 0:
			train_x[x][y] = (train_x[x][y] - mean[y]) / std[y]
for x in range(test_x.shape[0]):
	for y in range(test_x.shape[1]):
		if std[y] != 0:
			test_x[x][y] = (test_x[x][y] - mean[y]) / std[y]


train_y = np.array(train_y)
class_0 = []
class_1 = []
for i in range(train_x.shape[0]):
	if(train_y[i] == 0):
		class_0.append(train_x[i])
	else:
		class_1.append(train_x[i])
class_0 = np.array(class_0)
class_1 = np.array(class_1)
mean_0 = np.mean(class_0,axis = 0)
mean_1 = np.mean(class_1,axis = 0)
d = train_x.shape[1]
sigma_0 = np.zeros((d,d))
sigma_1 = np.zeros((d,d))
for i in range(train_x.shape[0]):
	if(train_y[i] == 0):
		sigma_0 += np.dot(np.transpose([train_x[i]-mean_0]), [train_x[i]-mean_0])
	else:
		sigma_1 += np.dot(np.transpose([train_x[i]-mean_1]), [train_x[i]-mean_1])
sigma_0 = sigma_0 / class_0.shape[0]
sigma_1 = sigma_1 / class_1.shape[0]
sigma = (sigma_0 * class_0.shape[0]  + sigma_1 * class_1.shape[0])/train_x.shape[0]
inv_sigma = inv(sigma)
w = np.dot(mean_0 - mean_1, inv_sigma)
b = -(0.5)*mean_0.dot(inv_sigma).dot(mean_0) + 0.5*mean_1.dot(inv_sigma).dot(mean_1) + np.log(float(class_0.shape[0]) / class_1.shape[0])

def sigmoid(x):
	return (1 / (1 + np.exp(-x)))
predict = []
for i in range(test_x.shape[0]):
	y = sigmoid(np.dot(w, test_x[i]) + b)
	if y > 0.5:
		predict.append(0)
	else:
		predict.append(1)

fp = open( sys.argv[6], "w")
fp.write("id,label\n")
for i in range(test_x.shape[0]):
	fp.write("%d,%d\n" %(i + 1,predict[i]))
fp.close()
