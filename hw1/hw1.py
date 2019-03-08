import numpy as np
import math
def read_data(name):
		data = []
		f = open(name, "r", encoding = "big5")
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			line = line.split(",")
			data.append(line)
		return data
train_data = read_data("train.csv")
del train_data[0]
data = [[] for i in range(18)]
k = 0
for i_row in train_data:
	for j in range(3, 27):
		if i_row[j] == "NR":
			i_row[j] = 0
		i_row[j] = float(i_row[j])
		if i_row[j] < 0:
			if k % 18 == 9:
				if j == 3:
					i_row[j] = max(float(i_row[j + 1]), 0)
				elif j == 26:
					i_row[j] = i_row[j - 1]
				else:
					if float(i_row[j+1]) != -1:
						i_row[j] = (float(i_row[j + 1]) + i_row[j - 1]) / 2
					else:
						i_row[j] = i_row[j-1]
			#elif k % 18 != 0:
			#	i_row[j] = float()#((i_row[j]) * (-1))
		if k % 18 == 15 or k % 18 == 14:
			i_row[j] = 0#0.0000000000001
		data[k % 18].append(float(i_row[j]))
	k += 1
train_x = [[] for i in range(5652)]
train_y = [[] for i in range(5652)]
for month in range(12):
	for hr in range(471):
		for cat in range(18):
			for i in range(9):
				train_x[month * 471 + hr].append(data[cat][month * 480 + hr + i])
				###second order
				#if cat == 9:
				#	train_x[month * 471 + hr].append(data[cat][month * 480 + hr + i]**2)
				###
		train_x[month * 471 + hr].append(1)
		train_y[month * 471 + hr].append(data[9][month * 480 + hr + 9])
train_x = np.array(train_x)
train_y = np.array(train_y)
dimension = 163# + 9
w = np.zeros(dimension).reshape(dimension,1)
lr = 0.1
lda = 0
iteration = 100000
xTx = np.dot(train_x.transpose(), train_x)
xTy = np.dot(train_x.transpose(), train_y)
def Ein(w):
	sum = 0.
	for i in range(5652):
		sum += (np.dot(train_x[i].transpose(), w) - train_y[i]) ** 2
	sum /= 5652
	return sum ** (1/2)
prev_grad = float()
for i in range(iteration):
	if i % 10000 == 0:
		print(i, Ein(w))
	gradient = 2.0 * (np.dot(xTx, w) - xTy) + 2 * lda + w
	prev_grad += gradient ** 2
	ada = np.sqrt(prev_grad)
	w -= lr * gradient / (ada + 0.0000000000001) 
np.save("model", w)
"""test_data = read_data("test.csv")
tdata = []
k = 0
for i_row in test_data:
	for j in range(2,11):
		if i_row[j] == "NR":
			i_row[j] = 0
		i_row[j] = float(i_row[j])
		if i_row[j] < 0:
			if k % 18 == 9:
				if j == 2:
					i_row[j] = max(float(i_row[j + 1]), 0)
				elif j == 10:
					i_row[j] = i_row[j - 1]
				else:
					if float(i_row[j+1]) != -1:
						i_row[j] = (float(i_row[j + 1]) + i_row[j - 1]) / 2
					else:
						i_row[j] = i_row[j-1]
			#elif k % 18 != 0:
			#	i_row[j] = float()#((i_row[j]) * (-1))
		if k % 18 == 15 or k % 18 == 14:
			i_row[j] = 0#0.0000000000001
		tdata.append(float(i_row[j]))
		###second order
		#if k % 18 == 9:
		#	tdata.append(float(i_row[j]) ** 2)
		###
	k += 1
	if k % 18 == 0:
		tdata.append(1)
tdata = np.array(tdata).reshape(240,dimension)
predict = np.dot(tdata, w)
fp = open("predict.csv", "w")
fp.write("id,value\n")
for i in range(240):
	if predict[i] < 0:
		predict[i] = 0.
	fp.write("id_%d,%f\n"%(i,predict[i]))
fp.close()"""
