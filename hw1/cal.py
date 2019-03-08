import sys
import numpy as np
dimension = 163
def read_data(name):
		data = []
		f = open(name, "r", encoding = "big5")
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			line = line.split(",")
			data.append(line)
		return data

w = np.load("model.npy")
test_data = read_data(sys.argv[1])
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
fp = open(sys.argv[2], "w")
fp.write("id,value\n")
for i in range(240):
	if predict[i] < 0:
		predict[i] = 0.
	fp.write("id_%d,%f\n"%(i,predict[i]))
fp.close()
