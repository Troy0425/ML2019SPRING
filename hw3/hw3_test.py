import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense,Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dropout, BatchNormalization
from keras.optimizers import  Adam
from keras.activations import relu, softmax
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

def readtest(file):
	print("Reading %s..." %(file))
	data = []
	label = []

	raw_data = np.genfromtxt(file, delimiter=',', dtype=str, skip_header=1)
	for i in range(len(raw_data)):
		tmpx = np.array(raw_data[i, 1].split(' ')).reshape(48, 48, 1)
		data.append(tmpx)
		
	data = np.array(data, dtype=np.float32) / 255.0
	return data
x_test = readtest(sys.argv[1])

model = load_model('CNNmodel')
predict = model.predict(x_test, batch_size=1024)

predict = np.argmax(predict, axis = 1).reshape(predict.shape[0], 1)
fp = open(sys.argv[2], 'w')
fp.write("id,label\n")
for i in range(predict.shape[0]):
	fp.write("%d,%d\n" %(i ,predict[i]))
fp.close()
