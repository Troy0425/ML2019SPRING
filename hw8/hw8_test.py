import numpy as np
import sys
from keras.models import Model
from keras.layers.core import Dense,Activation
from keras.layers import Input,Conv2D, MaxPooling2D, Flatten , DepthwiseConv2D,BatchNormalization,Layer,Dropout,GlobalMaxPooling2D
from keras.optimizers import  Adam
from keras.activations import relu, softmax
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import keras.backend as K
def relu6(x):
    return K.relu(x, max_value=6)
def _depthwise_conv_block(inputs, pointwise_conv_filters, strides=(1, 1)):
	x = DepthwiseConv2D((3, 3),padding='same',depth_multiplier=1,strides=strides)(inputs)
	x = Dropout(0.1)(x)
	x = BatchNormalization(axis = -1)(x)
	x = Activation(relu6)(x)
	x = Conv2D(pointwise_conv_filters, (1, 1),padding='same',use_bias=False,strides=(1, 1))(x)
	x = BatchNormalization(axis = -1)(x)
	return Activation(relu6)(x)
def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)): 
	x = Conv2D(filters, kernel,padding='same',use_bias=False,strides=strides)(inputs)
	x = Dropout(0.1)(x)
	x = BatchNormalization(axis = -1)(x)
	return Activation(relu6)(x)
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

input_img = Input(shape=(48, 48, 1))
x = _conv_block(input_img, 32, strides=(1, 1))
x = _depthwise_conv_block(x, 64, strides=(1, 1))
x = _depthwise_conv_block(x, 64, strides=(2, 2))
x = _depthwise_conv_block(x, 64, strides=(1, 1))
x = _depthwise_conv_block(x, 64, strides=(2, 2))
x = _depthwise_conv_block(x, 128,strides=(1, 1))
x = _depthwise_conv_block(x, 128,strides=(2, 2))
x = _depthwise_conv_block(x, 128,strides=(1, 1))
x = GlobalMaxPooling2D()(x)
#x = Flatten()(x)
x = Dense(units=56)(x)
x = Dense(units=7, activation = 'softmax')(x)
ar = np.load("best.npy")
model = Model(input_img, x)
model.set_weights(ar)
predict = model.predict(x_test)

predict = np.argmax(predict, axis = 1).reshape(predict.shape[0], 1)
fp = open(sys.argv[2], 'w')
fp.write("id,label\n")
for i in range(predict.shape[0]):
	fp.write("%d,%d\n" %(i ,predict[i]))
fp.close()
