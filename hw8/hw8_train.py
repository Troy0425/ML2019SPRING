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
def readtrain(file):
	print("Reading %s..." %(file))
	data = []
	label = []

	raw_data = np.genfromtxt(file, delimiter=',', dtype=str, skip_header=1)
	for i in range(len(raw_data)):
		tmpx = np.array(raw_data[i, 1].split(' ')).reshape(48, 48, 1)
		data.append(tmpx)
		tmpy = [0,0,0,0,0,0,0]
		tmpy[int(raw_data[i, 0])] = 1
		label.append(tmpy)

	data = np.array(data, dtype=np.float16) / 255.0
	label = np.array(label, dtype=int)
	return data, label

x_train, y_train = readtrain(sys.argv[1])

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

model = Model(input_img, x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
datagen = ImageDataGenerator(rotation_range=30,
							width_shift_range=0.15,
							height_shift_range=0.15,
							zoom_range=[0.8, 1.2],
							horizontal_flip=True)

#train_num = int(x_train.shape[0] * 0.9)
#x_val = x_train[train_num:]
#y_val = y_train[train_num:]
#x_train = x_train[0:train_num]
#y_train = y_train[0:train_num]
model.fit_generator(datagen.flow(x_train, y_train, batch_size=512),
					steps_per_epoch=3*int(x_train.shape[0]/512),
					#					validation_data=(x_val, y_val),
					epochs=160)
npw = model.get_weights()
npw = [i.astype('float16') for i in npw]
np.save('best',npw)
