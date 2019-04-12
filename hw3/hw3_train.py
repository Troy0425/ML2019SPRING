import numpy as np
import sys
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dropout, BatchNormalization
from keras.optimizers import  Adam
from keras.activations import relu, softmax
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

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

	data = np.array(data, dtype=np.float32) / 255.0
	label = np.array(label, dtype=int)
	return data, label

x_train, y_train = readtrain(sys.argv[1])

model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape = (48, 48, 1), padding = 'same', activation = 'relu' ) )
model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(768, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(768, (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units=1024, activation = 'relu'))
model.add(Dropout(0.45))

model.add(Dense(units=1024, activation = 'relu'))
model.add(Dropout(0.45))

model.add(Dense(units=1024, activation = 'relu'))
model.add(Dropout(0.45))

model.add(Dense(units=7, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(rotation_range=30,
							width_shift_range=0.15,
							height_shift_range=0.15,
							zoom_range=[0.8, 1.2],
							horizontal_flip=True)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=512),
					steps_per_epoch=int(x_train.shape[0]/512),
					epochs=300)

model.save('CNNmodel')
