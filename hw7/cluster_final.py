import numpy as np
import sys
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense,Activation
from keras.layers import Input, UpSampling2D, Conv2D, MaxPooling2D, Flatten , Dropout, BatchNormalization, Reshape
from keras.optimizers import  Adam
from keras.activations import relu, softmax
from PIL import Image
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import KMeans
X_train = []
for i in range(1, 40001):
	path = sys.argv[1] + '%06d.jpg'%(i)
	image = Image.open(path)
	image = np.array(image) / 255.0
	X_train.append(image)
X_train = np.array(X_train)
input_img = Input(shape=(32, 32, 3))
x = Conv2D(64, (3, 3), padding='same', activation = 'relu')(input_img)
x = Conv2D(64, (3, 3), padding='same',activation = 'relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), padding='same',activation = 'relu')(x)
x = Conv2D(128, (3, 3), padding='same',activation = 'relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
encoded = Dense(units = 32, activation = 'relu')(x)
x = Dense(units = 8192, activation = 'relu')(encoded)

x = Reshape((8, 8, -1))(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), padding='same',activation = 'relu')(x)
x = Conv2D(64, (3, 3), padding='same',activation = 'relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same',activation = 'relu')(x)
decoded = Conv2D(3, (3, 3), padding='same',activation = 'sigmoid')(x)

model = Model(input_img, decoded)
encoder = Model(input_img, encoded)
model.compile(optimizer='adam', loss='mse')
#print(model.summary())
#model.fit(X_train, X_train, batch_size=512, epochs=30,validation_split=0, shuffle=True)
#model.save("wholemodel")
#encoder.save("autoencoder")
encoder = load_model("autoencoder")
#model = load_model("wholemodel")
encoded_img = encoder.predict(X_train)
tsne = TSNE(n_jobs = 20, n_components = 2, random_state = 2)
encoded_img = tsne.fit_transform(encoded_img)

kmeans = KMeans(n_clusters=2).fit(encoded_img)
kmeans_label = kmeans.labels_

test_data = open(sys.argv[2], "r")
anscsv = open(sys.argv[3], "w")
anscsv.write("id,label\n")
lines = test_data.readlines()
del lines[0]
for line in lines:
	line = line.strip()
	line = line.split(",")
	if(kmeans_label[int(line[1])-1] == kmeans_label[int(line[2])-1]):	
		ans = 1
	else:
		ans = 0
	anscsv.write("%d,%d\n"%(int(line[0]),ans))
anscsv.close()	
test_data.close()



