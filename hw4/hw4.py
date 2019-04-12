import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense,Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dropout, BatchNormalization
from keras.optimizers import  Adam
from keras.activations import relu, softmax
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import plot_model
from keras import backend as K
from lime import lime_image
from skimage.segmentation import slic
from skimage import color
#from vis.visualization import visualize_cam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
x_train, y_train = readtrain(sys.argv[1])
model = load_model("CNNmodel")
names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
y_train_label = []
for y in y_train:
	for i in range(7):
		if y[i] == 1:
			y_train_label.append(i)
			break
IDs = [22, 845, 862, 28, 6, 15, 858]
t = 0
#allfig, ax = plt.subplots(7, 2, figsize = (12, 16))

###Saliency map
for i in IDs:

	pred = np.argmax(model.predict(x_train[i].reshape(-1, 48, 48, 1)))
	target = K.mean(model.output[:, pred])
	grads = K.gradients(target, model.input)[0]
	f = K.function([model.input, K.learning_phase()], [grads])

	y_grads = f([x_train[i].reshape(-1, 48, 48, 1) , 0])[0].reshape(48, 48, -1)

	y_grads *= -1
	y_grads = np.max(np.abs(y_grads), axis=-1, keepdims=True)

# normalize
	y_grads = (y_grads - np.mean(y_grads)) / (np.std(y_grads) + 1e-5)
	y_grads *= 0.1


# clip to [0, 1]
	y_grads += 0.5
	y_grads = np.clip(y_grads, 0, 1)

	heatmap = y_grads.reshape(48, 48)

	#ax[t, 0].imshow((x_train[i]*255).reshape((48, 48)), cmap = 'gray')
	#cax = ax[t, 1].imshow(heatmap, cmap = 'jet')
	#allfig.colorbar(cax, ax = ax[t, 1])
	#t += 1

	plt.figure()
	plt.imshow(heatmap, cmap=plt.cm.jet)
	plt.colorbar()
	plt.tight_layout()
	path = sys.argv[2]+'fig1_'+str(t)+'.jpg'
	fig = plt.gcf()
	fig.savefig(path)
	plt.show()

	t += 1
	

###filter activation
layer_output = model.layers[1].output
img_ascs = list()
for filter_index in range(32):
	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	loss = K.mean(layer_output[:, :, :, filter_index])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, model.input)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([model.input], [loss, grads])

	# step size for gradient ascent
	step = 1
	np.random.seed(0)
	img_asc = np.random.randn(48,48).reshape((1, 48, 48, 1))#.astype(np.float64) 
	# run gradient ascent for 20 steps
	for i in range(100):
		loss_value, grads_value = iterate([img_asc])
		img_asc += grads_value * step

	img_asc = img_asc[0]
	img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))
fig2, ax = plt.subplots(4, 8, figsize = (12, 12))
for i in range(32):
	ax[i//8, i%8].imshow(img_ascs[i], cmap = 'Blues')
	ax[i//8, i%8].set_title('filter %d' % (i))

path = sys.argv[2]+'fig2_1'+'.jpg'
fig2.savefig(path)
plt.show()
###
f = K.function([model.input, K.learning_phase()], [model.layers[1].output])
output = f([x_train[0].reshape(1,48,48,1),0])[0]
fig3, ax = plt.subplots(4, 8, figsize = (12, 12))
for i in range(32):
	ax[i//8, i%8].imshow(output[0,:,:,i], cmap = 'Blues')
	ax[i//8, i%8].set_title('filter %d' % (i))
path = sys.argv[2]+'fig2_2'+'.jpg'
fig3.savefig(path)
plt.show()
"""
"""
###Lime
def predict(input):
	return model.predict(color.rgb2gray(input).reshape(-1,48,48,1)).reshape(-1,7)

def segmentation(input):
	return slic(input)

for i in range(len(IDs)):
	np.random.seed(0)
	grayx_train = x_train[IDs[i]].reshape(48,48)
	rgbx_train = color.gray2rgb(grayx_train)
	explainer = lime_image.LimeImageExplainer()
# Get the explaination of an image
	explaination = explainer.explain_instance(
									image=rgbx_train, 
									classifier_fn=predict,
									segmentation_fn=segmentation
									)
# Get processed image
	image, mask = explaination.get_image_and_mask(
									label=y_train_label[IDs[i]],
									positive_only=False,
									hide_rest=False,
									num_features=5,
									min_weight=0.0
									)		

	# save the image
	plt.figure()
	plt.imshow(image)
	path = sys.argv[2]+'fig3_'+str(i)+'.jpg'
	plt.savefig(path)
	plt.show()

###

"""for i in range(len(IDs)):
	idx = IDs[i]
	img = visualize_cam(model, -1, i, x_train[idx], backprop_modifier=None, grad_modifier=None)
	plt.imshow(img)
	path = sys.argv[2]+'fig4_'+str(i)+'.jpg'
	plt.savefig(path)
	plt.show()
	"""
