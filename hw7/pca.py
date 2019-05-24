import numpy as np
import sys
import os
from skimage.io import imread, imsave
IMAGE_PATH = sys.argv[1]
img_shape = (600,600,3)
#test_image = ['1.jpg','10.jpg','22.jpg','37.jpg','72.jpg'] 
def process(input):
    M = np.copy(input)
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M
X_train = []

for i in range(415):
	path = os.path.join(IMAGE_PATH,'%d.jpg'%(i))
	image = imread(path)
	X_train.append(image.flatten())

X_train = np.array(X_train).astype('float32')
image_mean = np.mean(X_train, axis = 0)  
X_train -= image_mean
u, s, v = np.linalg.svd(np.transpose(X_train), full_matrices = False)
u_T = np.transpose(u)
"""
#------a------#
average = process(image_mean)
imsave('average.jpg', average.reshape(img_shape))
#------b------#
for x in range(5):
    eigenface = process(u_T[x])
    imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))  
"""
#------c------#
input_img_path = sys.argv[2]
input_img = imread(input_img_path)
X = input_img.flatten().astype('float32') 
X -= image_mean

# Compression
weight = np.array([X.dot(u_T[i]) for i in range(5)])  
# Reconstruction
reconstruct = process(np.transpose(u_T[0:5]).dot(weight) + image_mean)
imsave(sys.argv[3], reconstruct.reshape(img_shape)) 
"""
#------d------#
for i in range(5):
    number = s[i] * 100 / sum(s)
    print("%.3s"%(number))
"""
