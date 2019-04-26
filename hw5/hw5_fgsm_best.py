import numpy as np
import sys
import torch 
import torch.nn as nn
from scipy.misc import imsave
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 
from torch.autograd import Variable
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
# using pretrain proxy model, ex. VGG16, VGG19...
model = resnet50(pretrained=True)
# use eval mode
model.eval()

# loss criterion
loss = nn.CrossEntropyLoss()

trans = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    				])
invtrans1 = transforms.Compose([ 
					transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
					transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [ 1., 1., 1. ]),
                    ])
invtrans2 = transforms.Compose([ 
					transforms.ToPILImage(mode=None)
					])



epsilon = 0.01
count = 0
for i in range(200):
	raw_image = sys.argv[1]+'%03d' %i+'.png'
	image = Image.open(raw_image)

	originalimage = image	

	image = trans(image)
	image = image.unsqueeze(0)
	predict = model(image)
	predict = np.argmax(predict.data.numpy(), axis = 1)
	predict = torch.tensor(predict)
	image_variable = Variable(image, requires_grad=True)
	for j in range(4):
		print(i, j)

		zero_gradients(image_variable)

		output = model(image_variable)
		imgloss = loss(output, predict)
		imgloss.backward() 
		image_variable.data = image_variable.data + epsilon * torch.sign(image_variable.grad.data)
	attackimage = torch.squeeze(image_variable.data)
	attackimage = invtrans1(attackimage) 
	attackimage = torch.clamp(attackimage, min=0., max=1.)
	attackimage = invtrans2(attackimage)
	attackimage.save(sys.argv[2] + '%03d'%i + '.png')
	
	originalimage = trans(originalimage)
	originalimage = originalimage.unsqueeze(0)
	originalimage.requires_grad = True
	originalpredict = model(originalimage)
	originalpredict = np.argmax(originalpredict.data.numpy(), axis = 1)
	
	attackimage = trans(attackimage)
	attackimage = attackimage.unsqueeze(0)
	attackimage.requires_grad = True
	attackpredict = model(attackimage)
	attackpredict = np.argmax(attackpredict.data.numpy(), axis = 1)
	if(originalpredict != attackpredict):
		count += 1
count /= 200
#print("success rate =",count)
