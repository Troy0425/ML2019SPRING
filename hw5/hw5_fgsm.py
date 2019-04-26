import numpy as np
import sys
import torch 
import torch.nn as nn
from scipy.misc import imsave
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 

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



epsilon = 0.04
for i in range(200):
	print(i)
	raw_image = sys.argv[1]+'%03d' %i+'.png'
	image = Image.open(raw_image)
	
	image = trans(image)
	image = image.unsqueeze(0)
	image.requires_grad = True
	predict = model(image)
	predict = np.argmax(predict.data.numpy(), axis = 1)
	predict = torch.tensor(predict)

	zero_gradients(image)

	output = model(image)
	imgloss = loss(output, predict)
	imgloss.backward() 

	attackimage = image + epsilon * image.grad.sign_()
	attackimage = torch.squeeze(attackimage)
	attackimage = invtrans1(attackimage) 
	attackimage = torch.clamp(attackimage, min=0., max=1.)
	attackimage = invtrans2(attackimage)
	attackimage.save(sys.argv[2] + '%03d'%i + '.png')

