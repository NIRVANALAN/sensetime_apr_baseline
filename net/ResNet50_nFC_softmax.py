import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
		init.constant_(m.bias.data, 0.0)
	elif classname.find('BatchNorm1d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		init.normal_(m.weight.data, std=0.001)
		init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
	def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
	             return_f=False):
		super(ClassBlock, self).__init__()
		self.return_f = return_f
		add_block = []
		if linear:
			add_block += [nn.Linear(input_dim, num_bottleneck)]
		else:
			num_bottleneck = input_dim
		if bnorm:
			add_block += [nn.BatchNorm1d(num_bottleneck)]
		if relu:
			add_block += [nn.LeakyReLU(0.1)]
		if droprate > 0:
			add_block += [nn.Dropout(p=droprate)]
		add_block = nn.Sequential(*add_block)
		add_block.apply(weights_init_kaiming)
		
		classifier = []
		classifier += [nn.Linear(num_bottleneck, class_num)]
		classifier = nn.Sequential(*classifier)
		classifier.apply(weights_init_classifier)
		
		self.add_block = add_block
		self.classifier = classifier
	
	def forward(self, x):
		x = self.add_block(x)
		if self.return_f:
			f = x
			x = self.classifier(x)
			return x, f
		else:
			x = self.classifier(x)
			return x


class ResNet50_nFC_softmax(nn.Module):
	def __init__(self, class_num, id_num, dropout=0.5, stride=1, **kwargs):
		super(ResNet50_nFC_softmax, self).__init__()
		self.model_name = 'resnet50_nfc_softmax'
		self.class_num = class_num
		self.id_num = id_num
		
		model_ft = models.resnet50(pretrained=True)
		self.model = model_ft
		
		if stride == 1:  # use baseline stride=1
			model_ft.layer4[0].downsample[0].stride = (1, 1)
			model_ft.layer4[0].conv2.stride = (1, 1)
		
		# avg pooling to global pooling
		model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		model_ft.fc = nn.Sequential()
		
		self.features = model_ft
		self.num_ftrs = 2048
		num_bottleneck = 512
		
		for c in range(self.class_num + 1):
			if c == self.class_num:  # for identity classification
				self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, id_num, dropout))
			# nn.Sequential(nn.Linear(self.num_ftrs, num_bottleneck),
			# 		   nn.BatchNorm1d(num_bottleneck),
			# 		   nn.LeakyReLU(0.1),
			# 		   nn.Dropout(p=dropout),
			# 		   nn.Linear(num_bottleneck, self.id_num)))
			else:
				self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, 2, dropout))
				# nn.Sequential(nn.Linear(self.num_ftrs, num_bottleneck),
				#               nn.BatchNorm1d(num_bottleneck),
				#               nn.LeakyReLU(0.1),
				#               nn.Dropout(p=dropout),
				#               nn.Linear(num_bottleneck, 2)))
			# else:
			# 	self.__setattr__('class_{}'.format(c),  # for attribute?
			# 	                 nn.Sequential(nn.Linear(self.num_ftrs, num_bottleneck),
			# 	                               nn.BatchNorm1d(num_bottleneck),
			# 	                               nn.LeakyReLU(0.1),
			# 	                               nn.Dropout(p=dropout),
			# 	                               nn.Linear(num_bottleneck, 2)))
	
	def forward(self, x):
		x = self.features(x)
		return list(self.__getattr__('class_%d' % c)(x) for c in range(self.class_num + 1)), x
