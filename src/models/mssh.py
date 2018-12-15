from .layers.Residual import Residual
from .layers.InceptionResnet import InceptionResnet
import torch.nn as nn
import math

class MSSH(nn.Module):
	def __init__(self):
		super(MSSH,self).__init__()
		self.bn1 = nn.BatchNorm2d(3)
		self.relu = nn.ReLU(inplace = True)
		self.conv1 = nn.Conv2d(3, 64, bias = True, kernel_size = 7, stride = 2, padding = 3)
		self.inception1 = Residual(64,128)
		self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.inception2 = Residual(128,128)
		self.inception3 = Residual(128,256)
		self.up = nn.Upsample(scale_factor = 2)
		self.conv2 = nn.Conv2d(256, 16, bias = True, kernel_size = 1, stride = 1)

	def forward(self,x):
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv1(x)
		x = self.inception1(x)

		x1 = self.maxpool(x) ##64

		x1 = self.inception2(x1)
		x1 = self.inception3(x1)

		x2 = self.maxpool(x1) ##32

		x3 = self.maxpool(x2) ##16

		x4 = self.maxpool(x3) ##8

		x5  = self.maxpool(x4) ##4

		## Upsample

		x6 = self.up(x5) + x4 ## 8
		x7 = self.up(x6) + x3 ## 16
		x8 = self.up(x7) + x2 ## 32
		x9 = self.up(x8) + x1 ## 64

		out =  self.conv2(x9)

		return out



