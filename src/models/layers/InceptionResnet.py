import torch.nn as nn
import torch

class Residual(nn.Module):
	def __init__(self, numIn,numOut):
		super(Residual, self).__init__()
		self.numIn = numIn
		self.numOut = numOut
		self.bn = nn.BatchNorm2d(self.numIn)
		self.bn1 = nn.BatchNorm2d(self.numIn*2)
		self.relu = nn.ReLU(inplace = True)

		self.conv1 = nn.Conv2d(self.numIn, self.numIn, bias = True, kernel_size = 1)
		self.conv2 = nn.Conv2d(self.numIn, self.numIn, bias = True, kernel_size = 3, stride = 1, padding = 1)
		self.conv3 = nn.Conv2d(self.numIn*2, self.numIn, bias = True, kernel_size = 1)
		self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias = True, kernel_size = 1)

	def forward(self,x):
		residual = x
		out1 = self.bn(x)
		out1 = self.relu(out1)
		out1 = self.conv1(out1)
		out1 = self.bn(out1)
		out1 = self.relu(out1)
		out1 = self.conv2(out1)

		out2 = self.bn(x)
		out2 = self.relu(out2)
		out2 = self.conv1(out2)
		out2 = self.bn(out2)
		out2 = self.relu(out2)
		out2 = self.conv2(out2)
		out2 = self.bn(out2)
		out2 = self.relu(out2)
		out2 = self.conv2(out2)
		out3 = torch.cat((out1,out2),1)
		out3 = self.bn1(out3)
		out3 = self.relu(out3)
		out3 = self.conv3(out3)

		out = residual+out3

		out = self.conv4(out)

		return out 



