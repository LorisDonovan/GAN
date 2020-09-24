import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
import math


class Generator(nn.Module):
	def __init__(self, noiseSize=100):
		super(Generator, self).__init__()

		out_feature = 784

		self.fc1 = nn.Linear(noiseSize, 256)
		self.fc2 = nn.Linear(256, 512)
		self.fc3 = nn.Linear(512, 1024)
		self.out = nn.Linear(1024, out_feature)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.relu(self.fc3(x))
		x = torch.tanh(self.out(x))
		return x 

class GeneratorConv(nn.Module):
	def __init__(self, noiseSize=100, out_feature=1):
		super(GeneratorConv, self).__init__()

		self.conv1 = nn.ConvTranspose2d(noiseSize, 256, 4, bias=False)
		self.conv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, bias=False)
		self.conv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
		self.conv4 = nn.ConvTranspose2d(64, out_feature, 4, stride=2, padding=1, bias=False)

		self.batchNorm1 = nn.BatchNorm2d(256)
		self.batchNorm2 = nn.BatchNorm2d(128)
		self.batchNorm3 = nn.BatchNorm2d(64)
		self.noiseSize = noiseSize

	def forward(self, x):
		x = torch.relu(self.batchNorm1(self.conv1(x.reshape(-1, self.noiseSize, 1, 1))))
		x = torch.relu(self.batchNorm2(self.conv2(x)))
		x = torch.relu(self.batchNorm3(self.conv3(x)))
		x = torch.tanh(self.conv4(x))
		return x 


def main():
	net = GeneratorConv()
	print(net(torch.randn(1, 100)).shape)

if __name__ == '__main__':
	main()
