import torch 
import torch.nn as nn


class Discriminator(nn.Module):
	def __init__(self, in_feature=1, out_feature=1):
		super(Discriminator, self).__init__()

		in_feature = 784
		out_feature = 1

		self.fc1 = nn.Linear(in_feature, 1024)
		self.fc2 = nn.Linear(1024, 512)
		self.fc3 = nn.Linear(512, 256)
		self.out = nn.Linear(256, out_feature)

		self.leakyRelu = nn.LeakyReLU(0.2)
		self.dropout = nn.Dropout(0.3)

	def forward(self, x):
		x = self.leakyRelu(self.fc1(x.reshape(-1, 784)))
		x = self.dropout(x)
		x = self.leakyRelu(self.fc2(x))
		x = self.dropout(x)
		x = self.leakyRelu(self.fc3(x))
		x = self.dropout(x)
		x = torch.sigmoid(self.out(x))
		return x 

class DiscriminatorConv(nn.Module):
	def __init__(self, in_feature=1, out_feature=1):
		super(DiscriminatorConv, self).__init__()

		self.conv1 = nn.Conv2d(in_feature, 64, 4, stride=2, padding=1, bias=False)
		self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
		self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)
		self.conv4 = nn.Conv2d(256, out_feature, 3, stride=1, padding=0, bias=False)

		self.leakyRelu = nn.LeakyReLU(0.2)

	def forward(self, x):
		x = self.leakyRelu(self.conv1(x))
		x = self.leakyRelu(self.conv2(x))
		x = self.leakyRelu(self.conv3(x))
		x = torch.sigmoid(self.conv4(x))
		return x.reshape(-1, 1) 


def main():
	net = DiscriminatorConv()
	print(net(torch.randn(64, 1, 28, 28)).shape)

if __name__ == "__main__":
	main()