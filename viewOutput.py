import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch 
import torchvision.utils as vutils

from generatorModel import Generator, GeneratorConv
from discriminatorModel import Discriminator, DiscriminatorConv


def showOutputGrid(generator, device):
	generator.eval()
	nrow = 8
	ncol = 8
	num = nrow * ncol
	with torch.no_grad():
		output = generator(torch.randn(num, 100, device=device)).to('cpu')
	
	grid = vutils.make_grid(output, nrow=nrow, pad_value=2)

	plt.axis("off")
	plt.imshow(np.transpose(grid, (1,2,0)))
	plt.show()

def showAnimation(generator, device):
	generator.eval()
	nrow = 8
	ncol = 18
	num = nrow * ncol
	noise = torch.randn(num, 100, device=device)
	imgList = []
	
	for i in range(15):
		generator.load_state_dict(torch.load("SavedModels/generatorCheckpoint-"+str(i*2)+".pt"))
		with torch.no_grad():
			output = generator(noise).to('cpu')
		imgList.append(vutils.make_grid(output, nrow=ncol, pad_value=2))

	fig = plt.figure(figsize=(8,8))
	plt.axis("off")
	ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in imgList]
	ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
	plt.show()
	ani.save('anim.gif', writer='imagemagick')


def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	generator = GeneratorConv().to(device)
	
	# generator.load_state_dict(torch.load("SavedModels/generator.pt"))
	# showOutputGrid(generator, device)

	showAnimation(generator, device)

if __name__ == '__main__':
	main()
