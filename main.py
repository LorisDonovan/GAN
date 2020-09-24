import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as Datasets
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

from generatorModel import Generator, GeneratorConv
from discriminatorModel import Discriminator, DiscriminatorConv


DATA_PATH = "C:/_MyFiles/Python/_MachineLearning/_Datasets"
SAVED_MODEL_PATH = "SavedModels/"

EPOCHS = 30
LOG_INTERVAL = 100
SAVE_INTERVAL = 2
NOISE_SIZE = 100
BATCH_SIZE = 256
LOAD_MODEL = False

# DataLoader
compose = transforms.Compose([
	transforms.ToTensor() 
	,transforms.Normalize((0.5,), (0.5,)) # for dataset with values ranging [-1, 1]
])

trainSet = Datasets.MNIST(DATA_PATH, train=True, transform=compose, download=True)
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)


def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	generator = GeneratorConv(NOISE_SIZE).to(device)
	discriminator = DiscriminatorConv().to(device)

	generatorOptim = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
	discriminatorOptim = optim.Adam(discriminator.parameters(), lr=5e-4, betas=(0.5, 0.999))
	
	if LOAD_MODEL:
		generator.load_state_dict(torch.load(SAVED_MODEL_PATH + "generator.pt"))
		discriminator.load_state_dict(torch.load(SAVED_MODEL_PATH + "discriminator.pt"))
		generatorOptim.load_state_dict(torch.load(SAVED_MODEL_PATH + "generatorOptim.pt"))
		discriminatorOptim.load_state_dict(torch.load(SAVED_MODEL_PATH + "discriminatorOptim.pt"))

	criterion = nn.BCELoss()
	enumeratePerBatch = int(len(trainSet)/BATCH_SIZE)
	
	for epoch in range(EPOCHS):
		for i, (images, _) in enumerate(trainLoader):
			# training the discriminator with real images
			realImages = images.to(device)
			realOutput = discriminator(realImages)
			# realLabels = torch.empty(images.size(0), 1, device=device).uniform_(0.8, 1.0)
			realLabels = torch.ones(images.size(0), 1, device=device)
			lossReal = criterion(realOutput, realLabels)

			discriminator.zero_grad()
			lossReal.backward()

			# training the discriminator with fake images
			noise = torch.randn(images.size(0), NOISE_SIZE, device=device)
			fakeImages = generator(noise)
			fakeLabels = torch.zeros(images.size(0), 1, device=device)
			fakeOutput = discriminator(fakeImages.detach())
			lossFake = criterion(fakeOutput, fakeLabels)

			lossFake.backward()
			dLoss = lossReal + lossFake
			discriminatorOptim.step()

			# training the generator
			fakeOutput = discriminator(fakeImages)
			fakeLabels.fill_(1)
			gLoss = criterion(fakeOutput, fakeLabels)
			generator.zero_grad()
			gLoss.backward()
			generatorOptim.step()

			if i % LOG_INTERVAL == 0:
				print(f"[{epoch+1:3d}/{EPOCHS:3d}][{i:3d}/{enumeratePerBatch}] dLoss = {dLoss:.4f} | gLoss = {gLoss:.4f}")

		if epoch % SAVE_INTERVAL == 0:
			torch.save(generator.state_dict(), SAVED_MODEL_PATH + "generatorCheckpoint-" + str(epoch) + ".pt")
			torch.save(discriminator.state_dict(), SAVED_MODEL_PATH + "discriminatorCheckpoint-" + str(epoch) + ".pt")

	print("Training complete!")
	torch.save(generator.state_dict(), SAVED_MODEL_PATH + "generator.pt")
	torch.save(discriminator.state_dict(), SAVED_MODEL_PATH + "discriminator.pt")
	torch.save(generatorOptim.state_dict(), SAVED_MODEL_PATH + "generatorOptim.pt")
	torch.save(discriminatorOptim.state_dict(), SAVED_MODEL_PATH + "discriminatorOptim.pt")

if __name__ == '__main__':
	main()