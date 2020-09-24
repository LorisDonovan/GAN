# GAN
Python implementation of GAN using pytorch

## Run
* ```main.py``` contains the training loop for the GAN
* remember to check the ```DATA_PATH``` variable, and change it to your local directory
* there is a Generator and Discriminator for both Multilayer perceptron model as well as CNN
* you can train the GAN using either, all you have to do is use ```Generator``` or ```GeneratorConv``` likewise ```Discriminator``` or ```DiscriminatorConv``` as ```generator``` and ```discriminator``` in the training loop
* the training loop will generate the saved models and checkpoints, which you will need for generating gifs and images
* ```viewOutput.py``` contains the code for generating gifs or images of the output of the GAN
* to generate gifs use the ```showAnimation()``` function which takes in the Generator net and device (i.e., cpu or gpu)
* to generate images use the ```showOutputGrid()``` function which takes in the Generator net and device (i.e., cpu or gpu)
