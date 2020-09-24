# Adversarial Nets
* generator distribution $p_g$
* data $x$
* input noise variables $p_z(z)$
* Generator net $G(z;\theta_g)$
* Discriminator net $D(z;\theta_d)$ that outputs a single scalar
* $x$ came from the data rather than $p_g$
* Train $D$ to maximize the probability of assigning the correct label to both training examples and samples from $G$
* Simultaneously train $G$ to minimize $\log(1-D(G(z)))$


# Algorithm 

**for** number of training iterations **do**\
&emsp;**for** $k$ steps **do**\
&emsp;&emsp;Sample minibatch of $m$ noise samples $\{z^{(1)},...,z^{(m)}\}$ from noise prior $p_g(z)$\
&emsp;&emsp;Sample minibatch of $m$ examples $\{x^{(1)},...,x^{(m)}\}$ from generating distribution $p_{data}(x)$\
&emsp;&emsp;Update the discriminator by ascending its stochastic gradient:\
$$\nabla_{\theta_d}\frac{1}{m}\sum_{i=1}^{m}{[\log D(x^{(i)}) + \log{(1-D(G(z^{(i)})))}]}$$
&emsp;**end for**\
&emsp;Sample minibatch of $m$ noise samples $\{z^{(1)},...,z^{(m)}\}$ from noise prior $p_g(z)$\
&emsp;Update the generator by descending its stochastic gradient:
$$\nabla_{\theta_g}\frac{1}{m}\sum_{i=1}^{m}{\log{(1-D(G(z^{(i)})))}}$$
**end for**


# Experiments
* Generator net used a mixture of ReLU and sigmoid activations
* Discriminator net used maxout activations (leaky relu)
* Dropout was applied in training the discriminator net
* Used noise as input only to the bottommost layer of the generator network

## Loss function used (by me)
* Discriminator updated by gradient: 
$$\nabla_{\theta_d}\frac{1}{m}\sum_{i=1}^{m}{[\log D(x^{(i)}) + \log{(1-D(G(z^{(i)})))}]}$$
* Binary Cross Entropy:
$$BCE = -[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$
&emsp;&emsp;where, $\hat{y}_i$ is predicted values and $y_i$ is target values
* for loss related to real images: $\hat{y}_i = D(x^{(i)}),\ y_i = 1$
* for loss related to fake images: $\hat{y}_i = D(G(z^{(i)})),\ y_i = 0$
* instead of minimizing $\log{(1-D(G(z^{(i)})))}$ we maximize $\log{(D(G(z^{(i)})))}$ for generator loss


# Architectural guidelines for DCGAN
* Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).
* Use batchnorm in both the generator and the discriminator.
* Remove fully connected hidden layers for deeper architectures.
* Use ReLU activation in generator for all layers except for the output, which uses Tanh.
* Use LeakyReLU activation in the discriminator for all layers.

## Hyperparameters, Preprocessing and Architecture
* Not applying batchnorm to the generator output layer and the discriminator input layer
* No pre-processing was applied to training images besides scaling to the range of the tanh activation
function [-1, 1]. 
* All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128. 
* In the LeakyReLU, the slope of the leak was set to 0.2 in all models
* Used Adam optimizer with learning rate 0.0002 and momentum term $\beta_1$ at 0.5