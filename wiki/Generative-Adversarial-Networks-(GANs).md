## Table of Contents
* [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
* [How GANs Work](#how-gans-work)
* [GANs Process](#gans-process)

## Generative Adversarial Networks (GANs)
Generative Adversarial Networks are used to generate images that never existed before. They learn about the world (objects, animals and so forth) and create new versions of those images that never existed.

They have two components:
* A Generator - this creates the images.
* A Discriminator - this assesses the images and tells the generator if they are similar to what it has been trained on. 
  These are based off real world examples.

When training the network, both the generator and discriminator start from scratch and learn together.

## How GANs Work
G for Generative - this is a model that takes an input as a random noise singal and then outputs an image.

![Generative](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/generative.png)

A for Adversarial - this is the discriminator, the opponent of the generator. This is capable of learning about objects, animals or other features specified. For example: if you supply it with pictures of dogs and non-dogs, it would be able to identify the difference between the two.

![Discriminator Example](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/discriminator-example.png)

Using this example, once the discriminator has been trained, showing the discriminator a picture that isn't a dog it will return a 0. Whereas, if you show it a dog it will return a 1.

![Discriminator Scores](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/discriminator-scores.png)

N for Network - meaning the generator and discriminator are both neural networks.

## GANs Process
Step 1 - we input a random noise signal into the generator. The generator creates some images which is used for training the discriminator. We provide the discriminator with some features/images we want it to learn and the discriminator outputs probabilities. These probabilities can be rather high as the discriminator has only just started being trained. The values are then assessed and identified. The error is calculated and these are backpropagated through the discriminator, where the weights are updated.

![Step 1 - Discriminator](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/step1-discriminator.png)

Next we train the generator. We take the batch of images that it created and put them through the discriminator again. 
We do not include the feature images. The generator learns by tricking the discriminator into it outputting false positives.

The discriminator will provide an output of probabilities. The values are then assessed and compared to what they should have been. The error is calculated and backpropagated through the generator and the weights are updated.

![Step 1 - Generator](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/cv/step1-generator.png)

Step 2 - This is the same as step 1 but the generator and discriminator are trained a little more. Through backpropagation the generator understands its mistakes and starts to make them more like the feature.

This is created through a Deconvolutional Neural Network.

GANs can be used for the following:
* Generating Images
* Image Modification
* Super Resolution
* Assisting Artists
* Photo-Realistic Images
* Speech Generation
* Face Ageing

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/computer_vision/2.%20generative_adversarial_networks.py) for an example of a Deep Convolutional GANs.

```python
# Defining the generator
class G(nn.Module):
    def __init__(self):
        # Used to inherit the torch.nn Module
        super(G, self).__init__()
        # Meta Module - consists of different layers of Modules
        self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
                nn.Tanh()
                )
        
    def forward(self, input):
        output = self.main(input)
        return output

# Creating the generator
netG = G()
netG.apply(weights_init)

# Defining the discriminator
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False),
                nn.Sigmoid()
                )
        
    def forward(self, input):
        output = self.main(input)
        # .view(-1) = Flattens the output into 1D instead of 2D
        return output.view(-1)

# Creating the discriminator
netD = D()
netD.apply(weights_init)
```