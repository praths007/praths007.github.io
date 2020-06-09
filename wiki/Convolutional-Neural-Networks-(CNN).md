## Table of Contents
* [Convolutional Neural Networks](#convolutional-neural-networks)
* [Step 1 - Convolution](#step-1---convolution)
* [Step 2 - Rectified Linear Units (ReLU) Layer](#step-2---rectified-linear-units-relu-layer)
* [Step 3 - Pooling](#step-3---pooling)
* [Step 4 - Flattening](#step-4---flattening)
* [Step 5 - Full Connection](#step-5---full-connection)
* [SoftMax](#softmax)
* [Cross-Entropy](#cross-entropy)

## Convolutional Neural Networks
Convolutional Neural Networks take an input image that are then passed into a convolutional neural network which then provides an output label (image class).

![Convolutional Neural Networks](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/cnn.png)

For example, if you were to provide a picture of someone happy, the output label would say that they are happy. If you provided a picture of someone that is frowning, the output label would say that they are sad.

CNNs consist of two types: Black & White images and coloured images.

Using a 2x2px grid, here is an example of how they work:
* Black & White images - 2d array with each pixel having a value between 0 and 255
* Coloured images - 3d array with 3 layers (rgb). Each colour has its own intensity, each pixel has a value between 0 and 
  255, each pixel has 3 colours and the values are merged together for each one to make the appropriate colour.

![CNN Pixels](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/cnn-pixels.png)

## Step 1 - Convolution
The convolution function looks like this:

![Convolution Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/convolution-function.png)

To simplify things, we are going to run through an image that is within 1's and 0's. This is based off a black and white image so 0 being white, 1 being black.

In this example, our feature detector is 3x3 in size. Feature detectors can be larger or smaller and can also be known as a kernel or filter. The feature detector matches up the 0s and 1s to squares on the input image to then store it in a feature map. The detector runs through each column, 1 column at a time, this is known as a stride. In this example the stride is of 1px. The stride can be changed and is commonly used with 2px.

![Feature Detection](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/feature-detection.png)

Once the feature detector has finished going through the whole image, the Feature Map is completed. Here is our completed example:

![Completed Feature Detector](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/completed-detector.png)

A convolution operation is signified by an X in a circle. A Feature Map can also be known as Convolved Feature or the Activation Map. The purpose of the feature detector is the following:
* Compress the image into a feature map, making it easier to process (the more stride you have, the smaller the feature 
  map)
* Doing this loses some information but the feature detectors purpose is designed to focus on parts of the image that are 
  integral.
   * E.g. the feature detector has a certain pattern on it, the highest number in the feature map is when that pattern 
     matches up.
* It helps us to preserve the features of the image

A convolutional layer consists of multiple feature maps using different features of the image. Through the neural network training it decides which features are important.

![Convolutional Layer](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/convolutional-layer.png)

Some types of features include the following:

The feature map of each feature is on the left and the image representation is on the right.

![CNN Features](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/cnn-features.png)

## Step 2 - Rectified Linear Units (ReLU) Layer
We apply a rectifier to increase non-linearity in our CNN. The rectifier acts as the function that breaks up linearity. We want to increase non-linearity so that the images themselves are highly non-linear. 

![ReLU](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/relu.png)

In terms of linear, it refers to shading. For example: it goes from white to black in shaded steps.

![CNN Shading](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/cnn-shading.png)

## Step 3 - Pooling
Pooling uses spatial invariance on our neural network to take the relevant features from multiple images and determine that they are the same thing, no matter the angle or position of the item in the image.

![Image Rotation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/image-rotation.png)

Max Pooling scans through the choose feature map and takes the highest value out of the selected box input. For example: 2x2px, with a stride of 1px that only takes 1 value and discards the other 3.

If the box crossed over, you can continue as normal and just take the values from those inputs displayed.

![Max Pooling](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/max-pooling.png)

The purpose of this is to keep the main parts of the feature map and remove a large portion of the image. This accounts for preventing distortion and reduces the size of the feature map further for better processing speed. Doing this reduces the number of parameters, preventing overfitting on our model.

There are additional pooling methods such as:
* Mean/sub-sample pooling - this is the average number within the box
* Sum pooling - this is the summed number within the box

Pooling is also known as Downsampling.

## Step 4 - Flattening
Using the pooled feature map we flatten it into one column. Starting from the top row, left to right.

![Pooled Feature Map](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/pooled-feature-map.png)

This is done so that we can fit this into an Artificial Neural Network for further processing.

![Full Convolution](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/full-convolution.png)

## Step 5 - Full Connection
This adds an Artificial Neural Network onto the flattened input image. The hidden layers inside a Convolutional Neural Network are called Fully Connected Layers. These are a specific type of hidden layer which must be used within the CNN.

![CNN to ANN](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/cnn-to-ann.png)

Using Fully Connected Layers the output layer is passed into the input layer. This is used to combine the features into more attributes that predict the outputs (classes) more accurately.Once flattened, some features are already encoded from the vector but the ANN builds on that and improves it.

![Large ANN from CNN](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/large-ann-from-cnn.png)

Errors in CNNs are called a Loss Function which uses Cross-Entropy and mean-squared.

When the CNN is backpropagated we adjust the weights and adjust the feature map to determine if we are using the correct one. Having multiple output layers, we need to understand what weights we assign to the synapses to each output. 

Let's say we have a cat and a dog image. In this example, the answer is a dog. The neurons within the fully connected layers store one feature of each part of the animal and all neurons are connected to both the cat & dog. However, some neurons get activated to specify the distinction between the two. E.g. floppy ears, whiskers and nose.

To make the correct distinction between the images, the CNN needs a large amount of iterations to classify the difference between them.

## SoftMax
The SoftMax function is used to provide output values with a probability between each of them. For example: there is an image of a dog that has been passed through a CNN, the machine decides that the image is 0.95 likely to be a dog and 0.05 likely to be a cat.

![SoftMax Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/softmax-function.png)

The original value is squashed into much smaller values that both add up to 1 to allow the machine to provide a suitable probability of each image.

## Cross-Entropy
Cross-Entropy is a function that comes hand-in-hand with SoftMax. The original formula is:

![Original Cross-Entropy Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/original-cross-entropy-formula.png)

The function we are going to use (as it's easier to calculate) is:

![Modified Cross-Entropy Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/modified-cross-entropy-formula.png)

A Cross-Entropy function is used to calculate the Loss Function and helps to get a neural network to an optimal state. This method is the preferred method for classification. If using regression, you would use Mean Squared Error.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/deep_learning/0.%20supervised_networks/1.%20convolutional_neural_network.py) for an example of a CNN.

```python
#----------------------------------------
# Part 1 - Building the CNN
#----------------------------------------
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', data_format = 'channels_last' ))
# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2) ))

# Step 1 - Convolution (Secondary)
classifier.add(Convolution2D(32, (3, 3), activation = 'relu', data_format = 'channels_last' ))
# Step 2 - Max Pooling (Secondary)
classifier.add(MaxPooling2D(pool_size = (2, 2) ))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# Hidden Layer
classifier.add(Dense(units = 128, activation = 'relu'))
# Output Layer - uses sigmoid for 'Binary' (two values) and uses 'Softmax' for 3+ values
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```