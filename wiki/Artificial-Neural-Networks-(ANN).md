![ANN Basic Format](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/ann.png)

## Table of Contents
* [The Artificial Neural Networks](#the-artificial-neural-networks)
* [The Activation Function](#the-activation-function)
* [How an ANN Works](#how-an-ann-works)
* [How an ANN Learns](#how-an-ann-learns)
* [Gradient Descent](#gradient-descent)
* [Backpropagation](#backpropagation)
* [Training an ANN with Stochastic Gradient Descent](#training-an-ann-with-stochastic-gradient-descent)

## The Artificial Neural Networks

A human neuron is the basic building block of an Artificial Neural Networks. A neuron on it's own isn't that strong but with multiple of them, they work together to make some magic. The dendrites connect other neurons through the axon which sends an electrical impulse to another neuron it is connect to.

![Neuron](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/neuron.png)

Dendrites do not touch other neurons (this is known as Synapse), they use neurotransmitter molecules to provide other neurons with information through receptors.

In a machine, a neuron works looks like this:

![ANN Neuron](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/ann-neuron.png)

Inputs are the independent variables that must be either:
* Standardize - they have a mean of 0 and a variance of 1.
* Normalize - you take the minimum value and divide by the range of your values to get values between 0 and 1.

Some output value examples include:
* Continuous (price)
* Binary (will exit yes/no)
* Categorical - if categorical, your value will have more than one exit value

All synapses get assigned weights. Weights are how neural networks learn, adjusting the weights tell the neuron which signal is important and which isn't. It also effects whether the signal gets passed onto the neuron or not.

Here is what happens inside the neuron:
* Step 1 - It takes the weighted sum of all the values it is receiving. Adds them up and multiples them by the weight.
* Step 2 - It applies an activation function.
* Step 3 - The neuron decides whether to pass on the value to the output or not.

![Neuron Steps](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/neuron-steps.png)

## The Activation Function
This is a node that can be added to the output of your neural network. This is also known as a transfer function and can be attached in between two neural networks.

There are 4 common activation functions:
* Threshold function - if the value is less than 0, the function passes on 0. If the value is more than 0, the function 
  passes on 1.
* Sigmoid function - based on probability. Closer to 0 makes it a 0, closer to 1 makes it a 1.
* Rectifier function - goes from 0 and is 0. From there it gradually increases as the input value increases. This is the 
  most popular activation function in ANNs.
* Hyperbolic Tangent (tanh) function - like the Sigmoid function but goes below 0.

![Activation Functions](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/activation-functions.png)

Assuming the dependent variable is binary (y = 0 or 1), you would use either the Threshold function or the Sigmoid function. If you have a neural network within multiple neurons, you use a Rectifier function and then apply a Sigmoid function to then pass on the output.

![ANN Activation Functions](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/ann-activation-functions.png)

## How an ANN works
Using an example of identifying the price of a property. In the diagram the Bedrooms and Age are both 0 weights as they are not important values for this neuron. Area and Distance to City are important for this neuron, the neuron will only activate when the criteria it has been trained on is met. Each neuron consists of different requirements based on what input layer is synapsed to it.

![How ANNs work](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/how-anns-work.png)

## How an ANN Learns
With neural networks you provide input values and the output values, from there the network will figure out the rest itself.

![How ANNs Learn](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/ann-learn-diagram.png)

The diagram shows the output value as a predicted value and the actual value. We calculate the cost function that tells us what the error is in our prediction. Our goal is to minimise the cost function, the lower it is the closer the output value is to the actual value.

Once we have compared the output value and actual value we push this value back into the neural network. It then goes to the weights and updates them, this is called backpropagation.

![ANN Backpropagation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/ann-backpropagation-diagram.png)

## Gradient Descent
This relates to how we find the optimal weight value for the cost function. In simplest terms, we choose a weight and then determine which way it is decreasing, we then follow the decrease. If it starts to increase again, we back track and decrease the value again to find the lowest possible value.

For this to work, our cost function must be convex.

![Gradient Descent - Convex](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/convex.png)

If our cost function is not convex we would use Stochastic Gradient Descent. How this works is, we take each row of our dataset and run it through our neural network then adjust the weights for that row.

![Gradient Descent](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/gradient-descent.png)

Stochastic Gradient Descent helps you avoid the problem where you find the local minimums rather than the overall global minimum. This is because it has much higher fluctuations due to it calculating 1 row at a time, helping it find the global minimum rather than the local minimum. Stochastic Gradient Descent is faster than Batch Gradient Descent.

## Backpropagation
Backpropagation works in synergy with Forward Propagation. Forward Propagation is where information is entered into the input layer and then propagated forward to get our predicted output and compare those to the actual values.

Comparing the predicted output and actual value provides us with errors. We backpropagate those errors back through the neural network to train the network and adjust the weights.

Backpropagation allows you to keep track of what weight is assigned to what neuron and allows you to change all the weights at once.

## Training an ANN with Stochastic Gradient Descent
A step by step process of how ANNs are trained with Stochastic Gradient Descent can be found below:
* Step 1 - Randomly initialise the weights to smaller numbers close to 0 (but not 0).
* Step 2 - Input the first observation of your dataset into the input layer, each feature consists of one input node.
* Step 3 - Forward-Propagation. From left to right, the neurons are activated in a way that the impact of each neuron's 
           activation is limited by the weights. Propagate the activation's until getting the predicted result y.
* Step 4 - Compare the predicted result to the actual result. Measure the generated error.
* Step 5 - Back-Propagation. From right to left, the error is back-propagated. Update the weights according to how much 
           they are responsible for the error. The learning rate decides how much we update the weights.
* Step 6 - Repeat steps 1 to 5 and update the weights after each observation (Reinforcement Learning/Stochastic Gradient 
           Descent). Or, Repeat steps 1 to 5 but update the weights only after a batch of observations (Batch Learning).
* Step 7 - When the whole training set passed through the ANN, that makes an epoch. Repeat all steps until the cost 
           function is optimal (create more epochs).

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/deep_learning/0.%20supervised_networks/0.%20artificial_neural_networks.py) for an example of an ANN being created in Keras.

```python
#---------------------------------------------------------
# Part 2 - Making the Artificial Neural Networks
#---------------------------------------------------------
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer (input_dim) and the first hidden layer (units) with dropout
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer with dropout
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
```