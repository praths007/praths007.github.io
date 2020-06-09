## Table of Contents
* [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
* [The Vanishing Gradient Problem](#the-vanishing-gradient-problem)
* [Long Short-Term Memory Network (LSTM)](#long-short-term-memory-network-lstm)

## Recurrent Neural Networks (RNN)
Recurrent Neural Networks are like short-term memory within the human brain. These are represented as the Frontal Lobe part of the brain.

Artificial Neural Networks being the Temporal Lobe & Convolutional Neural Network being the Occipital Lobe.

![Brain](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/brain.png)

An RNN is an ANN that has been squashed together and had a new dimension added to it. RNN's are in a different layout to normal ANNs, they go vertical instead of horizontal. Each circle displayed is a whole layer of nodes not just one node.

The blue line in the diagram is called a temporal loop. This means that it's hidden layer gives an output and feeds back into itself. The purpose of these are to remember what was in the neuron previously and then pass this information onto itself in the future.

![Recurrent Neural Network](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/rnn.png)

Some RNN examples are as follows:
* One to many relationship - one input and multiple outputs. For example: you have 1 image and the computer describes the 
  image. It would first be fed into a CNN and then an RNN. The computer would then come up with words to describe the 
  image.

![One to Many Relationship](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/one-to-many-relationship.png)

* Many to one relationship - multiple inputs and one output. For example: you have a lot of text and you need to gauge if 
  it's a positive comment or a negative comment or how positive/negative is the comment.

![Many to One Relationship](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/many-to-one-relationship.png)

* Many to many relationship - multiple outputs and inputs. For example: using a translator, some languages depend on 
  whether the sentence you are using is gender based to then output the next set of words correctly for the sentence to 
  make sense.

![Many to Many Relationship](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/many-to-many-relationship.png)

## The Vanishing Gradient Problem
The RNN calculates the output to the desired output during the training providing you with an error value that is calculated using the cost function. 

The value of the cost function must be backpropagated through the network. This will go back depending on how many time steps you take.

![Vanishing Gradient Problem](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/vanishing-gradient-problem.png)

The Vanishing Gradient Problem is when you backpropagate through time. The issue lies with the weight recurring which is the weight used to connect the hidden layers to themselves in the unrolled temporal loop. 

No matter how many time steps you take, you need to multiple by the weight to go back a step which is where the problem arises. The more you multiple by something small, the value decreases very quickly.

![Weight Recurring](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/weight-recurring.png)

The lower the gradient of the network, the slower it is going to update the weights. If we were to train the original neurons on 1,000 epochs, through each pass back in time those neurons would take longer to train meaning your whole network wouldn't be trained properly, providing inaccurate results. If your weight recurring is too high, this is called an exploding gradient problem.

We can resolve these problems by doing the following:
* Exploding Gradient Problem
   * Truncated Backpropagation - you stop backpropagating after a set point.
   * Penalties - the gradient can be artificial reduced.
   * Gradient Clipping - maximum limit of the gradient, never go over this value. If it does, it will stay at that value.
* Vanishing Gradient Problem
   * Weight Initialization - setting your weights to minimise the likely of vanishing gradient.
   * Echo State Networks
   * Long Short-Term Memory Networks (LSTMs)

## Long Short-Term Memory Network (LSTM)
Long Short-Term Memory Networks make the Wrec from all neural networks at each time step add up to 1.

An amazing article from Colah for more information can be found here:
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

![LSTM](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/lstm.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/deep_learning/0.%20supervised_networks/2.%20recurrent_neural_network.py) for an example of a RNN.

```python
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
"""
50 neurons in our layer, return sequences is used when having additional layers.
Input shape only needs the timesteps and input_dim as the batch_size is taken into account automatically.
"""
regressor.add(LSTM( units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) ))
# Ignore 20% of the neurons
regressor.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```