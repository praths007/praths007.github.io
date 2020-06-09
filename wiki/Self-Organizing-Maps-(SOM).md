Self-Organizing Maps for reducing dimensionality. The purpose is to reduce the number of columns you have in a dataset and represent it onto a 2D plain.

![Self-Organizing Maps](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/som.png)

The map above represents the different states of poverty in different countries. It has put them into clusters based on 39 different indicators (columns of data). 

![Self-Organizing Map Node](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/som-node.png)

A Self-Organizing Map is an unsupervised learning algorithm, we feed it data and it will learn the correlations of the data itself. Based on that data it will categorise them using K-Means Clustering stating which ones are similar. SOMs are slightly different to neural networks, the weights on the synapses are characteristics of the node itself, they are coordinators for that node.

![Self-Organizing Map Network](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/som-network.png)

In the example we have 3 input nodes, if we had 20 the nodes would have 20 characteristics/weights. The SOM determines which nodes are closest to each row of data within the dataset using Euclidean Distance. It uses Best Matching Unit (BMU) to determine which node is the closest to the rows data.

When the SOM has determine what row the node belongs to, the node updates the nodes in a radius around it to move them closer to that node. Witch each epoch the radius around the nodes shrink, meaning less nodes are pulled towards it making your data more accurate. 

Key points to be aware of:
* SOMs retain topology of the input set.
* SOMs reveal correlations that are not easily identified.
* SOMs classify data without supervision.
* They have no target vector which means no backpropagation.
* They have no lateral connections between output nodes.

The process of Self-Organizing Maps:
* Step 1 - We start with a dataset composed of *n_features* independent variables.
* Step 2 - We create a grid composed of nodes, each one having a weight vector of *n_features* elements.
* Step 3 - Randomly initialize the values of the weight vectors to small numbers close to 0 (but not 0).
* Step 4 - Select one random observation point from the dataset.
* Step 5 - Compute the Euclidean Distances from this point to the different neurons in the network.
* Step 6 - Select the neuron that has the minimum distance to the point. This neuron is called the winning node.
* Step 7 - Update the weights of the winning node to move it closer to the point.
* Step 8 - Using a Gaussian Neighbourhood function on the mean of the winning node and update the weights of the winning 
           node neighbours to move them closer to the point. The neighbourhood radius is the sigma in the Gaussian 
           function.
* Step 9 - Repeat steps 1 to 5 and update the weights after each observation (Reinforcement Learning) or after a batch of 
           observations (Batch Learning), until the network converges to a point where the neighbourhood stops decreasing.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/deep_learning/1.%20unsupervised_networks/0.%20self_organizing_maps.py) for an example of a Self-Organizing Map.

```python
# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)
```