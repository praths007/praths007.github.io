## Table of Contents
* [Boltzmann Machine](#boltzmann-machine)
* [Energy-Based Models](#energy-based-models)
* [Restricted Boltzmann Machines (RBM)](#restricted-boltzmann-machines-rbm)
* [Contrastive Divergence](#contrastive-divergence)
* [Deep Belief Networks (DBN)](#deep-belief-networks-dbn)
* [Deep Boltzmann Machines (DBM)](#deep-boltzmann-machines-dbm)

## Boltzmann Machine
Boltzmann Machines are undirected models. They have no output layer, no specific layout and are bidirectional.

![Boltzmann Machine](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/boltzmann-machine.png)

All nodes of Boltzmann Machine generate data even if they are an input mode or not. This represents a system that uses multiple factors to make something function. Visible nodes are nodes that we can and do measure, hidden nodes are nodes that we cannot or do not measure.

To the Boltzmann Machine, all nodes are equally important and are the same. The Boltzmann Machine doesn't need any inputs, it can generate the parameters itself and generates different states of our system. For example: the temperature of our system is at 10 degrees, the machine will then go 'Ok, what if the temperature is at 12 degrees?' and generates that state.

We provide the Boltzmann Machine with a dataset to train it on and it identifies all possible connections of each parameter/column of this dataset to then adjust the weights of each node accordingly. This can then be used to monitor our system and provide us with information on what is abnormal behaviour to what it has been trained on.

## Energy-Based Models
The Boltzmann Distribution is an Energy-Based Model. The equation for Boltzmann Distribution looks like this:

![Boltzmann Distribution Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/boltzmann-distribution-equation.png)

The weights dictate how the model performs. Energy is defined in these machines through the weights of the synapses. Once the system is trained up and the weights are set, based on those weights the system will try and find the lowest energy state for itself possible.

The Energy Function equation for a Restricted Boltzmann Machine is as follows:

![Energy Function RBM Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/rbm-energy-equation.png)

## Restricted Boltzmann Machines (RBM)
Restricted Boltzmann Machines are the same as Boltzmann Machines except it has one restriction, hidden nodes cannot connect to each other and visible nodes cannot connect to each other.

![RBM Layout](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/rbm-layout.png)

Through the training process, the RBM learns how to allocate its hidden nodes to specific features. However, the RBM doesn't know what the nodes are in terms of naming, it just knows that one or some hidden nodes (features) have correlations to one or some of the visible nodes.

![RBM Genre & Movies](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/rbm-genre.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/deep_learning/1.%20unsupervised_networks/1.%20boltzmann_machines.py) for an example of a Restricted Boltzmann Machine.

```python
# Creating the architecture of the Neural Network
class RBM():
    # nv = visible nodes
    # nh = hidden nodes
    def __init__(self, nv, nh):
        # Initialize the weights - this consists of a matrix with the size of the hidden nodes and visible nodes
        self.W = torch.randn(nh, nv)
        # Initialize the bias and add a 2nd Dimension
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    
    ## Sample the hidden nodes
    def sample_h(self, x):
        # Define product of the weights
        # .t = transpose which is used to make the equation mathematically correct
        wx = torch.mm(x, self.W.t())
        # expand_as = make the activation function the same Dimension for each mini-batch
        activation = wx + self.a.expand_as(wx)
        # Probability value given the visible nodes
        # Given the value of the visible nodes we return the probability of each of the hidden nodes = 1
        p_h_given_v = torch.sigmoid(activation)
        # Based on the probability, activate the hidden node
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    ## Sample the visible nodes
    def sample_v(self, y):
        # Define product of the weights
        wy = torch.mm(y, self.W)
        # expand_as = make the activation function the same Dimension for each mini-batch
        activation = wy + self.b.expand_as(wy)
        # Probability value given the hidden nodes
        # Given the value of the hidden nodes we return the probability of each of the visible nodes = 1
        p_v_given_h = torch.sigmoid(activation)
        # Based on the probability, predict whether the user will like the movie or not
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    ## Contrastive Divergence
    # v0 = Input vector, e.g. ratings of all the movies by one user
    # vk = Visible nodes after k sampling
    # ph0 = Vector of probabilities, at first iteration the hidden nodes = 1 given the values of v0
    # phk = Probabilities of the hidden nodes after k sampling
    def train(self, v0, vk, ph0, phk):
        # Updating the weights
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        # The sum is used to keep the same dimensions of the bias
        self.b += torch.sum( (v0 - vk), 0 )
        self.a += torch.sum( (ph0 - phk), 0 )

nv = len(training_set[0])
nh = 100
batch_size = 100

rbm = RBM(nv, nh)
```

## Contrastive Divergence
Contrastive Divergence allows Restricted Boltzmann Machines to learn. The visible nodes are used to construct the hidden nodes and then those hidden nodes are used to construct the next set of visible nodes. The diagram below shows how this works.

![Contrastive Divergence](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/contrastive-divergence.png)

This process is called Gibbs Sampling. This is how it works:
* Within the first pass, the visible nodes are reconstructed by the hidden nodes. Those hidden nodes then try to recreate 
  the values of the visible nodes.
* Next, we feed the reconstructed visible nodes back into the hidden nodes which will reconstruct the visible nodes again 
  providing you with a new output.
* Again, those reconstructed visible nodes would be put back into the hidden nodes to be reconstructed again providing you 
  with another new out and so on.
* In the end, when a set of visible nodes have been input into the hidden nodes, the small value will be returned once it 
  has been reconstructed by the hidden nodes.

There is a method used with Contrastive Divergence that allows us to speed up the Gibbs Sampling process. We use a formula that takes the first two passes and identifies the weight of the energy of the RBM, providing us with the lowest energy value in the quickest amount of time.

![Gibbs Sampling Equation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/gibbs-sampling.png)

In the below graphs, green is the first pass, red is the second pass. Firstly, we identify which direction the energy is going. We then adjust the weights so that the starting point is at the lowest possible energy value. 

![Contrastive Divergence Graphs](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/contrastive-divergence-graphs.png)

## Deep Belief Networks (DBN)
Deep Belief Networks are when you stack several RBMs on top of each other.

![Deep Belief Networks](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/dbn.png)

Diagram explanation:
* In this diagram, there are a total of 3 RBMs.
* The green box contains the 1st RBM.
* The yellow box contains the 2nd RBM.
   * Even though the diagram shows the bottom layer in this box as a row of hidden nodes, these are in fact the visible 
     nodes for that RBM connected.
   * This bottom layer is the hidden nodes row for the 1st RBM and the visible nodes layer for the 2nd RBM.
* The purple box contains the 3rd RBM.
   * The bottom layer in this box is the hidden nodes row for the 2nd RBM and the visible nodes layer for the 3rd RBM.

The directionality is in the place for the 1st and 2nd RBM and they are directed downwards.

When training a DBN it uses two types of algorithms:
* Greedy Layer Wise Training - this trains the RBMs up separately, layer by layer, while they are not connected. It then 
  puts them together and adds in the directionality.
* Wake-sleep Algorithm - this is when you train all the RBMs on the DBN while they are connected, going all the way up through the RBMs and all the way back down. The ones going up are awake and the ones going down are asleep.

## Deep Boltzmann Machines (DBM)
Deep Boltzmann Machines are similar to Deep Belief Networks. They have the same structure as them but the main architectural difference is that they have no directionality. Their main purpose is to be used for outputting more advanced features.