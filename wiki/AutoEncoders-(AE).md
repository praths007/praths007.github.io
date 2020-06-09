## Table of Contents
* [AutoEncoders](#autoencoders)
* [Overcomplete Hidden Layers](#overcomplete-hidden-layers)
* [Sparse AutoEncoders](#sparse-autoencoders)
* [Denoising AutoEncoders](#denoising-autoencoders)
* [Contractive AutoEncoders](#contractive-autoencoders)
* [Stacked AutoEncoders](#stacked-autoencoders)
* [Deep AutoEncoders](#deep-autoencoders)

## AutoEncoders
The AutoEncoder encodes itself. This means the input nodes travel through the hidden nodes and the output of those hidden nodes are aimed to be replicated as the input nodes.

![AutoEncoder Layout](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/autoencoder-layout.png)

These can be used for things such as: feature detection, powerful recommender systems and encoding. Here are some simple examples of how they work:

![AutoEncoder Examples](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/autoencoder-examples.png)

Where there is a 1 from the input, there will be on a 1 on on the output. Sometimes AutoEncoders may include a bias, the bias is a constant added to the equation. AutoEncoders with a bias may look like this:

![AutoEncoder with Bias](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/autoencoder-bias.png)

Steps on training an AutoEncoder:
* Step 1 - We start with an array where the lines (observations) correspond to the users and the columns (the features) 
           correspond to the movies. Each cell (u, i) contains the rating (from 1 to 5, 0 if no rating) of the movie *i* 
           by the user *u*.
* Step 2 - The first user goes into the network. The input vector *x* = (r1, r2, ..., rm) contains all ratings for all 
           movies.
* Step 3 - The input vector *x* is encoded into a vector *z* of lower dimensions by a mapping function *f* (e.g. sigmoid 
           function):
   * Z = f(Wx + b) where W is the vector of input weights and b the bias.
* Step 4 - *z* is then decoded into the output vector *y* of same dimensions as *x*, aiming to replicate the input vector 
           *x*.
* Step 5 - The reconstruction error d(x, y) = ||x-y|| is computed. The goal is to minimize it.
* Step 6 - Backpropagation. From right to left, the error is backpropagated. The weights are updated according to how much 
           they are responsible for the error and the learning rate decides how much we update the weights.
* Step 7 - Repeat steps 1 to 6 and update the weights after each observation (Reinforcement Learning). Or repeat steps 1 
           to 6 but update the weights only after a batch of observations (Batch Learning).
* Step 8 - When the whole training set has passed through the ANN, this classes as an epoch. Repeat more epochs.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/deep_learning/1.%20unsupervised_networks/2.%20autoencoders.py) for an example of an AutoEncoder.

```python
# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        # First hidden layer (full connection) - encoding
        self.fc1 = nn.Linear(nb_movies, 20)
        # Second hidden layer (full connection) - encoding
        self.fc2 = nn.Linear(20, 10)
        # Third hidden layer (full connection) - decoding
        self.fc3 = nn.Linear(10, 20)
        # Fourth hidden layer (full connection) - decoding
        self.fc4 = nn.Linear(20, nb_movies)
        # Activation function
        self.activation = nn.Sigmoid()
    
    # Forward
    def forward(self, x):
        # Returns first encoded vector from first layer
        x = self.activation(self.fc1(x))
        # Returns second encoded vector from second layer
        x = self.activation(self.fc2(x))
        # Returns third encoded vector from third layer
        x = self.activation(self.fc3(x))
        # Returns fourth encoded vector from fourth layer
        x = self.fc4(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        # Creates a batch of a single input vector
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        # Checks to see if there is reviews in here that are greater than 0
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))
```

## Overcomplete Hidden Layer
Overcomplete Hidden Layers are a concept in which AutoEncoders have a hidden layer that is equal to the amount of inputs nodes or greater, this allows the AutoEncoder to have more features. When you have an equal amount or greater amount of hidden nodes in your AutoEncoders it can cheat. This means they will only pass through the hidden nodes that are parallel with the input nodes, ignoring any additional hidden nodes.

When trained up this can be a huge problem and will make it useless. We can resolve this by using any of the following types of AutoEncoders:
* Sparse AutoEncoders
* Denoising AutoEncoders
* Contractive AutoEncoders

## Sparse AutoEncoders
Sparse AutoEncoders are an AutoEncoder that has a hidden layer with a greater number than the input nodes but a regularization technique (this introduces sparsity) has been applied. 

A regularization technique is a technique that helps with preventing overfitting or stabilizes the algorithm. It adds a constraint on the loss function to prevent the AutoEncoders from using all its hidden layers at once.

![Sparse AutoEncoder](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/sparse-autoencoders.png)

## Denoising AutoEncoders
Denoising AutoEnocoders replace the input values with a modified version of the input values. We then randomly make some of those input values 0s, this will change with each epoch pass. 

Once this has been passed through the AutoEncoder you compare the results with the original input value. This type of AutoEncoder is a stochastic encoder, this makes it depend on the random selection of the input values.

![Denoising AutoEncoder](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/denoising-autoencoders.png)

## Contractive AutoEncoders
Contractive AutoEncoders add a penalty to the backpropagation of the AutoEncoder. Preventing the AutoEncoder from cheating. 

Further information can be found [here](http://www.icml-2011.org/papers/455_icmlpaper.pdf)

## Stacked AutoEncoders
Stacked AutoEncoders is when an AutoEncoder has an additional hidden layer.

Further information can be found [here](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf )

![Stacked AutoEncoder](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/stacked-autoencoders.png)

## Deep AutoEncoders
Deep AutoEncoders are RBMs that are stacked and pre-trained layer by layer. They are then unrolled and fine-tuned with backpropagation.

Further information can be found [here](https://www.cs.toronto.edu/~hinton/science.pdf)

![Deep AutoEncoder](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/dl/deep-autoencoders.png)