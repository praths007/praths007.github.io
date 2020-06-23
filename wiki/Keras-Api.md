## Table of Contents
* [Keras](#tensorflow)
   * [General](#general)
   * [Known issues](#known-issues)
   * [Regularization techniques](#regularization-techniques)
   * [Batch Normalization](#batch-normalization)
   * [Saving Models](#saving-models)
   * [Callbacks](#callbacks)
        * [Early Stopping](#early-stopping)
   * [Backpropagation and Gradient Descent](#backpropagation-and-gradient-descent)
   * [Regularizations](#regularizations)
   * [Bootstrapping and benchmarking hyperparameters](#bootstrapping-and-benchmarking-hyperparameters)
   * [Bulding ensembles using keras and sklearn](#bulding-ensembles-using-keras-and-sklearn)
   * [Hyperparameter tuning](#hyperparameter-tuning)
   * [Bayesion Hyperparameter Optimization](#bayesion-hyperparameter-optimization)
   * [Transfer learning for computer vision](#transfer-learning-for-computer-vision)
## Keras
### General
* Bias is the constant term in y=mx+c. Where m is the weight or Gradient and c is the constant or the bias. A bias is
needed to shift the curve from the origin. This helps in fitting data across any region.

### Known issues
* The weights are randomized before we start training a NN. So the output is going to be different and the error for
train will also be different each we run NN. Some of ways to overcome this is using bootstrap aggregating, dropouts etc.

### Batch Normalization
This [video](https://youtu.be/nUUqwaxLnWs) by Andrew NG explains this the best.
Normalizing input features to mean zero and variance 1 speeds up learning.
Batch normalization does similar thing for hidden layers.
Intuition- <br>
When training images with black cats, the network may not work well with colored cats.
This is because of the shift in the data and decision boundary wrt the origin.
The input to each hidden layer from the previous layer shifts around. (like inputs which are not normalized)
So it is better to add batch normalization layer.

It also has a slight regularization effect.
Each mini batch is scaled by mean/variance of only that mini batch. therefore this adds noise to each hidden layers
activations.
Larger mini batch reduces this noise. So regularization effect.

### Regularization techniques
* Weight sharing- as done in CNN's, applying the same filters across the image.

* Data Augmentation- Augmenting existing data and generate synthetic data with generative models

* Large amount of training data- thanks to ImageNet etc.

* Pre-training- For example say Use ImageNet learnt weights before training classifier on say Caltech dataset.

* The use of RelU's in Neural Nets by itself encourages sparsity as they allow for zero activations. In fact for more complex regions in feature space use more RelU's, deactivate them for simple regions. So basically vary model complexity based on problem complexity.

### Saving Models
Specifically in keras models can be saved in 3 formats viz. yaml, json and hdf5. hdf5 models are saved with weights,
for the others it is just string.
```python
model.save("path/to/file.yml/.h5/.json")
```
For serving models saved_model is used. The model is saved as a protobuf.
```python
from tensorflow import saved_model

saved_model.save(obj, "path/to/save", signatures, options)
```

### Callbacks
Used while model.fit(). For multiple functions such as:

#### Early Stopping
* Rather than deciding how many epochs we need to train the NN. We keep a separate validation set and train the NN such
that the val loss stops decreasing.
* Generally the training set has better loss compared to validation because the NN has those values and thus has learnt
those values.

```Python
from tensorflow.keras.callbacks import EarlyStopping
moniter = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(X, y, validation_data=(X_test, y_test), callbacks=[moniter], epochs=1000)
```

### Backpropagation and Gradient Descent
* Classic backpropagation
* Momentum backpropagation - This adds a parameter lambda which is used to push the optimizer away from the local minimum
so as to find a global minimum.
* Batch and online backpropagation - Generally every gradient is calculated for 1 row. In batching every row gradient 
vectors are added until batch size is reached. eg. After 10 elements are summed up then the change in weight is applied.
This makes things efficient.


### Things to tweak in backpropagation
* Learning rate - if too small, might get stuck in local optima
* Momentum - if too large stuff becomes erratic


#### Optimizers
* classic batch gradient descent -
Takes the whole data and runs gradient descent. It is very slow. Not like online which allows us to update
examples on-the-fly.

* Stochastic gradient descent -
The data is picked randomly and put into gradient descent. 

### Activations
ReLU
PReLU
tanh
sigmoid
softmax

#### Intuition behind using better optimizers.
A single learning rate is used for weights across the network. Maybe some neurons learn faster than others.
Sometimes learning rate can be decreased with time or put multiple learning rates. 
Move away from having single global momentum and learning rate.

Usually if a feature x is sparse (majority are 0) and input to a neuron, its weight/gradient will be 0 for most of the 
inputs. Therefore the weights for that neuron will not get enough updates, because majority of them are 0. Now if
feature x is sparse and important then the neuron representing that feature is not learning much. So we need to update 
the learning rate based on the frequency of features.

So if multiple 0's coming in > less frequency > needs higher learning rate.

if more non zeros coming in > greater frequency > needs lower learning rate.

* Adagrad - keep per weight decaying learning rate. Never increases. It is an algorithm for gradient-based 
optimization that does just this: It adapts the learning rate to the parameters, performing smaller updates
(i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates 
(i.e. high learning rates) for parameters associated with infrequent features. For this reason, it is well-suited for 
dealing with sparse data. 

Problem is the update decays to 0 very soon for ones with higher frequency. So no more updates after certain point.

* RMSprop - Solves problem of adagrad by dividing learning rate with exponentially decaying average of squared gradients.

* Adadelta - can go in either direction.

* ADAM (adaptive moment estimation)
Discovered in 2014.
Uses RMSprop denominator decay along with cumulative history of gradients.

## Kfold and stratifiedKFold cross validation
For KFold-
```python
from sklearn.model_selection import KFold

folds = KFold(5, shuffle=True, random_state=42)

for train, test in folds.split(x):
    x[train]
    y[train]
    x[test]
    y[test]
```

For StratifiedKFold-
```python
from sklearn.model_selection import StratifiedKFold
                                        # need to put random state otherwise different results
folds = StratifiedKFold(5, shuffle=True, random_state=42)

for train, test in folds.split(x, y):
    x[train]
    y[train]
    x[test]
    y[test]
```

5 fold cross validation. 1 fold validation rest are trains.
So 5 models for 5 fold. Now to bring them together ways are as follows:
* Choose model with lowest val loss. If there is major variance between the scores,then some validation fold has 
outliers.
* Present new data to all models and average out like an ensemble.
* Retrain new model by doing early stopping for each model on kfold and figure out epochs
for each and take average epochs to train the new model or maximum epochs.

  
## Regularizations
#### L1 and L2 regularizations
```python
from tensorflow.keras.layers import Dense

model.add(Dense(units, activation, activity_regularizer = regularizers.l1(1e-4)))
```

#### Dropouts
Simplifying model by removing some neurons. These neurons keep their weights but do not fire.

## Bootstrapping and benchmarking hyperparameters
Things that need tuning-
* number of layers
* number of units/neurons per layer
* activation used
* dropout percent
* L1 and L2 values for each layer
(also optimizer, class weights etc.)

Bootstrapping - <br/>
* random data picking with replacement. 
* Then the accuracy or rmse is averaged for all runs to get an idea of performance.
* Early stopping is used to understand how many epochs are needed

```python
from sklearn.model_selection import ShuffleSplit # for regression, StratifiedShuffleSplit # for classification
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, mean_squared_error
SPLITS = 50
boot = ShuffleSplit(n_splits=SPLITS, test_size=0.1, random_state=42)

moniter = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

epochs = moniter.stopped_epoch

score = np.sqrt(mean_squared_error(pred,test))

# track mean, stddev, and epochs
# slowly mean, epochs etc... will start to converge till we reach split 50
```

## CNN Layers
parameters- units/#of filters, filter, stride, padding, activation

total weights = filter size * filter size * # of filters

keras needs convnet height * weight * color dept (which is 1 if it is grayscale)

## ResNet
Skipping layers. Improves predicatability power of deeper NN.

## lr schedule code
smaller learning rates need more epochs as model trains at a slower rate.
```python
# learning rate can be reduced according to CIFAR competition after 
# 80, 120, 160, 180 epochs. the function can be used with a callback
```

## Bulding ensembles using keras and sklearn
```python
from sklearn.ensemble import RandomForestClassifier, KerasClassifier
def keras_func():
    #etc.

models = [RandomForestClassifier(n_estiamtors=10),
            kerasClassifier(keras_fn=)]

for i, model in enumerate(models):
    for train, test in k.split(X,y):
        model.fit(x_train, y_train)
        blend_test.append(pred(x_test))

# run logistic regression on blend_test and fit check accuracy with y_test
```

## Hyperparameter tuning
### Number of layers and Neuron counts
* Activation (parameter also)
* ActivityRegularization (parameter also)
* Dense
* Dropout
* Flatten
* Input (parameter also)
* Lambda - similar to python lambda function for mapping
* Reshape/permute - change shape of layers

### Activation functions
* linear - used for regression on output layer
* softmax
* tanh - used in lstms usually
* sigmoid
* hard_sigmoid - cheaper to compute
* exponential
* relu
* elu - can produce negative outputs used in GANS. exponential linear unit.

### Advanced activation functions
* LeakyReLU - prevents dead relu units. by adding some small bias.
* PReLU - learns alpha term which is also present in leaky

### Regularization
* L1, L2 activity_regularizar and kernel_regularizer
* dropout

### Batch Normalization
* can be used to combat vanishing gradient and also take care if the learning rate increases drastically and takes
nans

### Training parameters
* optimizers
* batch size - number of rows passed between for 1 complete forward and backward pass
* learning rate
* epochs - number of times the network sees the entire data
* iterations (not in keras) - 1 complete forward and backward pass

```python
def evaluate_network(dropout, lr, neuronPct, neuronShrink):
    boot = StratifiedShuffleSplit(2, test_size=0.2, random_state=42)

    mean_benchmark = []]
    epochs_needed = []
    num = 0
    neuronCount = int(neuronPct * 50000)

    for train, test in boot.split(x,y):
        X_train, X_test, y_train, y_test
        
        layer = 0
        while neuronCount > 20 and layer < 10:
            if layer==0:
                model.add(Dense(neuronCount, input_shape=X.shape[1], activation=PReLU))
            else:
                model.add(Dense(neuronCount), activation=PReLU())
            model.add(Dropout(dropout))
    
        neuronCount = neuronCount * neuronShrink
    
        model.add(Dense(1), activation=sigmoid)
        # return negative log loss
```


## Bayesion Hyperparameter Optimization
parameters are weights that backpropagation adjusts.
hyperparameters like layers etc. need to be set on our own.

Usually layers go in pyramid form. From large layers to smaller layers.

Nelder mead can also be used.
Bayesian optimization is used because it uses the past experience.
Kind of like multiarmed bandit problem.
```python
from bayes_opt import BayesianOptimzation

pbounds = {"dropout" : (0.0, 0.499),
            "lr": (0.0, 0.1),
            "neronPct": (0.01, 0.9),
            "neuronShrink": (0.1, 1)}


optimizer = BayesianOptimzation(f= evaluate_network,
pbounds = pbounds,
verbose=2)


optimizer.maximize(init_points=10, n_iter=100)

```


## LSTM
3 axes-
axis 1: training set elements (sequences) must be same size as y size
axis 2: members of the sequence (day 1, day2, day3, day4)
axis 3: features in the data (like input neurons)

```python
#converting to sequence

def to_sequence(seq_size, obs):
    x=[]
    y=[]
    for i in range(len(obs)-seq_size-1):


        window = obs[i:(i+seq_size)]
        after window = obs[i+seq_size]
        window = [[x] for x in window]
        x.append(window)
        y.append(after_window)

```


## Transfer learning for computer vision

### Use of MobileNet
It is used extensively because it is lightweight in terms of size and computations
```python
from tensorflow.keras.applications import MobileNet

model = MobileNet(weights="imagenet", include_top=True)
# if include_top is False we shear the output layer during transfer learning
```

For retraining new layers and assigning layers as trainable or non trainable
```python
base_model = MobileNet(weights="imagenet", include_top=False)
base_model.summary() # get the summary to understand how many output layers are present and which ones were removed
x=base_model.output
x.add(GlobalAveragePooling2D())
x.add(Dense())
x.add(Dense())
preds=x.add(Dense(3, activation="softmax"))


model = Model(iputs=base_model.input, outputs=preds)

for layer in model.layers[:20]: # input layers not trainable
    layer.trainable=False
for layer in model.layers[20:]: new layers that were added are trainable
    layer.trainable=True
```

Keras can also read from a directory
Create folder names with class names and put images inside themwith any name.
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.applications.resnet50 import preprocess_input

train_datagen = ImageDatagenerator(preprocessing_function=preprocess_input)
train_datagen.flow_from_directory("Users/praths/Downloads",
target_size=(128,128),
clor_mode="rgb",
batch_size=1,
class_mode="catagorical",
shuffle=True)
```

Transfer learning can also be used for getting vector embeddings for NLP and doing feature engineering.