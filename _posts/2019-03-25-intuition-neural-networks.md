---
layout: post
mathjax: true
comments: true
title: "Neural network using gradient descent"
excerpt: "Implementation of neural net layers, activation function and backpropagation using R"
date: 2019-03-25
---

I tried to implement a 2 layer (1 input and 1 hidden layer) neural network in R using gradient descent optimization. The
steps followed for execution are similar to logistic regression only difference being the use of **backpropagation** 
which
is used to adjust the weights in each layer of the network based on the rate of change of cost in the output layer.
My primary objective for this post is to understand this backpropagation. Complete codes for this exercise can be found 
[here](https://github.com/praths007/machine_learning_intuition).

#### Neural Nets - An Intro
The most basic type of neural net we use in machine learning mimics the neurons of a human brain. A neural network can 
be thought of layers of interconnected nodes. Every node in this net has a **weight** and **bias**. They act as a black
box which takes some input and bias and based on its inherent weights and activation gives some output. 
These weights can be adjusted to tweak the output. One way of adjusting these weights is called backpropagation. 

#### Analogy with logic gates
Logic gates in computer science also tend to be analogous to neural networks in some aspects.

![and_gate_network](/assets/neural_net_1_and_gate.png){:height="300px" width="600px"}{: .center-image }

Here -30 is the bias and +20 and +20 are the weights. The activation used is a sigmoid function. This structure
implements an **AND** gate. If I want to convert this into an **OR** gate I need to increase the value of +20, +20 to be
+40, +40 (more than \\( \lvert 30 \rvert \\)). So one can change the weights and biases to obtain different outputs.
 
#### Steps for execution
##### 1. Load dataset  
Similar to logistic regression I have used the famous iris dataset with 2 features `Petal.Length` and `Petal.Width`. 
My dependent variable will be
`Species`. I have also added an X intercept which will be 1. The data is split into train and test with a 70:30 ratio.

```python
iris_data = iris[which(iris$Species %in% c("setosa", "versicolor")),]
iris_data$Species = ifelse(iris_data$Species == "setosa", 1, 0)

### adding X intercept to the data
iris_data$intercept = 1

iris_data = cbind(intercept = iris_data$intercept, 
                  iris_data[,c("Petal.Length", "Petal.Width",  "Species")])
                  
index = unlist(createDataPartition(iris_data$Species,p =  0.7))

train_x = as.matrix(iris_data[index,c("intercept", "Petal.Length", "Petal.Width")])
train_y = as.matrix(iris_data[index,c("Species")])

test_x = as.matrix(iris_data[-index, c("intercept", "Petal.Length", "Petal.Width")])
test_y = as.matrix(iris_data[-index, c("Species")])
```
##### 2. Initialize \\( \theta \\) parameters  
\\( \theta \\) is the [gradient](/2019/03/15/machine-learning-intuition-gradient-descent/). Start by assigning random
values to this vector. Its values will change with each iteration of gradient descent.

```python
init_theta = as.matrix(c(0, 0.5, 0.5), 1)
```
##### 3. Multiply independent variables x with \\( \theta \\) (weights) and apply activation function
Here I multiple the weights/gradient with the independent variables and apply an 
[activation function](https://en.wikipedia.org/wiki/Activation_function). The activation
in my case is [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) as I want to limit my output probability between
 0 and 1. This will give me my \\( h_\theta(x) \\) which is the function/model that maps \\( x \\) 
 (independent variable) to \\( y \\) (dependent variable). 

```python
z1 = train_x %*% c(t(init_theta))

## activation function - used to limit values between 0 and 1
sigmoid = function(x){
  return(1 / (1 + 2.71^-x))}

hthetax = mapply(sigmoid, z1)
```
##### 4. Calculate cost
Here I calculate difference between my predicted \\( h_\theta(x) \\) and actual value \\( y \\). This is calculated using a 
[cost function](/2019/03/15/machine-learning-intuition-gradient-descent/). In my case the cost function is logarithmic
loss or [logloss](https://datawookie.netlify.com/blog/2015/12/making-sense-of-logarithmic-loss).

```python
logloss_cost = function(y, yhat){
  return(-(1 / nrow(y)) * sum((y*log(yhat) + (1 - y)*log(1 - yhat))))}
 
logloss_cost(train_y, hthetax)
```
##### 5. Calculate gradient
My objective is to minimize this cost so I will use the partial derivative of my cost function to calculate by how much
I need to change my \\( \theta \\) so that I am able to reduce the maximum cost. Here \\( \alpha \\) will be my step or
learning rate by which I want to update my \\( \theta \\).

```python
## gradient calculation
alpha = 0.01
theta_iter = alpha/nrow(train_x) * (t(train_x) %*% c(hthetax - train_y))
``` 
##### 6. Update \\( \theta \\)
Then I will update theta according to my partial derivative.
```python
## updating gradient
init_theta = init_theta - theta_iter
init_theta
```
##### Gradient descent
The repetition of steps 1 to 6 to minimize cost is termed as gradient descent. The following section shows the code for
gradient descent:

```python
cost = c()
gradient_descent_for_logloss = function(alpha, iterations, initial_theta, train_x, train_y)
{
  ## initialize theta (weights)
  theta = initial_theta
  for(i in seq(1, iterations)){

  z1 = train_x %*% c(t(theta))
  
  ## apply activation
  hthetax = mapply(sigmoid, rowSums(z1))
  
  ## calculating cost J(theta)
  cost <<- c(cost, logloss_cost(train_y, hthetax))
  
  ## gradient calculation
  ## using derivative of cost function J(theta)
  theta_iter = alpha/nrow(train_x) * (t(train_x) %*% c(hthetax - train_y))
  
  ## gradient update
  theta = theta - theta_iter
  
  slope = theta[2]/(-theta[3])
  intercept = theta[1]/(-theta[3]) 
  plot(iris_data$Petal.Width, iris_data$Petal.Length)
  abline(intercept, slope)
  Sys.sleep(0)
  
  }
  return(theta)
  
}
alpha = 0.01
## keep changing epochs, more epochs = more steps towards minimum cost
epochs = 600

thetas = gradient_descent_for_logloss(alpha, epochs, as.matrix(c(0, 0.5, 0.5), 1), train_x, train_y)

thetas

################################################

## predicting using theta values
z_op = test_x %*% c(t(thetas))

pred_prob = mapply(sigmoid, z_op)

preds = ifelse(pred_prob > 0.5, 1, 0)

confusionMatrix(as.factor(preds), as.factor(test_y))
## 96% accuracy
```
 

