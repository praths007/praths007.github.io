---
layout: post
mathjax: true
comments: true
title: "Logistic Regression - Intuition (in R)"
excerpt: "Understanding logistic regression using gradient descent"
date: 2019-03-15
---

Since the past few weeks I have been taking the certificate course on machine learning taught by Andrew Ng. Even
though I had taken the free version of this course way back during college, some aspects of it eluded me at that time. 
This was because the scripts used in this course are written to work with octave and since the language was not part 
of my curriculum I skipped some of the practical assignments and simply skimmed through the theory.

Recently, I thought it would be a good exercise to go through the course once again and try to understand all its 
nuances. One way of doing this was to implement all the assignments in R without using any library. Following is my
attempt at implementation of logistic regression using gradient descent. This also includes the intuition behind
gradient descent and cost function. Complete codes for this exercise can be found 
[here](https://github.com/praths007/machine_learning_intuition).

#### Logisitic Regression - An Intro

Logistic regression is one of the basic classification algorithms used in classical machine learning. Where we try to
separate 2 different types of elements using a **decision boundary** in an n-dimensional space. The decision boundary is
 created in such away that it should completely separate the elements and no element of one type should be present
 with the element of another type (eg. the organization of apples and oranges in fruit basket is separated from one
 another using a cardboard strip. This cardboard strip acts as a decision boundary).
 
 ![apples_oranges_decision_boundary](/assets/logit_reg_1_apples_oranges.jpg){:height="200px" width="300px"}{: .center-image }
 
#### Decision Boundary
 There are many ways to create this decision boundary. The most basic method is to fit a line through my data so
 that there is minimum distance between the data points and my fitted line. And based on this fitted line I will be
 able to obtain my decision boundary. Eg. Consider that my fitted curve is a 
 [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function, then for any value
 where \\( h _\theta(x) \\) >= 0.5, \\( y \\) = 1 and for \\( h _\theta(x) \\) < 0.5, \\( y \\) = 0. Given a value of
  \\( x \\), \\( h _\theta(x) \\) is the hypothesis or "probability" of \\( y \\) being 1 or 0. And \\( h _\theta() \\) 
  is the activation or model used to map \\( y \\) to \\( x \\), which in this case is a sigmoid function.
  
#### Gradient Descent - intuition
##### What exactly is a Gradient?
Recalling the equation of a straight line:  

$$ 
y = mx + c
$$
 
Rewriting this in a slightly different form (used by Prof. Ng in his lectures) we get:  

$$
h _\theta(x) = \theta_0 + \theta_1(x)
$$
 
Here \\( \theta \\) is the gradient analogous to slope m in the equation of a line. If I recall, the slope of a line in 
a 2D plane it is calculated as \\( \frac{y_2 - y_1}{x_2 - x_1} \\) 
**i.e. by how much will y change if x is changed by a certain amount?**

Now I want to fit a line through my data in such a way that it passes through majority of the points. So that in the
future if I want to **predict** the position of an unknown point I can just use the line I used to fit my data. What I 
want is that the line that I am trying to fit exactly traces my data points so that the distance 
(error in machine learning culture) between my fitted line and data points is minimum. 
To do this I need to understand that by how much (magnitude) and in what direction I need to change the slope of my line
 so that it fits majority of my data points.

##### What is a cost function?

This is where the cost function comes into picture. Cost is simply the distance (error) between my data points and 
fitted line and ideally this must be as close to 0 as possible. By taking its partial derivative I calculate how much 
 (magnitude) cost will change if I change \\( \\theta \\) by a certain amount (this is the same logic which is used for
calculating slope of a line, i.e. magnitude of change in y with respect to x. And also a general idea of **calculus** 
which deals with the mathematics of change of one quantity with respect to another.) 


Getting an idea of the cost function and gradient I move on to the **descent** in gradient descent.

##### What is descent in gradient descent?






In case of logistic regression the cost function is logarithmic loss or 
[logloss](https://datawookie.netlify.com/blog/2015/12/making-sense-of-logarithmic-loss/).


Applying the same logic, \\( \theta \\) is a vector which has the partial derivatives of a cost function, 
which is nothing but the rate of change of \\( y \\) (dependent variable) with the rate of change of \\( x \\) 
(independent variable) for each independent variable. So if I have 2 features/independent variables I will have 2 
values in \\( \theta \\).
 
 
 



