---
layout: post
mathjax: true
comments: true
title: "Logistic regression using gradient descent"
excerpt: "Implementing gradient descent in R for logit regression"
date: 2019-03-15
---

This is an implementation of the logistic regression assignment from Andrew Ng's machine learning class. Since the 
original implementation of these assignments is in Octave, and because I am more comfortable with R I tried 
implementing the same using R. Complete codes for this exercise can be found 
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

 ![logit_reg_dec_boundary](/assets/logit_reg_2_regression_boundaries.png){:height="300px" width="500px"}{: .center-image }
 
 



