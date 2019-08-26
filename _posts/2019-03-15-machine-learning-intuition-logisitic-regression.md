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
 There are many ways to create this decision boundary. The most basic method is to fit a line through your data so as
 that there is minimum distance between the data points and your fitted line. And based on this fitted line you will be
 able to obtain your decision boundary. Eg. Consider that your fitted curve is a 
 [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function, then for any value
 where \\( h _\theta(x) \\) >= 0.5,\\( y \\) = 1 and for \\( h _\theta(x) \\) < 0.5, \\( y \\) = 0. \\( h _\theta(x) \\)
  is the hypothesis or "probability" of \\( y \\) being 1 or 0, given a value of \\( x \\). And \\( h _\theta() \\) 
  is the activation or model used to map \\( y \\) to \\( x \\), which in this case is a sigmoid function.