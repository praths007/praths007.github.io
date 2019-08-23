---
layout: post
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