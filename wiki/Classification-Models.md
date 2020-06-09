## What is the difference between Regression and Classification?
Regression is used to predict data that can be measured (continuous data).

Classification is used to predict which data should be categorised together (discrete data).

## Table of Contents
* [Logistic Regression](#logistic-regression)
* [K-Nearest Neighbours (K-NN)](#k-nearest-neighbours-k-nn)
* [Support Vector Machine (SVM)](#support-vector-machine-svm)
* [Kernel SVM](#kernel-svm)
* [Naive Bayes](#naive-bayes)
* [Decision Tree Classification](#decision-tree-classification)
* [Random Forest Classification](#random-forest-classification)
* [Evaluating Classification Models Performance](#evaluating-classification-models-performance)

## Logistic Regression
This model is used for binary classification problems when the dependent variable consists of 2 categories (E.g. Yes and No). In simple terms, it combines a Single Linear Regression & a Sigmoid Function together.

Blue is the Single Linear Regression, Green is the Logistic Regression. 

![Logistic Regression](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/logistic-regression.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/2.%20classification/0.%20logistic_regression.py) for an example of a logistic regression. To make this use the [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) class from the Scikit-Learn library.

```python
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
```

## K-Nearest Neighbours (K-NN)
This is used to assign new data points to the correct category of data.

You can read more about this model [here](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/) in an amazing blog by Kevin Zakka.

![K-Nearest Neighbours](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/k-nearest-neighbours.png)

This is how it works:
* Step 1 - Choose the number K of neighbours (Common number for this is 5).
* Step 2 - Take the K nearest neighbours of the new data point, respective to the Euclidean distance.
* Step 3 - Among these K neighbours, count the number of data points in each category.
* Step 4 - Assign the new data point to the category where you counted the most neighbours.
* FIN - Your model is ready.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/2.%20classification/1.%20k_nearest_neighbors.py) for an example of a K-Nearest Neighbours model. To make this use the [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) class from the Scikit-Learn library.

```python
# Fitting Classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
```

## Support Vector Machine (SVM)
This uses a Maximum Margin Hyperplane to separate 2 categories of data. This hyperplane has an equal space between support vectors (points closest to the hyperplane) to separate out the categories. The model looks at data that is 'out of place' and not as common so that it can determine it's supporting vectors.

![Support Vector Machine](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/svm.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/2.%20classification/2.%20support_vector_machine.py) for an example of an SVM. To make this use the [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) class from the Scikit-Learn library.

```python
# Fitting SVM to the Training set
from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
```

## Kernel SVM
This is a more advanced SVM that can be used when data is clumped together. This model uses a mapping function to turn the data into a 3-dimensional space where a separating hyperplane can be easily found.

![Kernel SVM](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/kernel-svm.png)

There are multiple kernel functions that can be used. Some examples are:
* Gaussian RBF Kernel
* Sigmoid Kernel
* Polynomial Kernel

![Kernel SVM Functions](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/kernel-svm-functions.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/2.%20classification/3.%20kernel_svm.py) for an example of a Kernel SVM. To make this use the [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) class from the Scikit-Learn library.

```python
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
```

## Naive Bayes
This uses Bays' Theorem to predict what category new data points belong to.

![Naive Bayes](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/naive-bayes.png)

Walks being a category. 

Equation broken down into parts:
* Prior Probability - Category total divided by the total observations.
* Marginal Likelihood - Create a small radius around the new data point location and calculate the number of similar 
   observations divided by the total observations.
* Likelihood - The total number of similar observations for the category divided by the total number of that category.
* Posterior Probability - Total probability of each category.

How it works: 
* Step 1 – Calculate the first category using the Bayes' Theorem equation.
* Step 2 – Calculate the second category using the Bayes' Theorem equation. 
* Step 3 – Determine which probability out of category A or category B is higher.
* Step 4 – New data point gets converted into winning category. 

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/2.%20classification/4.%20naive_bayes.py) for an example of a Naive Bayes model. To make this use the [GaussianNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) class from the Scikit-Learn library.

```python
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

## Decision Tree Classification

![Decision Tree Graph](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/decision-tree-graph.png)

These are used when the response variable categorical.

For example we have two variables: age and weight. These are used to predict if a person is going to sign up for a gym membership or not. In our training data it shows that 90% of the people who are older than 40 signed up - these are split into one category. Entropy or Gini Index can be used to measure the similarities in Classification trees.

![Decision Tree Classification](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/decision-tree-classification.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/2.%20classification/5.%20decision_tree_classification.py) for an example of a decision tree classification. To make this use the [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) class from the Scikit-Learn library.

```python
# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
```

## Random Forest Classification
This is a form of ensemble learning, it uses multiple of the same algorithm (in this case the Decision Tree Classification) and puts them together to make a more powerful model.

How it works:
* Step 1 - Pick at random K data points from the Training set.
* Step 2 - Build the Decision Tree associated to these K data points.
* Step 3 - Choose the number Ntree of trees you want to build and repeat steps 1 & 2.
* Step 4 - For a new data point, make each one of your Ntrees predict the category to which the data points belong. 
           Assign the new data point the category that wins the majority vote.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/2.%20classification/6.%20random_forest_classification.py) for an example of a random forest classification. To make this use the [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) class from the Scikit-Learn library.

```python
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
```

## Evaluating Classification Models Performance
This section consists of helpful factors that can be used to improve classification models.

### Confusion Matrix
This is used to give a quick overview of how well your model is performing. It usually consists of 4 cells in a table (sometimes more), the top left & bottom right cells are the amount of correct predictions. Calculating the values in these cells, you can easily identify if the dataset has a good accuracy rating. 

![Confusion Matrix](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/confusion-matrix.png)

### Accuracy Paradox
This is used to determine how the model is doing in terms of Precision and Recall. This is needed along with accuracy to make a suitable model. 

Precision is the number of True Positives divided by the number of True Positives and False Positives. It is also called the Positive Predictive Value (PPV).

![Sensitivity](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/sensitivity.png)

Recall is the same equation as Precision. It is also called Sensitivity or the True Positive Rate.

![Detailed Confusion Matrix](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/detailed-confusion-matrix.png)

### Cumulative Accuracy Profile (CAP) Curve
This is a method of Accuracy Paradox. It's used to compare different model results to assess which model performs better. 

Cap Curve Analysis is the process of identifying the accuracy ration of a model. This can be calculated in two ways:

1. Finding the area of the Perfect Model and the area of the Good Model then dividing them by each other. This gives a ration between 0 and 1, the closer to 1 the better the model.

![Cap Curve Analysis method 1](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/cap-curve-analysis1.png)

2. Find 50% on the horizontal axis and look where it crossed the model. Follow the crossover to the vertical axis. 

![Cap Curve Analysis method 2](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/cap-curve-analysis2.png)

How a model scales:
* If your model is < 60% = Rubbish
* If your model is between 60% -> 70% = Poor
* If your model is between 70% -> 80% = Good
* If your model is between 80% -> 90% = Very Good
* If your model is between 90% -> 100% = Too Good

If your model is 'Too Good' then double check your model. Having a too good model is known as overfitting.
