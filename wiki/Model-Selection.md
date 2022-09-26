Model Selection is used to improve your model performance that consists of choosing the best parameters of your machine learning models. 

When we build a machine learning model, we have 2 types of parameters:
* The parameters that the model has learned - these are the parameters that were changed and found to be optimal values when running the model.
* The parameters that we choose ourselves - for example: the kernel parameter in the Kernel SVM model. These parameters are called the hyperparameters.

As we can choose the optimal values of our hyperparameters these parameters are not learnt by our models. We need to determine another method to choose these optimal values which we can do by using a technique called Grid Search.

Before using Grid Search we need to use K-Fold Cross Validation on our models.

## Table of Contents
* [K-Fold Cross Validation](#k-fold-cross-validation)
* [Grid Search](#grid-search)

## K-Fold Cross Validation
This helps with optimizing a way to evaluate our model's performance. K-Fold Cross Validation is a technique that splits our Training set into 10 iterations. 1 iteration consists of 10 folds, we train our model on 9 folds and test our model on 1 fold. 

We run through each iteration using 1 different fold per iteration to test our model on.
<!---
![K-Fold Cross Validation](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/k-fold-cross-validation.png)
--->
We then take the average from the accuracies of the 10 iterations and compute the standard deviation.
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/7.%20model_selection/grid_search.py) for an example of K-Fold Cross Validation. To make this use the [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) class from the Scikit-Learn library.
--->
```python
# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
```

## Grid Search
This is used to improve our model's performance and helps us identify which hyperparameters to select and their optimal values when using a machine learning model.

This is done by creating a dictionary containing options of lists of hyperparameters for your model and using it with Grid Search to identify which would be the optimal values & parameters for that model. 
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/7.%20model_selection/grid_search.py) for an example of Grid Search. To make this use the [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) class from the Scikit-Learn library.
--->
```python
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{ 'C' : [1, 10, 100, 1000], 'kernel' : ['linear'] },
              { 'C' : [1, 10, 100, 1000] , 'kernel' : ['rbf'], 'gamma' : [0.5, 0.1, 0.01, 0.001, 0.0001] }]

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
```