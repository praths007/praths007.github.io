## Table of Contents
* [General](#general)
* [Scikit-Learn (Sklearn)](#scikit-learn-sklearn)
   * [General](#general-1)
   * [Data Preprocessing](#data-preprocessing)
   * [Model Selection](#model-selection)
   * [Accuracy & Predictions](#accuracy--predictions)
   * [Models](#models)
* [Keras](#keras)

## General

* import [numpy](http://www.numpy.org/) as np
* import [pandas](https://pandas.pydata.org/) as pd
* import [matplotlib.pyplot](https://matplotlib.org/) as plt

## [Scikit-Learn (Sklearn)](http://scikit-learn.org/stable/)

#### General
* .fit() - used to find the internal parameters of a model
* .transform() - used to map new or existing values to data
* .fit_transform() - does both fit and transform
* .predict() - used to make predictions

* from [xgboost](https://xgboost.readthedocs.io/en/latest/) import XGBClassifier - XGBoost gradient boosting software


#### Data Preprocessing
* from sklearn.preprocessing import ...
   * [LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) - used for categorical data
   * [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) - used for dummy variables
   * [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) - used for standardising data
   * [MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) - Used for normalising data
   * [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) - Used to replace empty spaces/missing data within a dataset


#### Model Selection
* from sklearn.model_selection import ...
   * [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) - used for splitting data into test sets and training sets
   * [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) - used for K-Fold Cross Validation
   * [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) - used for grid search (tuning models)


#### Accuracy & Predictions
* from sklearn.metrics import [confusion_matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) - used to identify the accuracy of a trained model


#### Models
* from sklearn.preprocessing import [PolynomialFeatures](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) - Used for creating Polynomial Regressions

* from sklearn.svm import ...
   * [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - the model class for Support Vector Classification & Kernel SVM
   * [SVR](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) - the model class for Support Vector Regression

* from sklearn.linear_model import ...
   * [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) - used for Linear Regressions (single and multiple variables).
   * [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - used for Logistic Regressions

* from sklearn.tree import ...
   * [DecisionTreeRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) - used for Decision Tree Regression
   * [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) - used for Decision Tree Classification

* from sklearn.ensemble import ...
   * [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) - used for Random Forest Regression
   * [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - used for Random Forest Classification

* from sklearn.neighbors import [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) - K-Neighbours Classification model
* from sklearn.naive_bayes import [GaussianNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) - Naive Bayes model

* import [scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) as sch - A popular library that can be used for dendrogram creation in Hierarchical Clustering

* from sklearn.cluster import ...
   * [AgglomerativeClustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) - A Hierarchical Clustering Model
   * [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - K-Means clustering model

* from sklearn.discriminant_analysis import [LinearDiscriminantAnalysis](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis) as LDA - Linear Discriminant Analysis model

* from sklearn.decomposition import ...
   * [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) - Principal Component Analysis model
   * [KernelPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) - Kernel PCA model


## Keras

* from keras.models import [Sequential](https://keras.io/models/sequential/) - basic building block to creating a model

* from keras.layers import ...
   * [Dense](https://keras.io/layers/core/#dense) - basic function for linear models
   * [Dropout](https://keras.io/layers/core/#dropout) - used to add dropout to layers
   * [Flatten](https://keras.io/layers/core/#flatten) - used to flatten convolutional layers
   * [Conv2D](https://keras.io/layers/convolutional/#conv2d) - a basic convolutional layer
   * [MaxPooling2D](https://keras.io/layers/pooling/#maxpooling2d) - used to apply max pooling to a convolutional layer

* from keras.wrappers.scikit_learn import [KerasClassifier](https://keras.io/scikit-learn-api/) - used to wrap a sequential model to allow the model to be fit to datasets