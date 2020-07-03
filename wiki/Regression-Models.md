## What is the difference between Regression and Classification?
Regression is used to predict data that can be measured (continuous data).

Classification is used to predict which data should be categorised together (discrete data).

## Table of Contents
* [Simple Linear Regression](#simple-linear-regression)
* [Multiple Linear Regression](#multiple-linear-regression)
* [Methods for Building Models](#methods-for-building-models)
* [Polynomial Regression](#polynomial-regression)
* [Support Vector Regression (SVR)](#support-vector-regression-svr)
* [Decision Tree Regression](#decision-tree-regression)
* [Random Forest Regression](#random-forest-regression)
* [Evaluating Regression Models Performance](#evaluating-regression-models-performance)

## Simple Linear Regression
This consists of 1 independent variable.

![Simple Linear Regression](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/simple-linear-regression.png)

* A dependent variable (DV) – is a variable that you are trying to explain. For example: How does a person’s salary change with the years of experience they have.

* An independent variable (IV) – is a variable that is causing the dependent variable to change.

* Coefficient for the IV - this is how the effect or how a unit change in the IV changes the DV.

* Constant - this is the expected mean value of the DV when the IV is equal to 0. This is also known as the intercept and is the starting point of the regression line in a graph.

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/1.%20regression/0.%20linear/simple_linear_regression.py) for an example of a simple linear regression. To make a Linear Regression use the [LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) class from the Scikit-Learn library.

```python
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

# Create the SLR machine/model
regressor = LinearRegression()

# Fit the training data to the model
regressor.fit(X_train, y_train)
```

## Multiple Linear Regression
Same as a Simple Linear Regression but with more variables.

![Multiple Linear Regression](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/multiple-linear-regression.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/1.%20regression/0.%20linear/multiple_linear_regression.py) for an example of a multiple linear regression. This example uses backward elimination.

```python
# Fitting the Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predictng the Test set results
y_pred = regressor.predict(X_test)
```

### Assumptions of a Linear Regression
This relates to assumptions made on a dataset to determine what the best model is to use. 

This consists of the following:
1. Linearity - the data has a linear relationship between the dependent variable and independent variables.

2. Homoscedasticity - This is when the independent variables size differs from around the regression line. The further their variance the higher this will be.

3. Multivariate normality - This is when using a multiple linear regression and we assume that the residuals are normally distributed. A residual is the distance between an observed value of the dependent variable and the predicted value.

4. No auto-correlation - This means that there is no statistical relationship between the data.

5. Lack of multicollinearity - When using a multiple linear regression, we assume that the independent variables are not highly correlated with each other. This is tested using Variance Inflation Factor (VIF) values.

## Methods for Building Models
1. All-in - this is when you throw in all your variables. You would use this when:

   * This is commonly done when you have prior knowledge on the model.
   * If you don't have a choice. E.g. the company you work for tells you to use these specific variables.
   * When preparing for Backward Elimination. 

2. Backward Elimination

   * Step 1 - Select a significance level to stay in the model. E.g. SL = 0.05 (5%).
   * Step 2 - Fit the full model with all possible predictors (Optimal Matrix and Features).
   * Step 3 - Consider the predictor with the highest P-value. If P > SL, go to Step 4 otherwise go to FIN.
   * Step 4 - Remove the predictor.
   * Step 5 - Fit model without this variable (you need to rebuild the model to fit that number of variables). 
              Once this has been done, repeat from Step 3.
   * FIN - The model is ready.

3. Forward Selection

   * Step 1 - Select a significance level to enter the model. E.g. SL = 0.05 (5%).
   * Step 2 - Fit all simple regression model's y ~ Xn. Select the one with the lowest P-value.
   * Step 3 - Keep this variable and fit all possible models with one extra predictor added to the one(s) you already 
              have.
   * Step 4 - Consider the predictor with the lowest P-value. If P < SL, go to Step 3, otherwise go to FIN.
   * FIN - Keep the previous model.

4. Bidirectional Elimination

   * Step 1 - Select a significance level to enter and stay in the model. E.g. SL ENTER = 0.05 (5%), SL STAY = 0.05 (5%).
   * Step 2 - Perform the next step of Forward Selection (new variables must have: P < SL ENTER to enter).
   * Step 3 - Perform ALL steps of Backward Elimination (old variables must have P < SL STAY to stay). Once done, repeat 
              Step 2.
   * Step 4 - No new variables can enter and no old variables can exit.
   * FIN - The model is ready.

5. Score Comparison/All Possible Models

   * Step 1 - Select a criterion of goodness of fit (e.g. Akaike criterion).
   * Step 2 - Construct All Possible Regression Models 2n-1 total combinations.
   * Step 3 - Select the one with the best criterion.
   * FIN - The model is ready.

## Polynomial Regression
This is like the Multiple Linear Regression except it has independent variables with powers. This creates curved lines on a graph rather than straight ones.

<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/1.%20regression/1.%20non_linear/polynomial_regression.py) for an example of a polynomial regression comparing against a linear regression model. To make a Polynomial Regression use the [PolynomialFeatures](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) class from the Scikit-Learn library.
--->
```python
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
```

## Support Vector Regression (SVR)
This is a fast and accurate way of using datasets, it learns quickly and is systematically improvable. Variants of SVR are used throughout science including Kriging and Gaussian Process (GP). SVR is a generalization of the Support Vector Machine (SVM) classification model.
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/1.%20regression/1.%20non_linear/support_vector_regression.py) for an example of an SVR model. To make an SVR use the [SVR](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) class from the Scikit-Learn library.
--->
```python
# Fitting the SVR to the dataset
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)
```

## Decision Tree Regression
<!---
![Decision Tree Graph](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/decision-tree-graph.png)
--->
These are used when the response variable is numeric or continuous. For example, the predicted price of a consumer good. 

We fit a regression model to the target variable using each of the independent variables and the data is split at several split points. At each split point, the "error" between the predicted value and the actual values is squared to get a "Sum of Squared Errors (SSE)". The errors of the split points are compared and the variable/point yielding the lowest SSE is chosen as the root node/split point. This process is recursively continued.
<!---
![Decision Tree Regression](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/decision-tree-regression.png)

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/1.%20regression/1.%20non_linear/decision_tree_regression.py) for an example of a decision tree regression model. You can make a decision tree regression model by using the [DecisionTreeRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) class from the Scikit-Learn library.
--->
```python
# Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
```

## Random Forest Regression
This is a form of ensemble learning, it uses multiple of the same algorithm (in this case the Decision Tree Regression) and puts them together to make a more powerful model.

How it works:
* Step 1 - Pick at random K data points from the Training set.
* Step 2 - Build the Decision Tree associated to these K data points.
* Step 3 - Choose the number Ntree of trees you want to build and repeat steps 1 & 2.
* Step 4 - For a new data point, make each one of your Ntrees predict the value of Y for the data point in question. 
           Assign the new data point the average across all the predicted Y values.
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/1.%20regression/1.%20non_linear/random_forest_regression.py) for an example of a random forest regression model. You can make this using the [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) class from the Scikit-Learn library.
--->
```python
# Fitting the Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)
```

## Evaluating Regression Models Performance
This section consists of 2 helpful factors that can be used to improve regression models.

### R-Squared
R-squared is a statistical measure of how close the data is to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression. This can be used as a 'Goodness of Fit' and is a metric made up of Sum of Squares of Residuals & the Total Sum of Squares. Use this when determining how well the model predicts new observations and whether the model is too complicated.
<!---
![R-Squared](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/r-squared.png)

![R-Squared Equations](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/r-squared-equations.png)
--->
R-squared is always between 0 and 100%:
0% indicates that the model explains none of the variability of the response data around its mean.
100% indicates that the model explains all the variability of the response data around its mean.
The higher the R-squared, the better the model fits your data.

### Adjusted R-Squared
This is a modified version of R-squared that has been adjusted depending on the number of independent variables in the model. Use this when comparing models with different numbers of predictors (IVs).

This metric increases only if the new IV improves the model more than would be expected by chance. It decreases when an IV improves the model by less than expected by chance. The adjusted R-squared can be negative and is always lower than the R-squared.
<!---
![R-Squared Equations](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/adjusted-r-squared.png)
--->