# What is Machine Learning?
Machine Learning is a field of computer science that gives computers the ability to learn without being explicitly programmed. This is done through algorithms that learn from data you have trained them on. 

## Table of Contents
* [Importing Data](#importing-data)
* [Missing Data](#missing-data)
* [Categorical Data, Label Encoding & Dummy Encoding](#categorical-data-label-encoding--dummy-encoding)
* [Splitting the Dataset](#splitting-the-dataset)
* [Feature Scaling](#feature-scaling)

## Data Preprocessing
This is a core fundamental of a Machine Learning model. It is used to ensure that your data is prepared correctly for the model to accept the data from your dataset or database. 

Preprocessing your data can be different for every model and dataset that is being used, it's important to understand the following rules when doing this.

### Importing Data
By far the most important step - importing the data. This can be done simply by using the pandas library via the [.read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) function.

Here is an example:
```python
# Importing the dataset
dataset = pd.read_csv('data.csv')
```

From there, you need to make sure the dependent variables and independent variables are being selected correctly.

A dependent variable (DV) – is a variable that you are trying to explain. For example: How does a person’s salary change with the years of experience they have.

An independent variable (IV) – is a variable that is causing the dependent variable to change. 

![dataset example](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/dataset-example1.png)

We can use the pandas [.iloc](http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.iloc.html) function to output all columns and rows for both variables. To get the values we would add .values to the end of the function.

```python
# Outputs the columns Country -> Salary + all it's values - Independent Variable (IV)
X = dataset.iloc[:, :-1].values

# Outputs the last column + all its values - Dependent Variable (DV)
y = dataset.iloc[:, 3].values
```

### Missing Data
Sometimes our dataset will have empty fields. In order to prevent any errors with our models we need to occupy this space. One method is to fill them with the mean of each column. This can be done using the [imputer class](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) from the Scikit-Learn library.

```python
# Import library and class
from sklearn.preprocessing import Imputer

# Initalize Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

# Grab only the columns with the missing data
imputer = imputer.fit(X[:, 1:3])

# Replace the missing fields of data with the mean of the column
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

### Categorical Data, Label Encoding & Dummy Encoding
If your categorical data isn't a set of numbers, this needs to be converted to allow them to be recognised within our models. 

Firstly, we need to Label Encode the data to convert them to values. Each category has a value, if there are 3 categories these will be converted to 0, 1 & 2 (E.g. France being 0, Germany being 1 & Spain being 2). 

Next we Dummy Encode the data to allow our model to understand each categories index. These are automatically put to the front of your dataset.

![dummy encoding example](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/dummy-encoding.png)

If there are 2 categories we only need to Label Encode the categorical data as it will be a binary response (0 & 1).

This can be done through the classes [Label Encoding](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) and [One Hot Encoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) from the Scikit-Learn library.

```python
# Import library and classes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Initalise LabelEncoder for IV
labelEncoder_X = LabelEncoder()

# Change IV Values to an array of 'label numbers' & add them to X
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

# Use Dummy Encoding: Split the IV into 3 separate columns - only first column
oneHotEncoder = OneHotEncoder(categorical_features=[0])

# Dummy Encode IV & convert to array
X = oneHotEncoder.fit_transform(X).toarray()

# Initalise LabelEncoder for DV
labelEncoder_y = LabelEncoder()

# As yes & no responses - don't need to Dummy Encode DV (binary responses)
y = labelEncoder_y.fit_transform(y)
```

#### Dummy Variable Trap
Sometimes you may encounter the dummy variable trap. This is when you have ALL your dummy encoded variables included in your model, doing this can prevent the regression from working properly. This is due to them being highly correlated. For example: if you have 9 dummy variables, you only need 8 to be included. Always remove 1 dummy variable from the total number.

```python
# Avoiding the Dummy Variable Trap (Remove first column from X)
X = X[:, 1:]
```

### Splitting the Dataset
We need to split the dataset into 2 sets: the test set & the training set. This is done to provide our model with enough information to make effective at making predictions. We train it on the majority of the data and then test its performance on a smaller sample. 

An 80/20 split is a recommend number for training your models but this can be increased and decreased as you see fit. 

This can be implemented through the function [Train Test Split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from the Scikit-Learn library. 

Small datasets will not always need to be split.

```python
# Import library and function
from sklearn.cross_validation import train_test_split

# Test set: 20%; Training Set: 80%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Feature Scaling
In our dataset we have two columns that are both on different scales, Age & Salary. This will cause issues with our model if we do not put them onto the same scale. 

There are 2 types of Feature Scaling: Standardisation & Normalisation.

![feature scaling](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/feature-scaling.png)

Standardisation is when features are rescaled to have the properties of a standard normal distribution. This means that the features are centred around 0 with a standard deviation of 1. Standardisation is commonly used in general Machine Learning models.

Standard deviation is a measurement of spread that is used to calculate how spread out a set of data is. A low standard deviation means that the data is clustered around the mean/average and a high standard deviation means that the data is spread over a wider range of values.

Normalisation, also known as Min-Max Scaling, is an alternative to Standardisation. This is when data is scaled to a fixed range (usually 0 to 1). The purpose of this it to end up with smaller standard deviations, which can prevent the effect of outliers. Normalisation is more commonly used in Deep Learning and image processing. 

You can read more about this in an incredible blog by Sebastian Raschka [here](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html).

In this example I use Standardisation, this can be taken from the Scikit-Learn library using the [StandardScaler class](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

Normalisation can be implemented using the [MinMaxScaler class](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) from the Scikit-Learn library.

```python
# Import library and classes
from sklearn.preprocessing import StandardScaler

# Initialise Feature Scaling
sc_X = StandardScaler()

# Scale X Training set by fitting it to X & transforming it
# X Train must be done first before X Test. This ensures that both sets are on the same scale
X_train = sc_X.fit_transform(X_train)

# Scale X Test set by transforming it
X_test = sc_X.transform(X_test)
```