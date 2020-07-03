Dimensionality Reduction is used to reduce a dataset down to a smaller number of independent variables that can be plotted onto a graph. These independent variables will be the ones that have the biggest impact on the dataset. Dimensionality Reduction consists of two types: Feature Selection and Feature Extraction.

Feature Extraction creates new variables from your existing dataset.

Feature Selecting allows you to select which independent variables you want to use your dataset against.
<!---
![Dimensionality Reduction](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/dimensionality-reduction.png)
--->
## Table of Contents
* [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
* [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
* [Kernel Principal Component Analysis (Kernel PCA)](#kernel-principal-component-analysis-kernel-pca)

## Linear Discriminant Analysis (LDA)
Create new axis that maximum the separability of the categories.  This can be done by maximizing the distance between the means of the 2 categories. Then minimizing variation becomes:

(mew1 - mew2)/s1^2 + s2^2

where s1 and s2 is the scatter/spread/ variability/noise in the data.



From the *n* independent variables of your dataset, LDA extracts *p* ≥ *n* new independent variables that separate the most classes of the dependent variable. LDA is a supervised model.
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/6.%20dimensionality_reduction/0.%20linear_discriminant_analysis.py) for an example of a LDA. To make this use the [LinearDiscriminantAnalysis](http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) class from the Scikit-Learn library.
--->
```python
# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test= lda.transform(X_test)
```

## Principal Component Analysis (PCA)
PCA works by projecting the data points on orthogonal lines/principal components. This way the X1 and X2 variables are represented using the eigen vector on the principle component. This eigen vector is nothing but the distance of datapoint from the origin calculated using Pythagoras theorem on X1 and X2. 

PC1 = X1^2 + X2^2

In turn X1 and X2 will be transformed into loading scores for principal component.

More explanation in this [video](https://www.youtube.com/watch?v=FgakZw6K1QQ) by statquest with josh starmer.

From the *m* independent variables of your dataset, PCA extracts *p* ≥ *m* new independent variables that explain the most variance of the dataset, regardless of the dependent variable. As the dependent variable is not considered, PCA is an unsupervised model.
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/6.%20dimensionality_reduction/1.%20principal_component_analysis.py) for an example of a PCA. To make this use the [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class from the Scikit-Learn library.
--->
```python
# Applying PCA
from sklearn.decomposition import PCA
# pca = PCA(n_components = None)
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# Used to identify percentages of IVs
explained_variance = pca.explained_variance_ratio_
```

## Kernel Principal Component Analysis (Kernel PCA)
A kernelized version of PCA. This maps the data to a higher dimension using the kernel trick and extracts new principle components. This is commonly used for non-linear problems but makes the data linearly separable.
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/6.%20dimensionality_reduction/2.%20kernel_pca.py) for an example of a Kernel PCA. To make this use the [KernelPCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) class from the Scikit-Learn library.
--->
```python
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
```