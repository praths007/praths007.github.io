XGBoost is a piece of software with an algorithm that is used to increase your model's performance and execution speed. It is mainly used with decision trees.

The documentation and installation instructions can be found [here](https://xgboost.readthedocs.io/).

See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/8.%20xgboost/xgboost.py) for an example of XGBoost. 

```python
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)
```