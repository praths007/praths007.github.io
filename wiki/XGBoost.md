XGBoost is a piece of software with an algorithm that is used to increase your model's performance and execution speed. It is mainly used with decision trees.

The documentation and installation instructions can be found [here](https://xgboost.readthedocs.io/).
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/8.%20xgboost/xgboost.py) for an example of XGBoost. 
--->
```python
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)
```

## Intuition
### Example of Adaboost
This [video](https://youtu.be/LsK-xG1cLYA) by Josh Stramer explains this clearly. Following is the gist of it.

Weak learners are generally stumps with only 2 leaves and 1 root and considering only 1 feature.
Steps:
* Assign Sample weight (eg. if 8 sample then each sample gets 1/8)
* Make first stump
    * Ignore weights for the time being because they are the same
    * Calculate gini index or entropy to get the best feature for splitting the stump
* Calculate how much say the stump has in final classification
    * Some stumps have more say in final classification based on say.
    * It is determined based on how well it classified samples
    * eg. if it made 1 error i.e. total incorrectly classified samples is 1. 
        * Total error for stump is sum of weights for incorrectly classified sample. 
        * So error for 1 misclassification in this case is 1/8.
        * Total error must always add up to 1
    * amount of say = 1/2 * log (1-total_error/total error
        * this is like a tanh graph so if the log value is very large the amount of day i highly negative and vice versa.
* Modify the weights so that the next stump considers the error this stump made
    * Now the current stump incorrectly classified 1 sample
        * So increase sample weight that was incorrectly classified
        * new sample weight = sample_weight * e^(+amount of say)
            * here if amount of say is big the new sample weight is high and the previous stump did a good job 
            classifying samples therefore the missclaissifcation will be assigned a very high weight (since the
            previous stump missed out even though it was a good split)
            * the reverse is true is the job was not good
        * and decrease the sample weight for correctly classified samples
            * new sample weight = sample_weight * e^(-amount of say)
            * Here if the previous stump did a good job then the new sample weight must be lower as these are already
            classified in a proper manner. (e^-ve is like a slide from y axis to x axis, so the value of e^-x decreases
            as x increases)
* Normalize the reassigned sample weights so they add up to 1.

* On to making the second stump - 
    * In theory use weighted gini index by checking the sample weights
    * Otherwise make new dataset with randomly picking more quantiites of samples with more weights.
        * This is done by using weights as a distribution and different ranges are selected and samples randomly picked.
        * Ultimately new sample has more of the missclassified samples so they have kind of a higher weight as they are
        the same and treated as a block creating large penalty for being missclassified.
        
    * Final classification is based on adding amount of say, the larger value is the classification Yes/no.
    

### Example of XGBoost    
