## Table of contents

- [When to retrain](#when-to-retrain)


## When to retrain
As time changes the distributions within data keep changing. So a model deployed a year back might not be the same.
eg. risk for a life insurer, if person is fairly average - non smoker, avg height etc. NN will give average prediction. 
As we progress mortality is improving. More people are living to high end. Smokers are becoming less. Therefore ground
truth is changing. So the model needs to be retrained.

### Techniques on a closed dataset
This [video](https://youtu.be/K2Tjdx_1v9g) explains in detail.
Dataset shift and Covariate Shift. 
Measuring drift in dataset.

KS-Statistic: <br>
```python
from scipy import stats

stats.ks_2samp(train_df["kitchensq"], train_df["kitchensq"])
# this has pvalue 1 no change
# but

stats.ks_2samp(train_df["kitchensq"], test_df["kitchensq"])
# this statistic shows change p value becomes 0

# similary this can be tested for every other column in train and test set
```

Detecting drift between training and testing dataset. <br>
* sample train and test set into smaller sets
* add another column source_training to check which set they came from
* combine together and re randomize them
* fit a model (random forest) can the model predict where an individual row came from
    * Use each column one by one to fit model on source_training
* ideally we should not be able to predict source_training. SO anything with AUC > 0.75 means that there is a drift.
* eg. stuff like timestamp, id is ever increasing so test and train will be totally different. 
