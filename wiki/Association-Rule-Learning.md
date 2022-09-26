'People who bought... also bought...'

Association rules analysis is a technique to uncover how items are associated to each other.

## Table of Contents
* [Apriori](#apriori)
* [Eclat](#eclat)

## Apriori
This algorithm is commonly used in data mining. It consists of 3 parts - support, confidence and lift. 
<!---
![Apriori](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/apriori.png)
--->
Using a list of movies as an example, M stands for a set of movies.
<!---
![Support](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/apriori-support.png)

![Confidence](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/apriori-confidence.png)

![Lift](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/apriori-lift.png)
--->
This is how it works:
* Step 1 - Set a minimum support and confidence.
* Step 2 - Take all subsets in transactions having higher support than minimum support.
* Step 3 - Take all the rules of these subsets having higher confidence than minimum confidence.
* Step 4 - Sort the rules by decreasing lift.
<!---
See the code [here](https://github.com/Achronus/Machine-Learning-101/blob/master/coding_templates_and_data_files/machine_learning/4.%20association_rule_learning/0.%20apriori.py) for an example of an Apriori model. This uses an amazing python script [apyori](https://github.com/ymoch/apyori) from ymoch.
--->
```python
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
```

## Eclat
This is a simplified version of Apriori and is used to identify basics information on sets of items that have been purchased together. This only has 1 part - the support.
<!---
![Support](https://acius.co.uk/wp-content/themes/acius/machine_learning/imgs/ml/apriori-support.png)
--->
This is how it works:
* Step 1 - Set a minimum support.
* Step 2 - Take all the subsets in transactions having higher support than minimum support.
* Step 3 - Sort these subsets by decreasing support.