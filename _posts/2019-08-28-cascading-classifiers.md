---
layout: post
mathjax: true
comments: true
title: "Cascading Classifiers"
subtitle: "Using a variant of ensemble learning to reduce false positives"
date: 2019-08-28
gh-repo: praths007/cascading_classifiers
gh-badge: [star, fork, follow]
---

A few months back I was working on a problem where I was dealing with a highly imbalanced (1:10) dataset. My features
were also not too good which resulted in a mediocre model. Another major issue I faced was a high number of false 
positives. I was working on an inventory optimization problem and my client was not 
particularly happy with having too many false positives because that would mean we were losing out on the good inventory.
Complete codes for this exercise can be found [here](https://github.com/praths007/cascading_classifiers).

#### Cascading classfiers
To reduce the false positives I used a concept of ensemble learning called cascading classifiers. The central idea here
is to build multiple levels of classification models and at each level I remove all my negative predicted values. In 
this way I am forcing the next level of model to retrain on all negative predicted values. It so happens that if I have 
a highly imbalanced dataset with lower number of positives, with each level of the cascade, as my negatives keep getting
removed, the model will be forced to train on the small number of positives. This will in turn result in correctly 
predicting the positives, reducing the number of false positives.

Specifically for this exercise I have used the simplest glm. But this technique can be used with any classification
algorithm. e.g. At my work I used extreme gradient boosting which gave pretty good results.

#### Steps for execution of normal glm
##### 1. Load dataset and split to test and train
I have used a sample opensourced dataset since I cannot use my clients data. But the general idea is same. Example 
datasets can be found [here](https://sci2s.ugr.es/keel/imbalanced.php). I have used the yeast dataset.

```python
## yeast dataset
# A imbalanced version of the Yeast data set, 
# where the positive examples belong to class ME3 and the negative examples belong to the rest.
yeast_dat = read.table("yeast1.dat",sep = ",",skip = 15) %>% as_tibble()

# convert target variable to an integer
yeast_dat = yeast_dat %>%
            mutate(target = ifelse(V9 ==" positive",1,0)) %>%
            mutate(V9 = NULL)

# check there are no strings or characters in any column
str(yeast_dat)

# create test and train data
index = unlist(createDataPartition(yeast_dat$target,p =  0.7))
train = yeast_dat[index,]
test = yeast_dat[-index,]
```
##### 2. Run glm and check the accuracy
###### 2.a glm execution
```python
# run logistic regression for classification
reg1 = glm(target ~ .,data = train,family='binomial')

# summary of the model
summary(reg1)
```
###### 2.b Get confusion matrix on test data
```python
## checking accuracy of glm
##########################################################
# get probabilty scores for validation set
preds = predict(reg1, test %>% select(-target), steps = 1)

ROCRpred = prediction(preds, test$target)
ROCRperf = performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf)

# get AUROC
auc = performance(ROCRpred, measure = "auc")
auc = auc@y.values[[1]]
auc
# 0.81

# probablity density plot
dat = data.frame(dens = preds, lines = test$target)
densityplot(~dens,data=dat,groups = lines,
            plot.points = FALSE, ref = TRUE, 
            auto.key = list(space = "right"))

# get confusion matrix
pred = ifelse (preds > 0, 1, 0)
caret::confusionMatrix(as.factor(pred), as.factor(test$target), positive = "1")

##########################################################
Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 302  64
         1  31  47
                                          
       'Positive' Class : 1        
```
Observe that the model does a a fairly good job of classifying the negatives (0), but mediocre job of classifying positives
(1). And also gives 31 false positives, out of the 47 classified as positives. This is nearly 40% of the values being
false negatives which is not a good result.

#### Steps for executing 2 level cascading classifier using glm
#### 1. Level 1 (train dataset)
##### 1.a Fit model on train and get confusion matrix
Since we have already fit our data previously we will use the same model as before for level 1 of the cascade.
Here we run the model on train data and check the confusion matrix.
```python
# cascaded classifier iterations (used glm) - 2 level cascade
##########################################################
###### implementing cascade (level 1)
## fit training set
train$predictions = predict(reg1, train %>% select(-target))

#################################################
## get density plots to determine threshold
dat = data.frame(dens = train$predictions, lines = train$target)
densityplot(~dens,data=dat,groups = lines,
            plot.points = FALSE, ref = TRUE,
            auto.key = list(space = "right"))

# opt =  optimalCutoff(train$target, train$predictions, optimiseFor = "missclasserror", returnDiagnostics = FALSE)


## classify based on threshold
train$pred_yeast = ifelse(train$predictions > 0, 1, 0)
#################################################

cfm_level_1 = caret::confusionMatrix(as.factor(train$pred_yeast),
                                     as.factor(train$target), positive = "1")
cfm_level_1

          Reference
Prediction   0   1
         0 663 200
         1  57 118
        
        'Positive' Class : 1 
```
##### 1.b Remove correctly predicted negatives from train
```python
#################################################
## removing correctly predicted 0's 2nd level cascade
## for train
train_lvl2 = train %>%
  filter((pred_yeast == 1) | (target == 1))

train_lvl2 = train_lvl2 %>%
  select(-c(pred_yeast, predictions))
################################################
```
#### 2. Level 1 (test dataset)
##### 2.a Check confusion matrix for level 1 test data
```python
## for test
test$predictions = predict(reg1, test %>% select(-target))


dat = data.frame(dens = test$predictions
                 , lines = test$target)
densityplot(~dens,data=dat,groups = lines,
            plot.points = FALSE, ref = TRUE,
            auto.key = list(space = "right"))

opt = optimalCutoff(test[,c("target")], test$predictions, optimiseFor = "misclasserror", returnDiagnostics = FALSE)


## classify based on threshold (considered less false negatives - yeasts classified as non yeasts)
test$pred_yeast = ifelse(test$predictions > opt, 1, 0)
#################################################

cfm_level_1_test = caret::confusionMatrix(as.factor(test$pred_yeast),
                                          as.factor(test$target), positive = "1")
cfm_level_1_test

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 302  62
         1  31  49
         
         'Positive' Class : 1  

```

##### 2.b Remove negative predicted from test
For test we will never know if we have correctly predicted a row, because in real world scenario this data will
not be available. So we remove all negatively predicted values
```python
#################################################
test_lvl2 = test %>%
  filter((pred_yeast == 1))

test_lvl2 = test_lvl2 %>%
  select(-c(pred_yeast, predictions))
```

#### 3. Level 2 (train dataset)
After the first level we proceed to the second level of the cascade. 2 things to keep in mind is that we have removed
the **correctly** predicted negatives from our train data and the **predicted negatives** from our test data, since for
test we have no way of knowing of knowing the actuals.

##### 3.a Fit model on level 2 train and get confusion matrix
```python
## fitting
##########################################################

reg2 = glm(target ~ .,data = train_lvl2,family='binomial')


## high difference in validation and training accuracy hints
## high variance


train_lvl2$predictions = predict(reg2, train_lvl2 %>% select(-c(target)))

#################################################
## get density plots to determine threshold
dat = data.frame(dens = train_lvl2$predictions
                 , lines = train_lvl2$target)
densityplot(~dens,data=dat,groups = lines,
            plot.points = FALSE, ref = TRUE,
            auto.key = list(space = "right"))

opt = optimalCutoff(train_lvl2[,c("target")], train_lvl2$predictions, optimiseFor = "misclasserror", returnDiagnostics = FALSE)


## classify based on threshold
train_lvl2$pred_yeast = ifelse(train_lvl2$predictions > opt, 1, 0)
#################################################

cfm_level_2 = caret::confusionMatrix(as.factor(train_lvl2$pred_yeast),
                                     as.factor(train_lvl2$target), positive = "1")
cfm_level_2

Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0  13   7
         1  44 311
         
        'Positive' Class : 1
```
It can be seen that the false positives reduced from 57 in level 1 confusion matrix to 44 in level 2 confusion matrix.
This looks like a good sign, because then we would have also have reduced false positives in our test data.
##### 3.b Remove correctly predicted negatives from train
Similar to level 1 we remove the **correctly** predicted negatives from train
```python
#################################################
## subsetting predicted "1s" for 2nd level cascade
## for train
train_lvl3 = train_lvl2 %>%
  filter((pred_yeast == 1) | (target == 1))

train_lvl3 = train_lvl3 %>%
  select(-c(pred_yeast, predictions))
```

#### 4. Level 2 (test dataset)
##### 4.a Check confusion matrix for level 2 test data
```python
################################################
## for test
test_lvl2$predictions = predict(reg2, test_lvl2 %>% select(-target))


dat = data.frame(dens = test_lvl2$predictions
                 , lines = test_lvl2$target)
densityplot(~dens,data=dat,groups = lines,
            plot.points = FALSE, ref = TRUE,
            auto.key = list(space = "right"))
## no visible separation of probability densities
## this should be the last level of cascade

opt = optimalCutoff(test_lvl2[,c("target")], test_lvl2$predictions, optimiseFor = "misclasserror", returnDiagnostics = FALSE)

## classify based on threshold
test_lvl2$pred_yeast = ifelse(test_lvl2$predictions > opt, 1, 0)
#################################################

cfm_level_2_test = caret::confusionMatrix(as.factor(test_lvl2$pred_yeast),
                                          as.factor(test_lvl2$target), positive = "1")
cfm_level_2_test

Confusion Matrix and Statistics

          Reference
Prediction  0  1
         0 10  6
         1 21 43
         
          'Positive' Class : 1 
```

##### 4.b Remove negative predicted from level 2 test
```python
#################################################
test_lvl3 = test_lvl2 %>%
  filter((pred_yeast == 1))


test_lvl3 = test_lvl3 %>%
  select(-c(pred_yeast, predictions))

```

#### 5. Compare confusion matrix from level 1 with level 2
```python
#################################################
test_check_2 = test_lvl2


test_check_2 = test_check_2 %>%
  rowwise()%>%
  mutate(labels = assign_labels(target, pred_yeast))



test_check_2 = test_check %>%
  rbind(test_check_2)

### comparing cascaded confusion matrix with first iteration confusion matrix
#################################################
cfm_level_1_test
          Reference
Prediction   0   1
         0 302  62
         1  31  49

caret::confusionMatrix(as.factor(test_check_2$pred_yeast),
                       as.factor(test_check_2$target), positive = "1")
          Reference
Prediction   0   1
         0 312  68
         1  21  43

```
As we can see the false positives have reduced from 31 to 21. This effect is substantial in cases where the dataset is
large and has more features.


