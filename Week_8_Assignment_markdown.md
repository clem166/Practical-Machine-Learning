---
title: "Activity Prediction based on Weight Lifting Data"
author: "Clement Yeung"
date: "July 14, 2019"
output: 
  html_document: 
    keep_md: yes
---


# Introduction
We are using data from http://groupware.les.inf.puc-rio.br/har, which contains personal activity data collected using devices such as Jawbone Up, Nike FuelBand and FitBit - in order to see if we can determine whether we can identify how the activity is being performed. The data collected contains lifts performed correctly and incorrectly in 5 different ways. The purpose of this activity is to fit a model to see if we can classify an observation into these groups using only the measurment data

## Dataset preperation

```r
#set libraries
library(dplyr)
library(caret)
library(ggplot2)

#set wd
set.seed(123)
setwd("C:/Users/theGameTrader/Documents/R/coursera/Practical Machine Learning")

training = read.csv("./pml-training.csv")
testing = read.csv("./pml-testing.csv")

#X variable appears to be the index and the data is sorted
pm_training = training[, -1]
pm_testing =  testing[, -1]

#exploratory analysis
head(pm_training)
colnames(pm_training)
summary(pm_training)
```
The data appears to have been sorted, with the X variable being the index of the dataset. A quick boxplot would reveal that the index perfectly seperates each classe. However, this is an artefact of the data and would have had poor performance in a new testing dataset - and was hence removed.


```r
colMeans(is.na(pm_training))
#drop columns where there is no data
pm_training2 = pm_training[, -which(colMeans(is.na(pm_training)) > 0.95) ]
```

Taking a look at the summary statistics and general layout of the data. There are a large number of variables which have missing data. I've dropped these variables to improve computation time as well as ensure that the remaining variables can be used for estimation.

### Factor Variable Analysis

```r
str(pm_training2)
nlevels(pm_training2)
```
When looking at the characteristics of the columns, it was noted that a large number of variables appear to have been incorrectly classified by the intial dataset import. In particular, there are factors which have thousands of levels - which would be better treated as numeric. If left in as factors, this can cause issues with the model fitting.


```r
#identifying factor variables
factor_cols = colnames(Filter(is.factor, pm_training2))
f_training = pm_training2[, which(colnames(pm_training2) %in% factor_cols)]
#levels in each factor variable
factor_levels = f_training %>% sapply(nlevels)
head(factor_levels)
```

```
##           user_name      cvtd_timestamp          new_window 
##                   6                  20                   2 
##  kurtosis_roll_belt kurtosis_picth_belt   kurtosis_yaw_belt 
##                 397                 317                   2
```

The above code grabs the variables which are factors from the training dataset and then checks the number of factor levels for each variable. We can see that it is variables of a certain type that seem to have large factor levels that could be better described numerically. 


```r
#variables that should be numeric instead of factor
to_match = c("kurtosis", "skewness", "min", "max", "amplitude")
match_cols = grepl(paste(to_match, collapse="|"), colnames(f_training))
#convert the above matched cases to numeric
f_training2=f_training
f_training2[, match_cols] = apply(f_training[, match_cols], 2, function(x) as.numeric(as.character(x)))
f_training3 = f_training2[, -which(colMeans(is.na(f_training2)) > 0.95)]
str(f_training3)
```
We extract the measurements that would be better treated as numeric variables, being the datapoints that we have collected describing the type of excercise being done. We then perform the same missing-data analysis that we did on the previous dataset


```r
#non-factor variables
nf_training = pm_training2[ , -which(colnames(pm_training2) %in% factor_cols)]
clean_training = cbind(nf_training, f_training3)
```

We combine the factor and non-factor datasets together so that we have our original training set that has been cleaned.

# PreProcessing

```r
pca_preproc = preProcess(clean_training[ , -length(clean_training)] , method=c("center", "scale", "pca"))
pca_preproc
```

```
## Created from 19622 samples and 58 variables
## 
## Pre-processing:
##   - centered (55)
##   - ignored (3)
##   - principal component signal extraction (55)
##   - scaled (55)
## 
## PCA needed 26 components to capture 95 percent of the variance
```

```r
pca_training = predict(pca_preproc, clean_training[, -length(clean_training)])
pca_testing = predict(pca_preproc, pm_testing)
```
As there are still a large number of variables, we perform principal components analysis to further cut down the features set. We center, scale and then use the preProcess function from caret to clean the data - excluding the dependent variable that we are trying to predict.


```r
pca_training2 = data.frame(clean_training$classe, pca_training)
```
We then recombine the dataset to include the variable of interest.

# Model Fitting

```r
mtry_no = floor(sqrt(ncol(pca_training)))
tune_grid = expand.grid(.mtry=mtry_no)
```
To use the pre-proccessed dataset in the modelling component, we need to set the mtry number - we have set it at the default rule of thumb that is used in the caret package. If we had used the default option of preprocessing in the caret function instead of doing it seperately; the feature search space would have been based on the total features of the original dataset which would have caused errors. 


```r
model3 = train(clean_training.classe ~ ., data=pca_training2, method = "rf", 
               trControl = trainControl(method="cv", number=5), tuneGrid = tune_grid)
model3
```

```
## Random Forest 
## 
## 19622 samples
##    29 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15697, 15698, 15696, 15699, 15698 
## Resampling results:
## 
##   Accuracy   Kappa   
##   0.9842525  0.980078
## 
## Tuning parameter 'mtry' was held constant at a value of 5
```

```r
model3$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 5
## 
##         OOB estimate of  error rate: 1.25%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5562    8   10    0    0 0.003225806
## B   35 3738   24    0    0 0.015538583
## C    1   32 3378   11    0 0.012857978
## D    0    0  103 3106    7 0.034203980
## E    0    0    0   15 3592 0.004158581
```
We fit the model using the randomforest method on PCA preprocessed data, and we set the number of cross validation folds to 5. The accuracy of the model is at around 98.5%, and the OOB error rate is estimated to be at 1.27%


```r
predict3 = predict(model3, pca_training2)
confusionMatrix(predict3, pca_training2$clean_training.classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2844     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
predict3_2 = predict(model3, pca_testing)
summary(predict3_2)
```

```
## A B C D E 
## 7 8 1 1 3
```
On the training dataset, the model appears to be able to correctly identify the classe variable for all observations. We then fit the testing dataset using the randomforest model. We were able to identify all cases correctly based on the quiz results.

## Additional Notes
There were other models that we tested included CART and randomforest without PCA. The results of the randomforest model without PCA were the same as the one with PCA - however, it took much longer to run so we have not included it in this analysis. The CART model ran much faster than either model, and was used to help identify issues with the dataset - e.g. the identification of the index variable being the dominating variable if left in the dataset. We have not included this analysis in this document, and instead have just used the results to clean the dataset.


