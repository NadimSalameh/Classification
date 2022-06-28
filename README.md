
# Classification

In this project, we will use Logistic Regression as a machine learning model to predict passenger survival on the Titanic.


## Authors

- [@NadimSalameh](https://github.com/NadimSalameh)


## Technologies and Tools

* python 3.9.7
* Jupyter Notebook
* Kaggle
* Pandas 
* numpy
* Matplotlib
* seaborn
* Sklearn
* LogisticRegression
* Kaggle
## Display confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, predictions, normalize=None)

```
![App Screenshot](https://github.com/NadimSalameh/Classification/blob/main/Display_Confusion_Matrix.png)

## Introduction

 In this project, we will use Logistic Regression as a machine learning model to predict passenger survival on the Titanic.

## Step 1 : Set up your project.

* Define Buisness Goal
* Set up Python + bash
* Create project folder


## Step 2 : Get Data.

* Download csv files from https://www.kaggle.com/competitions/titanic/data
* Load Data into Pandas.


## Step 3 : Train and split the data.

* The naming of datasets  is confusing.
*Note* :The *train.csv* file is your full dataset. You will need to split this
dataset into training and test sets — training set to build your model on, and test
set to evaluate how your model is doing.

The *test.csv* is not the "real" test set in that it doesn't give you labels. This is
just a dataset *Kaggle* uses to evaluate your model. You can think of this as *kaggle_submission.csv*.

## Step 4 : Exploratory Data Analysis(EDA)

* Create heatmap for Null Values
* Create countplot , boxplot, displot and pairplot



## Step 5 : Feature Engineering

* Deal with missing values with feature Engineering
* Use *SimpleImputer* for *Age* columns to fill null values by the mean of the column
* Use *one-hot-encoding* for categorical columns :*Sex*, *Embarked*, *Pclass*

## Step 6 : Apply Logistic Regression model

* Fit and transform train data(X_train), and transform test data
* Apply Logistic Regression 
* Create prediction
* Display confusion Matrix
* Upload to Kaggle.