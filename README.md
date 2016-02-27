# Titanic - Machine Learning From Disaster

This is my attempt at creating a classifier from a dataset of Titanic passengers. The dataset is from a [Kaggle](http://www.kaggle.com) competition. It uses a Random Forest Classifier from the [sklearn library](http://scikit-learn.org/stable/) and a Grid Search to find optimal hyperparameters.

## Usage

In the terminal, simply type

```
python titanic_survivor_classifier.py
```

## Files
* train.csv - training data with labels
* test.csv - testing data without labels
* titanic_survivor_classifier.py - main Python file to load, preprocess, and predict data
* rfpredictions.csv - Predictions on the test data
