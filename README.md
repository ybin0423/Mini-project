# Bitcoin price prediction with ML

## Introduction
The increasing variability in Bitcoin prices has prompted the exploration of machine-learning models for price prediction. In this project, I aim to forecast the future price of Bitcoin by leveraging the Bitcoin price dataset from 1 January, 2021 to 12 May 2021 in 1-minute intervals. We have implemented three different algorithms, namely Linear Regression, Decision Tree Regression, and XGBoost Regression, to compare their predictive performance.

## Problem Formulation
The dataset used for this analysis was obtained from *https://www.kaggle.com/datasets/aakashverma8900/bitcoin-price-usd/data.*
**Input**: The input features for our machine learning models include historical attributes such as 
Open, High, Low, and Volume.
**Output**: The output, or the target variable, is the closing price of Bitcoin.
**Dataset**: The dataset comprises various features, including Open Time, Close Time, Quote Asset Volume, Number of Trades, Taker Buy Base Asset Volume, Taker Buy Quote Asset Volume, and more.
**Number of Samples**: The dataset consists of 3229 samples, from 2014-09-17 to 2023-07-20
with each entry representing a specific time frame.

## Baseline

**Decision Tree Regression**

● Hyperparameters: I conducted a grid search for hyperparameter tuning, exploring
options such as max_depth, min_samples_split, and min_samples_leaf.

● Tuned Parameters: The best parameters obtained from the grid search were used to
initialize the Decision Tree Regression model, from max_depth = [3, 5, 7], minimum
sample split = [2, 5] and minimum sample leaf = [1, 2, 4].

**Ridge Regression**

● Hyperparameters: The alpha parameter, controlling the regularization, was optimized
through grid search.

● Tuned Parameters: The best alpha value obtained from the grid search was employed in
training the Ridge Regression model, from alpha = [0.001, 0.1, 1].

**XGBoost Regression**

● Hyperparameters: I performed a grid search for parameters like learning_rate,
n_estimators, max_depth, subsample, and colsample_bytree.

● Tuned Parameters: The optimal parameters determined from the grid search were
utilized to configure the XGBoost Regression model, from learning_rate = [0.01, 0.1],
n_estimators = [30, 50], maximum depth = [3, 5, 7], subsample = [0.8, 0.9, 1.0] and
colsample by tree = [0.8, 0.9, 1.0].

