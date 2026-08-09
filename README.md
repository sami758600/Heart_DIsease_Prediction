#  Heart Disease Prediction using Machine Learning

## Project Overview

This project is an end-to-end **Machine Learning classification project** that predicts whether a patient is likely to have heart disease based on clinical measurements.

The project was built as a **binary classification problem**, where:

- `0` → No Heart Disease
- `1` → Heart Disease

The main model selected for the project is **Logistic Regression**.

The project was not limited to simply training a model. The complete workflow included:

- Understanding the dataset
- Checking data quality
- Exploring feature distributions
- Detecting outliers
- Studying features against the target
- Checking correlations
- Feature selection using L1 regularization
- Splitting data into training and testing sets
- Feature scaling
- Understanding and using ML Pipelines
- Cross-validation
- Training Logistic Regression
- Evaluating the model using multiple classification metrics
- Comparing Logistic Regression with SVM models
- Saving the trained model
- Creating a prediction function
- Building a Streamlit web application

---

#   Problem Statement

Heart disease prediction is a classification problem where patient information is used to predict whether the patient belongs to the heart-disease class.

The main goal of this project was to build a machine learning model capable of identifying positive heart-disease cases while paying particular attention to **false negatives**.

A false negative occurs when:

```text
Actual → Heart Disease
Model  → No Heart Disease

## My Experience
From this project I understood the ML pipelines as well as EDA. At first I thought  it is easy to do projects as we just need a dataset 
identify null values and outliers  remove or correct them  and remove useless features
but from this project I understood  that we cant remove  any feature just because we think its useless  we need to first see its distribution and find the upper and lower bounds
there are different methods for understanding data of continuous values and categorical values
this project made me learn something new.
