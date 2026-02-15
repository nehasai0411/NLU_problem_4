# Sport vs Politics Text Classification (BBC Dataset)

## Overview
This project implements a machine learning based text classifier that predicts whether a given news document belongs to the SPORT category or the POLITICS category.

The system is trained and evaluated on the BBC News Classification Dataset.
Three different feature representations and three different machine learning models are compared to satisfy the assignment requirement.

## Problem Statement
Given a text document, classify it into one of the following two classes:

- SPORT
- POLITICS

The classifier is built using standard machine learning techniques with the following feature representations:

- Bag of Words (BoW)
- TF-IDF
- TF-IDF with n-grams (unigrams + bigrams)

At least three ML techniques are compared quantitatively.

## Dataset
### Source
BBC News Classification Dataset (commonly used for text classification tasks).

### Dataset Format
The dataset is stored in a CSV file:
- bbc_data.csv

It contains two columns:
- data   -> the text of the news article
- labels -> the category label (sport, politics, etc.)

### Classes Used
Only the following labels were selected:
- sport
- politics

After filtering:
- Total documents used: 928
- Classes: SPORT, POLITICS

## Methods Used

### Feature Representations
1. Bag of Words (BoW)
2. TF-IDF
3. TF-IDF with n-grams (1,2)

### Machine Learning Models
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (Linear SVM)

## Train/Test Split
- 80% Training
- 20% Testing
- Stratified splitting is used to preserve class distribution.

## Results (Test Set)

| Method | Feature Type | Accuracy |
|--------|--------------|----------|
| Multinomial Naive Bayes | Bag of Words | 1.00 |
| Logistic Regression | TF-IDF | 0.989 |
| Linear SVM | TF-IDF (1,2) n-grams | 0.989 |

Note: Very high accuracy is expected because SPORT and POLITICS articles have highly distinct vocabularies.

