# MachineLearning02450

 This repository contains materials used as part of the course 02450 Introduction to Machine Learning and Data Mining, at DTU in Spring 2023. More details about the course can be found on the [course website](https://www2.compute.dtu.dk/courses/02450/).

The topics covered includes the following:

**Data Analysis**

- Data, feature extraction
- Principal Component Analysis (PCA)
- Measures of similarity, summary statistics and probabilities
- Probability densities
- Data visualization

**Supervised Learning**

- Decision trees
- Linear regression
- Logistic regression
- K-Nearest Neighbors (KNN)
- Bayes and Naive Bayes
- Artificial Neural Networks (ANN)
- AUC and ensemble methods
- Cross-validation
- Performance evaluation

**Unsupervised Learning**

- K-means
- Hierarchical clustering
- Mixture models, density estimation
- Association mining

## Project

A self-defined project was completed in a group of 3 as a hands-on application of the concepts taught in class. My group worked on the [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) was obtained from UCI Machine Learning Repository. The data was
collated from Capital Bikeshare System based in Washington D.C. in the United States of America
between 2011 and 2012 by Hadi Fanaee-T.

The project had 2 primary goals:

- Regression model: to predict the total count of bike users based on the attributes that describe the weather conditions (temp, atemp, humidity, windspeed).
- Classification model: to classify the type of day (working day vs non-working day) based on the weather conditions (weathersit, temp, atemp, humidity, windspeed).

The machine learning models and cross validation workflow were built from scratch instead of using the pre-built models available in scikit-learn. The project was split into 2 report submission.

### Project 1

This report was more qualitative in nature, describing the problem, data set, and the steps taken for data cleaning. It covered the data attributes, transformations, issues, summary statistics, visualization and principal component analysis (PCA).

### Project 2

This was a technical report and contained the summary of results obtained after running the machine learning models. For regression, ANN, linear regression and baseline models were used. For classification, KNN, logistic regression and baseline models were used.
