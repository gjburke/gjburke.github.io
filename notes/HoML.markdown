---
layout: note
title: HoML
note_title: Hands on Machine Learning
author: Aurélien Géron 
---

# Table of Contents

1. [The Machine Learning Landscape](#the-machine-learning-landscape)
2. [End to End Machine Learning Project](#end-to-end-machine-learning-project)
3. [Classification](#classification)

# The Machine Learning Landscape

What is... Machine Learning?
- giving the ability of computers to "learn" from data
- performance increases after being a piece (or set) of data
Why use it?
- can be better for tasks that require automation, if there's no known algorithm, complex data

## Types of ML Systems

**Supervised** vs **Unsupervised**

>Supervised:
>- labelled w/ desired solutions
>- ex: classifciation, predictors. regression algs and extensions, k neareast neighbors, decision tree, rand forest

>Unsupervised:
>- Data is unlabelled 
>- ex: clustering, anomaly/novelty detection, visualizations & dimension reduction, association

>Semisupervised: have some labelled data in a lot of unlabelled data, often used to better identify classes in clustering

>Reinforcement: some sort of "reward" mechanism to "teach" the algorithm the proper output

**Batch** and **Online Learning**

>Batch:
>- system trained with all data at once, then deployed ("offline learning")
>- can be really resource intensive, taking a while and lot of computer to train
>- usually we cant incremental learning

>Online Learning
>- trained incrementally in 'mini-batches'
>- adapts to the new data on the fly, great for continuous flow
>- learning rate controls that increment 
>- need to be aware of new, data data that might decrease the performance

**Instance** vs. **Model-based**

>Instance
>- model generalizes by comparing new data to "learned" data
>- something like KNN, onyl really compares to data in set

>Model-based
>- model makes predictions that fit the training data (separate from data)
>- For devs: study data, select model, train it, apply to make predicitons

## Main Challenges in Machine Learning

- Not enough training data
  - more data is almost always better (as long as its good quality)
- Non-representative training data
- Poor quality training data
  - some outliers, missing a few features
- Irrelevant features
  - need to make sure of good feature selection and extraction
- Overfitting the training data
  - size of dataset should correlate with model complexity
  - model that can capture intricacies will find them in noise
  - to fix, you can try:
    - simplifying the model
    - getting more trainign data
    - reducing the dataset noise
- Underfitting
  - model is not strong enough
  - need to...
    - set a more powerful model
    - get better features
    - less constraint/regularization

## Testing and Validating

>Need to split set into test and train to make sure model is generalizing

>For model selection and tuning, need to also generalize past a particular test set
>- holdout validation - take part of a training set, evaluate multiple models, choose the best
>- cross validation - test on many, smaller validation tests

>Watch out for data mistmatch, where your testing set may be too similar to the training and not generalized enough for the problem at hand
>- may want to get even more general validation set

# End to End Machine Learning Project

Overall, looks like:
1. [Look at the big picture](#looking-at-the-big-picture)
2. [Get the data](#get-the-data)
3. [Discover, visualize data to gain insights](#discover-and-visualize-data-to-gain-insights)
4. [Prepare the data for ML algorithms](#prepare-the-data-for-ml-algorithms)
5. [Select and train a model](#select-and-traing-a-model)
6. [Fine tune it](#fine-tune-a-model)
7. [Present your solution](#present-your-solution)
8. [Launch, monitor, maintain the model](#launch-monitor-maintain)

## Look at the big picture

- Find exactly what the business objective is (determines how you frome the problem, algs, performance measure, etc)
- How the model situated within pipelines

>Steps:
>- Frame the problem (unsupervised vs supervised, regression, classifcation, etc)
>- Select performance measure (RSME, MAE, other error or success measures)
>- Check your assumptions (about the business side, what data you'll have, what you actually need to do)

## Get the data

- create workspace, download data, look at data structure
- then do some preliminary analysis/visualizations
- create a test set (DONT LOOK!) with good random and representative sampling methods

## Discover and visualize data to gain insights

Don't include the test data!
Visualize, play around with, analyze:
- different features, distibutions
- looking at feature correlations
- attribute combinations

## Prepare the data for ML algorithms

Write functions for this! (can use scikit learn pipelining for this)

Data cleaning
- deal with missing features: get rid of datapoint, whole attribute, or set to some mean or specific value

Handling text or categorical data
- usually set to some numerical or vector representation
- one Hot Encoding as a vector representation, but can be large if many options

Feature scaling
- need features to have similar range of values
- min-max scaling (normalization): shifted and rescaled to one
- standardization: subtract mean val, divide by standard deviation <- less affected by outliers

Custom transformers, transformation pipelines
- can use scikit-learn to automate certain dta transformations (data cleaning, handling non-numerics, feature scaling, etc)
- use customeTransformer class for this
- can put the custom transformer into transformtation pipelines to stack them, automate data cleaning

## Select and traing a model

Don't touch test set until confident with a model, but you may want to use a smaller training validation set, or do cross-validation.

Train a variety of models, evaluate under/over fitting and such, go from there

## Fine-tune a model

- grid search, randomized search, others
- meant to find the best hyperparameters
- once you think you've found your best, evaluate the final model on the test set (usually will be worse than corss-val, especially if hypertuned a lot. BUT DON'T RETUNE)

## Present your solution

This is the business aspect of things, so remember that not everyone understand machine learning

Should include things like:
- Recommendations on implementation of your model
- Good documentation for your model, reports, etc
- Showing the clear impact of your model (layman should understand at least the important parts, may have different presentations for different knowledge bases)
- Overall good communication (keep each audience in mind)

As stressed in some of the bullets, keep your audience in mind. If you are presenting your solution to the finance department vs the models department, you may need to use different terminology and explain to different levels of knowledge about models. But your impact should be clear to everyone.

## Launch, Monitor, Maintain

This involves...
- Plugging it into the pipeline
- Tests for the model's performance (trying to stop rot)
- Some sort of human evaluation and checking for the model's inputs and output within the pipeline
- Evualtion of the new input data to the model (prevent bad data)
- Retrain models on a regular basis (automate as much as possible)

# Classification

## Binary Classification

Basically a yes or no question, is something or isn't, etc.

### Performace Measures

Confusion Matrix

- tells you FP, FN, TP, TN of predictions
- rows are the actual class, cols are the predictors

>From this, you can determine:

>- <u>precision</u> - accuracy of the positive predictions - TP / (TP+FP) or TP / Total Positive Predictions
>- <u>recall/sensitivity</u> - ratio of positives detected by classifier - TP / (TP + FN) or TP / Total Actual Positives
>- <u>F1 Score</u> - harmonic mean of precision and recall - 2 * (precision * recall )/(precision + recall)

Precision/Recall Tradeoff

- Increasing precision reduces recall, and vice versa
- kind of looks like this:
  - ![Threshold vs. PR values graph](\assets\images\for-notes\pr-threshold.png)
- Or with a PR Curve, where we want to be close to the top right
  - ![PR Curve](\assets\images\for-notes\pr-curve.png)

The ROC Curve

- Similar to the PR Curve, but plots true positive rate against the false positive rate
- True positive = recall/sensitivity, false positive = 1 - TNR = specificity
- ![ROC Curve](\assets\images\for-notes\roc-curve.png)
  - want it to be as close to top left as possible

When choosing, we prefer PR when positive class is rare or when you care more about false positive than false negatives, and ROC curve otherwise.

## Multiclass Classification

Some algorithms support this intrinsically, others don't

With the ones that don't, use:
- one vs. all: train classifier for each class, pick one with the highest chance of being that class
- one v. one: train classifier for each pair of classes, compare and see most likely

## Error Analysis

- Use confusion matrix to gain insights
- Modify dataset, model, training approach,etc. based on insights


