---
layout: note
title: HoML
note_title: Hands on Machine Learning
author: Aurélien Géron 
---

# Table of Contents

1. [The Machine Learning Landscape](#the-machine-learning-landscape)
2. [End to End Machine Learning Project](#end-to-end-machine-learning-project)

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