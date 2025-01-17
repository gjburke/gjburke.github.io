---
layout: note
title: HoML
note_title: Hands on Machine Learning
author: Aurélien Géron 
---

# Forward (on the notes)

Just for some conext for someone who isn't me...

These are notes that are copied from my hand-written notes that I've already taken. I'm not a big fan of computer notes, especially because I like to flex my organization and things on tge page when writing, in a way that's too cumbersome to replicate on screen. So please bear with me while I try and find a nice way to structure everything with markdown. I want these notes so I can access them anywhere. They are a more stripped version of my hand-written notes, so if these are sparse (especially of equations) just know that they are probably written elsewhere, it was probably just annoying to type it.

Alright, now on with the notes!

# Table of Contents

1. [The Machine Learning Landscape](#the-machine-learning-landscape)
2. [End to End Machine Learning Project](#end-to-end-machine-learning-project)
3. [Classification](#classification)
4. [Training Models](#training-models)
5. [Support Vector Machines](#support-vector-machines)
6. [Decision Trees](#decision-trees)
7. [Dimensionality Reduction](#dimensionality-reduction)
8. [Unsupervised Learning Techniques](#unsupervised-learning-techniques)

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

# Training Models

## Linear Regression

Basic Idea:

y_hat = t0 + t1\*x1 + t2\*x2 + ... + tn\*xn

where...
- yhat = predicted value
- t = vector of model parameters
- x = vector of features
- n = number of features

A vectorized version: yhat = h_t(x) = t * x = t.T @ x

We want to minimize the cost between this linear model and the datapoints. For now we will use the Mean Squared Error (MSE) of the model and datapoints. This is the sum of the squared error of the predicted and actual values of the data points, divided by the number of data points (m).

MSE(x, h_t) = (1/m) * sum((h_t(x) - y)^2)

Now how do we go about optimizing these parameters?

### The Normal Equation

There is a closed form for this solution, to find the values of theta (t) that minimize the cost function. Here it is:

optimal_theta = (X.T @ X)^(-1) @ X.T @ y <- can be calculated quicker with SVD

### Gradient Descent

This closed form can be costly, in time and memory, and sometimes is not possible to calculate!

Our solution is gradient descent!

- iterative approach to minimizing the cost function
- measures the gradient of the cost function, moves in the direction of the descending gradient
- as it moves down, it (hopefully) approaches a minimum cost value

An important parameter here is the learning rate

- determines how "fast" you move down the gradient
- too fast/high, never can reach a minimum, "bouncing around"
- too slow/low, never reaches min in time and may fall into local minima

Other properties of GD include

- Susceptible to local minima, if cost function has them
- works within parameter space, so more parameters means more time and complexity because there is more space to search

### Batch Gradient Descent

This is an implementation of gradient descent that uses the gradient (woah) of the cost function. This means you need to compute the partial derivatives of the cost function.

partial_MSE_j(t) = (2/m)\*sum(t.T @ x - y)\*x_j or partial_MSE(t) = (2/m) * X.T * (X*t - y)

Main problem with this is that it's very computationally intense to compute gradients at every step. How can we help this?

### Stochastic Gradient Step

This, for every step, just picks a random instance in the training set and adjusts the parameters to that single instance. 

Properties

- much less regular than other methods
- it is training on much less data
- randomness can help jump out of local minima, but also means you can't really converge
  - can fix this inability to converge with a learning schedule, slow tapering of learning rate

### Mini-batch Gradient Descent

This one is a healthy cross between both stochastic and batch gradient descent. It does batch gradient descent on a smaller sets of parameters each step.

![Table Comparing Linear Regression Algroithms](\assets\images\for-notes\regression-table.png)

## Polynomial Regression

- Add powers of your features as features, then use linear regression algorithms (reflect those powers in your final predictor's equation, matching the right parameters)
- Can find relationships between features when there are multiple features
- Watch out! Too high of a degree can likely overfit your model

Can look at learning curves to test over/under fitting. They plot the training set size and resulting cost of the trained model.

- Underfitting curve: both (validation and training) cost values have reached a plateau, but they are close and pretty high 
  - ![Underfitting Curve](\assets\images\for-notes\underfit-curve.png)
- Overfitting curve: the training cost value is much much lower than the validation cost value, there is a big gap
  - ![Overfitting Curve](\assets\images\for-notes\overfit-curve.png)

The Bias Tradeoff - generalization error can be expressed as a sum of three errors:

- Bias - due to wrong assumptions - likely underfitting
- Variance - excessive sensitivity to small variations - likely overfitting
- Irreducible factor - noisiness in the data itself - can be addressed at the data level as well

The genral rule is that you must trade bias and variance. With the same/similar modal, if you make a change that increases bias, it'll descrease variance, and vice versa.

## Regularized Linear Models

Regularization is used to add more bias/lower the variance of a model. Good to have a little in general, add more when a model is overfitting

### Ridge Regression

Regularization term added to the cost function is the sum of the square of all the parameter values: alpha * sum(t^2)

- meant to force minimization of parameter values
- hyperparameter alpha controls how much regularization there is
- good default regularization technique 

### Lasso Regression

Another regularization term, but uses the one-norm instead of squares: alpha * sum(t)

- tends to completely eliminate the weights of the least important features
- usually performs feature selection, outputs a spare model (few nonzero weights)
- use this if features aren't needed, but probably prefer elastic net because its less durastic

### Elastic Net

Middle gorund between ridge and lasso, with a hyperparameter to control the ratio of each. Basically has both equations added on, but with ratio parameter to controls how much one or the other contributes.

- usually use this is features aren't needed, usually preferred over lasso because it's less durastic

## Logistic Regression

- commonly used to tell probability that something is in a certain class
- outputs the logistic of a result:

>phat = h_t(x) = sigmoid(x.T @ t) where sigmoid(t) = 1/(1+exp(-t))

- with the output of the result, you then put some threshold of probability to determine class

Training and cost function

>For the cost function, if y=1: cost = -log(phat) and if y=0: cost = -log(1-phat)

>This can be summed up with the log loss function - not writing that :) - of which there is no known closed form, so we have to use partials (not writing those either)

Decision boundary - where the no and yes meet, specific value/probability from sigmoid

Softmax regression

- Used for multiple classes
- Computes the softmax for each class, takes greatest probability (not writing equations)
- When you take the argmax of the softmax, that is your prediction

# Support Vector Machines

An SVM is capable of doing linear or nonlinear classification, regression, and even outlier detection. They are well-suited for classification of complex, but small or medium sized datasets.

## Linear SVM Classification

![Graphical representation of two-class svm model](\assets\images\for-notes\svm-graph.png)

- think of it as "fitting the widest possible street" between classes
- the decision boundary is fully determined by the points on teh boundary, these are the "support vectors"

### Hard Margin Classification

- the bounds are set by edge support vectors
- no points inside the bounds, NONE
- this has some problems:
  - only works when the data is linearly seperable
  - very sensitive to outliers

### Soft Margin Classification

- keep a good balance between keeping the street as large as possible and limiting margin violations
- hyperparameter "C" controls the ratio, higher C means a wider street but more violations allowed

## Nonlinear SVM CLassification

- could add more polynomial features and classify linearly
- but low poly degree can't handle complexity, and too high degree makes the model too slow

### Polynomial Kernel

- to solve this, we use a kernel trick
- can add high order polynomials without actually adding the features
- you set a degree d, hyperparameter C as usual, and then another hyperparameter "coef0" which controls how much the model is influenced by high vs low degree polynomials

### Adding Similarity Features

- is a way to tackle nonlinear problems as well
- you add features using a similarity factor - measures how much each intance resembled a landmark instance
  - an example of this may be the Gaussian Radial Basis Function (RBF)
- basically, computer these new features based on the distance from landmark, and it will work with linear classification

### Gaussian RBF Kernel

- similar to polynomial features, you can also make the similarity function into a kernel
- adds a hyperparameter gamma that controls the bell curve of the RBF

## Choosing which kernel

1. Try the linear kernel, especially if the training data size is large or if there is lots of features
2. Try GRBF if the dataset or features are not too large to be a limiting factor
3. Then you can go to others, (like polynomial)

![Table of different SVM functions and deatils about them](\assets\images\for-notes\svm-table.png)

## SVM Regression

Instead of tryin gto fit the largest road with least margin violations, you fit as many points as possible within the street with the smallest width

# Decision Trees

These guys can do classification, regression, and even multi-output classification. They are capable of fitting complex datasets, and are also a fundemental part of Random Forests

### Making Predictions

1. Start at the root Node
2. Check teh boundary, move left/right
4. Repeat 2 until on leaf
4. Check class/value on the

### Node Properties 

- Samples: how many training instances it applies to in general
- Value: how amny training instances of each class it applies to (separated by class)
- Impurity Measure: gini impurity or entropy - measure of how many of a different class is in the ere 

### The CART Training Algorithm

Scikit-learn uses the Classification and Regression Tree algorithm

1. Splits the training set into two subsets using a single feature k and threshold t_k
2. Searches for the pair (k, t_k) that produces the purest subsets
3. Once it successfully splits the training in two, it recursively splits until the max depth is reached or if a split won't increase purity of the classes

Note: this is a greedy algorithm, so it doesn't always get the most optimal way of splitting. But, the solution for the most optimal is a P-NP problem, so you can't really solve it. Just know that this is pretty simple, and there are probably other algorithms,

### Computational Complexity

- note: m is number of instances, n is number of features
- predictions are fast, even with large datasets - big O of log(m)
- training compares all features on all samples at each node - big O og n*mlog(m) 

### Regularization Hyperparameters

- decision trees will probably overift if unconstrained (parameters are not known at the start, only through the runtime!)
- need to restrict the freedom of parameter choices during training

Parameters:
- max depth
- max samples split
- min samples for a leaf
- min weight for a fraction of a leaf
- max leaf nodes
- max features

Also can do pruning as well

### Regression

This can also be used for regression, but it gives a kind of step-wise function looking thing that I think wouldn't be too useful

### Instability

- this type of model is particularly unstable, especially to data transformations
- for example: the boundaries with decision trees are orthogonal, so if you rotate a dataset, you may get a jagged edge
  - ![Jagged decision tree](\assets\images\for-notes\jagged-graph.png)
- it is also sensitive to small variations in training data

# Ensemble Learning and Random Forests

Technique where you take a group of predictors (an ensamble) and make predictions by combining theirs. An example of this is a Random Forest, which is an ensembel of decision trees (we'll get to it more later).

### Voting Classifiers

You make an ensemble by training a few different types of classifiers on a single problem, training set

- hard-voting: prediction is the class that gets the most votes (majority of classifier's output)
- soft-voting: predict the class with the highest aggregate probability (needs models that output probability)

### Bagging and Pasting

You make an ensamble through using the algorithm, but only training each on subsets of the training data

- bagging: get samples with replacement
- pasting: get the samples without replacement

Then the aggregation function for all the models is usually a statistical mode (most common answer) for classification, and a mean for regression. The net result is a similar bias, and less overall variance

### Random Patches and Trandom Subspaces

- Random Patches: random sampling of instances and features
- Random Subspaces: random sampling of features

Trades a bit more bias for lower variance

### Random Forests

These are an ensemble of decision trees, usually using bagging (with the max samples set to the size of the training set). Basically, creates and trains a large amount of decision trees (with a little more randomness in their generation) and takes the most common result for classificaiton.

- generally have all the hyperparameters of a decision tree and bagging algorithms
- also introduces more randomness by only sampling from a subset of features for each node split for each tree

Extra Trees

- make trainign even more random by setting random thresholds for each node on a tree
- trades more bias for lower variance

Random forests make it easy to measure the relative importance of each feature by looking at which impact impurity the most. They are also just really powerful classifier models.

### Boosting

The idea is to train predictors sequentially, each trying to correct the one before.

AdaBoost

- pays more attension to the training instances that were misclassified (adjusts the relative weights of data points missed)
- in teh end, it aggregates of all the models' output (with weighting depending on each model's accuracy)

Gradient Boosting

- next model tries to fit the residual errors of the previous
- make predictions by adding the predictions of all models/predictors
- can also subsample the training instances, making it stochastic

### Stacking

This is where you train a model to evaluate the predictions of an ensemble of models

1. Split the set
2. Train the ensemble of models on one hal
3. Once those are trained, feed the other half of the data through those models
4. Train the aggregator model on these outputs

You can also stack the stacked with even more to stack!

# Dimensionality Reduction

### The Curse of Dimensionality

- more dimensions (a ton of them) leads to slow training, harder to find a good solution
- more/higher dimensions -> more sparse dataset, training instances are further from each other
  - leads to higher risk of overfitting
- need exponentially more data for more features to mitigate this risk

## Dimensionality Reduction

- you usually are able to reduce the number of features significantly (whether it be manually, with algorithms, etc)
- models can perform better with less, but well selected, relevant features
- its also useful for data visualization (DataViz), which is especially important when presenting to non-techinical stakeholders

### Main Approaches:

- Projection - project data on a lower-dimensional plane
  - not the best approach when the data plane is not linear
- Manifold Learning - models a manifold (like a folded or twisted plane) that the data lies on
  - assumes the data lies on such a manifold and that projecting it onto the manifold would simplify the problem somehow

### PCA

1. identifies the hyperplane that lies closest to the data, then projects the data onto it
2. selects the axis that preserves the greatest variance, then second axis is orthogonal that that with preserving the second greatest, etc.
  - the explained variance ratio shows how much variance each component is responsible for
  - these axes can be found with the SVD (like from numerical methods!)
3. we can then project the instances down to d-Dimensions with the principal component vectors
  - projecting it back with the reverse operation will have a recontruction error, but if done well will not lose much information
4. to determine how many dimensions we want, we can use teh explained variance ratio and add dimensions until at a certain threshold of cumulative explained variance

Incremental PCA

- since the usual PCA needs all the data in memory, that may be a limiting factor
- incrememntal PCA splits the trainign set into multiple mini-batches and feeds it one batch at a time

Kernel PCA

- used to do complex, nonlinear projections
- good a preserving clusters after projection, or sometimes unrolling datasets in a twisted manifold
- unsupervised, so you gotta search for the right kernel (grid, or based on reconstruction error - with extra steps)

## Local Linear Embedding (LLE)

- nonlinear dimensionality reduction
- manifold technique, not reliant on projections

1. measures how each instance relates to its closest neighbors
2. looks for a low-dimensional representation where these relationships are preserved

## Others

- Multidimensional Scaling
- Isomap
- t-Distributed Stochastic Neighbor Embedding
- Linear Descriminant Algorithms

# Unsupervised Learning Techniques

The vast majoirty of data is unlabelled, good labelled data is hard to come by or generate. We can take advantage of this unlabelled data with unsupervised (and semisupervised) techniques. 

Main algortihms include:

- Clustering: group similar instances
- Anomaly detection: detect "abnormal" instances
- Denstiy estimation: estimating the probability density function

## Clustering

- identifying similar instances, grouping them together
- has a variety of applications: segmentation, data analysis, dimension reduction, anomaly detection, semi-supervised learning, search engines/similarity search
- what a "cluster" is depends on the context, different algorithms give different clusters

### K-means (loyd-Forgy)

- simple algorithm, cluster simply and efficiently
- but you need to specify the number of clusters beforehand (not always obvious!)
- uses hard clustering: only one cluster per instance
- doesn't do well with different diameter blobs

The algorithm

1. Assign centroids (centers of the cluster) randomly
2. Label instances, update centroids, repeat
- label instances to the closest centroid
- update the centroid to the new "center" of all the labelled instances

Optimizations

- when evaluating which model is best, can use the inertia to see which is best
  - intertia is the sum of the mean squared distances of each instance to its centroid
- better centroid initialization: make sure that they're spread out from one another
  - can also run multiple times with different random initializations
- the algorithm was made better by doing some smart storage of certain distances, means less recalculation was required, leads to quicker algorithm
- mini-batch k-means that trains on small batches of data, leads to better memory management

Selecting a good centroid amount

- sometimes its not obvious
- can't just use inertia
  - if you do, you want to use a value around the "elbow" of the graph of centroid amount vs resulting inertia
- can use the silouhette coefficient/score, which ranges from 1 (point close to center) to 0 (on the border of the cluster) to -1 (on the wrong side, in another cluster)

Limits

- not good with ellipses, varying sizes, non-spherical clusters

Use cases

- image segmentation
- preprocessing, dimension reduction
- semi-supervised learning

### DBScan

- defines clusters as regions of high instance density
- works well if clusters are dnese enough, and well-seperated by low-density regions

The algortihm:

1. For each instance, count the number of instances within some small distance (epsilon) from it
- the instances within epsilon from another instance are within that instance's neighborhood
2. If an instance has at least min_sample instances in its neighborhood, then its a core instance

- all instances in the neighborhood of a core instance are in teh same cluster, and a chain of cores creates one big cluster
- any instance that is not in a core's neighborhood is an outlier

Important note:

- this algorithm cannot predict which cluster something will belong to
- instead, now that you have the classes/clusters, you can train a classification algorithm with the labels formed by the clusters

Overall this is very powerful

- capable of identifying a number of clusters, even with funky shapes
- robust to outliers
- just two hyperparameters (epsilon and min_samples)
- but... if the density varies across the structures or is just too low in general, then this doesn't work as well

## Guassian Mixtures

This is a probabalistic and generative model that assumes the data is a mixture of several gaussian distributions. It works on ellipsoid-like shapes of data, but not much else

How it works (I think) <- this explanation isn't great ngl

- you choose the number of "clusters"/distributions, the algortihm choosing starting centers, means, covariance, and mixing coefficients for each
- the model fits the means, covariences, and mixing coefficients to the dataset
  - it first evaluates the probability that the instance belongs to each distribution
  - then it updates the parameters to maximize the likelihood that the instances are "under" the model
- within that, it has methods with the eval and update functions to keep the distributions to just fit the data, not go overboard

But you can think if this as a k-means that find the cluster's centers, as well as the size, shape, orientation, and relative weights. Because its made from distributions, it uses soft-cluster assignments, where you can measure the likelihood it belonds to any cluster.

Properties

- struggles on many dimensions, clusters, or few instances
- can have different coveriance/distribution types

Selecting Number of Distributions

- similar to k-means
- use theoretical information criterion
  - BIC or AIC
- select at elbow or minimum

Uses

- anomaly detection: high vs low density (where low desnity are outliers)
  - need effective density threshold
- novelty detection: anomaly detection but wihtout any outliers in the training set

### Bayseian-Gaussian Mixture

Basically a similar model, nit ow searches for the optimal number of clusters as well. The cluster parameters are no longer fixed, so some can go to essentially zero


