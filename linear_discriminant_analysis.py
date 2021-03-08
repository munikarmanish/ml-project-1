#!/usr/bin/env python3

__author__ = "Manish Munikar"
__email__ = "manish.munikar@mavs.uta.edu"

from collections import defaultdict

import numpy as np


class LinearDiscriminatAnalysis:
    """
    Linear discriminant analysis classifier.
    """

    def fit(self, X, y):
        """
        Fit the model to the given dataset.

        Args:
            X:  training inputs
            y:  training ouputs
        """
        # convert to ndarray
        X, y = np.array(X), np.array(y)

        # check dimensions
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        self.m = X.shape[0]  # number of training samples
        self.n = X.shape[1]  # number of features
        self.classes = set(y)
        self.k = len(self.classes)

        # calculate priors
        self.priors = {c: y[y == c].size / self.m for c in self.classes}

        # calculate means
        self.means = {c: X[y == c].mean(axis=0) for c in self.classes}

        # calculate covariance matrix
        means = np.array([self.means[c] for c in y])
        self.cov = (X-means).T @ (X-means) / self.m
        self.icov = np.linalg.inv(self.cov)  # inverse of the covariance matrix

    def predict(self, x):
        """
        Predict the most probable class for the given input.

        Args:
            x:  input sample to predict the class of

        Returns:
            the most probable class
        """
        # convert to numpy array
        x = np.array(x)

        # check dimensions
        assert x.ndim == 1
        assert x.shape[0] == self.n

        # find log likelihood of each class
        best_likelihood = -float('inf')
        best_class = None
        for c in self.classes:
            diff = x - self.means[c]
            p = -diff @ self.icov @ diff.T
            if p > best_likelihood:
                best_likelihood = p
                best_class = c

        return best_class


if __name__ == "__main__":
    # prepare dataset
    Dtrain = np.loadtxt('gender_train.txt').astype(int)
    Dtest = np.loadtxt('gender_test.txt').astype(int)
    Xtrain, ytrain = Dtrain[:, :3], Dtrain[:, 3]
    Xtest, ytest = Dtest[:, :3], Dtest[:, 3]

    # train the model
    model = LinearDiscriminatAnalysis()
    model.fit(Xtrain, ytrain)

    # get the predicted classes for the test set
    for x in Xtest:
        y = model.predict(x)
        print(f"Input = {x},   Output = {y}")
