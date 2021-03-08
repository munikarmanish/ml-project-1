#!/usr/bin/env python3

__author__ = "Manish Munikar"
__email__ = "manish.munikar@mavs.uta.edu"

import numpy as np


def sigmoid(x):
    """
    Stable implementation of the sigmoid/logistic function. It is compatible
    with numpy arrays.
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


class LogisticRegressionClassifier:
    """
    Logistic regression binary classifier, with maximum likelihood estimate as
    the performance metric and batch gradient descent as the optimizer.
    """

    def __init__(self, learning_rate=0.0001):
        """
        Set some initial model parameters.
        """
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=1000, debug=False):
        """
        Train the model on the given dataset. Initialize weights from a random
        distribution if not already initialized.

        Args:
            X:      training inputs
            y:      training target labels
            epochs: number of times to iterate over the whole training set
            debug:  if true, will print training errors
        """
        # check arguments
        X, y = np.array(X), np.array(y)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        self.n = X.shape[1]  # number of features

        # randomly initialize weights
        if not hasattr(self, 'weights'):
            self.weights = np.random.random(self.n)*0.02 - 0.01

        # start the training
        for epoch in range(epochs):
            # forward pass
            h = self.h(X)

            # weight update
            self.weights += self.learning_rate * np.matmul(y-h, X)

            # evaluate
            if debug:
                print(self.evaluate(X, y))

    def h(self, x):
        """
        The hypothesis function. Returns the probability of the given input
        belonging to the successful class (label 1).

        Args:
            x:  input sample/s

        Returns:
            the probability of `x` belonging to class 1
        """
        x = np.array(x)
        return sigmoid(np.matmul(self.weights, x.T))

    def evaluate(self, X, y):
        """
        Compute the performance metric of the model on the given dataset.

        Args:
            X:  training inputs
            y:  training target labels

        Returns:
            Maximum likelihood of the model parameters, given the dataset
        """
        h = self.h(X)
        log = np.log(h.clip(min=0.0000001))
        log_ = np.log((1-h).clip(min=0.0000001))
        return np.sum(np.dot(y, log) + np.dot(1-y, log_))


if __name__ == "__main__":
    # prepare dataset
    Dtrain = np.loadtxt('gender_train.txt')
    Dtest = np.loadtxt('gender_test.txt')
    Xtrain, ytrain = Dtrain[:, :3], Dtrain[:, 3]
    Xtest, ytest = Dtest[:, :3], Dtest[:, 3]

    # train the model
    model = LogisticRegressionClassifier(learning_rate=0.0001)
    model.fit(Xtrain, ytrain, epochs=2000)

    # get the predicted classes for the test set
    for x in Xtest.astype(int):
        y = model.h(x)
        print(f"Input = {x},   Output = {y:.3f}")
