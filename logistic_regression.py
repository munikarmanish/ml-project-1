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

    def features(self, X):
        """
        Get all input features (including the constant/bias input).
        """
        X = np.array(X)
        # add the bias term
        if X.ndim == 1:
            return np.hstack([1, X])
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y, learning_rate=0.001, epochs=1000, debug=False):
        """
        Train the model on the given dataset. Initialize weights from a random
        distribution if not already initialized.

        Args:
            X:  training inputs
            y:  training target labels
            learning_rate:  the learning rate
            epochs:  number of times to iterate over the whole training set
            debug:  if true, will print training errors
        """
        # check arguments
        X, y = np.array(X), np.array(y)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

        X = self.features(X)
        self.n = X.shape[1]  # number of features

        # randomly initialize weights
        if not hasattr(self, 'weights'):
            self.weights = np.random.random(self.n) * 0.2 - 0.1

        # start the training
        for epoch in range(epochs):
            # forward pass
            h = self.h(X)

            # weight update
            self.weights += learning_rate * ((y-h) @ X)

            # print error for debugging
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
        # check if the model has weights
        assert hasattr(self, "weights"), "model is not trained"

        # add the bias term
        if x.shape[-1] < self.n:
            x = self.features(x)

        return sigmoid(np.matmul(self.weights, x.T))

    def evaluate(self, X, y):
        """
        Compute the performance metric of the model on the given dataset.

        Args:
            X:  training inputs
            y:  training target labels

        Returns:
            log likelihood of the model parameters, given the dataset
        """
        h = self.h(X)
        log = np.log(h.clip(min=0.00000001))
        log_ = np.log((1-h).clip(min=0.00000001))
        return np.mean(-y*log - (1-y)*log_)


if __name__ == "__main__":
    # prepare dataset
    Dtrain = np.loadtxt('gender_train.txt')
    Dtest = np.loadtxt('gender_test.txt')
    Xtrain, ytrain = Dtrain[:, :3], Dtrain[:, 3]
    Xtest, ytest = Dtest[:, :3], Dtest[:, 3]

    # train the model (decrease learning rate after every 100 epochs)
    model = LogisticRegressionClassifier()
    alpha = 0.0001
    for i in range(10):
        model.fit(Xtrain, ytrain, epochs=100, learning_rate=alpha)
        alpha /= 5

    # get the predicted classes for the test set
    for x in Xtest.astype(int):
        y = model.h(x)
        print(f"Input = {x},   Output = {y:.3f}")
