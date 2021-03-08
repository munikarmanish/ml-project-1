#!/usr/bin/env python3

__author__ = "Manish Munikar"
__email__ = "manish.munikar@mavs.uta.edu"

import itertools

import numpy as np


class PolynomialLinearRegressor:
    """
    A simple implementation of a linear regression model that can learn
    polynomial functions of any order for any number of variables.
    """

    def __init__(self, nvar, order):
        """
        Initialize the model by setting up the number of features and
        the polynomial exponents.

        Args:
            nvar:   number of input variables
            order:  order of polynomial
        """
        # check input arguments
        assert order > 0, "invalid order"

        self.nvar = nvar
        self.order = order

        # prepare the polynomial exponents
        self.exponents = self.get_exponents(nvar, order)
        self.n = len(self.exponents)  # number of features

    def get_exponents(self, nvar, order):
        """
        Return the list of polynomial exponents for the given number of
        variables and the given order.

        Args:
            nvar:   number of input variables
            order:  order of polynomial

        Returns:
            list of polynomial exponents
        """
        f = []
        for o in range(0, order+1):
            f.extend(
                [l for l in itertools.product(range(o+1), repeat=nvar)
                 if sum(l) == o]
            )
        return np.array(f)

    def features(self, x):
        """
        Convert input variables into model features.

        Args:
            x:  input sample or list of input samples

        Returns:
            list of features for each input sample
        """
        # check arguments
        assert x.shape[-1] == self.nvar
        assert x.ndim in [1, 2]

        if x.ndim == 1:
            return np.power(x, self.exponents).prod(axis=1)
        elif x.ndim == 2:
            return np.power(
                np.expand_dims(x, axis=1), self.exponents
            ).prod(axis=2)

    def fit(self, X, y):
        """
        Fit the model to the given dataset, using analytic solution.

        Args:
            X:  training inputs
            y:  training outputs
        """
        # check arguments
        X, y = np.array(X), np.array(y)
        assert X.ndim == 2, "invalid input dimension"
        assert y.ndim == 1, "invalid output dimension"
        assert X.shape[1] == self.nvar, "invalid input size"
        assert X.shape[0] == y.shape[0], "unequal input/output"

        X, y = np.matrix(self.features(X)), np.matrix(y).T
        self.weights = np.array(np.linalg.inv(X.T * X) * X.T * y).flatten()

    def predict(self, x):
        """
        Predict the output for the given input.

        Args:
            x:  input sample or a list of input samples

        Returns:
            predicted output for each input sample
        """
        # check arguments
        assert hasattr(self, "weights"), "model is not trained"
        x = np.array(x)
        assert x.shape[-1] == self.nvar
        assert x.ndim in [1, 2]

        return np.dot(self.features(x), self.weights)

    def evaluate(self, X, y):
        """
        Compute the model's error (sum of squared error) for the given
        test dataset.

        Args:
            X:  training inputs
            y:  training outputs

        Returns:
            the total error
        """
        # check arguments
        X, y = np.array(X), np.array(y)
        assert X.ndim == 2, "invalid input dimension"
        assert y.ndim == 1, "invalid output dimension"
        assert X.shape[1] == self.nvar, "invalid input size"
        assert X.shape[0] == y.shape[0], "unequal input/output"

        # get the predictions
        ypred = self.predict(X)

        # calculate error of prediction vs ground truth
        return np.sum(np.power(ypred - y, 2))


if __name__ == "__main__":
    # prepare datasets
    Dtrain = np.loadtxt("PolyTrain.txt")
    Dtest = np.loadtxt("PolyTest.txt")
    Xtrain, ytrain = Dtrain[:, :2], Dtrain[:, 2]
    Xtest, ytest = Dtest[:, :2], Dtest[:, 2]

    # try different order
    for order in range(1, 5):
        # feed the training set to the model
        print(f"Fitting a polynomial of order {order}")
        model = PolynomialLinearRegressor(nvar=Xtrain.shape[-1], order=order)
        model.fit(Xtrain, ytrain)

        # evaluate train/test errors
        train_error = model.evaluate(Xtrain, ytrain)
        test_error = model.evaluate(Xtest, ytest)
        print(f"\tTrain error: {train_error}")
        print(f"\tTest error:  {test_error}")
