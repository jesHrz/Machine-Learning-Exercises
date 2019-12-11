import numpy as np


class Linear_Regression:

    def __init__(self):
        self.cost_ = None
        self.theta_ = None

    def fit(self, X, y, learning_rate=0.07, max_iteration=1500):
        m, n = np.shape(X)
        n += 1
        assert np.shape(y)[0] == m
        self.X_ = np.hstack((np.ones((m, 1)), X))
        self.y_ = y
        self.cost_ = np.zeros((max_iteration, 1))
        self.theta_ = np.zeros((n, 1))
        for i in range(max_iteration):
            self.cost_[i] = Linear_Regression.cost_value(self.X_, self.y_, self.theta_)
            self.theta_ -= (learning_rate / m) * np.dot(self.X_.T, np.dot(self.X_, self.theta_) - self.y_)

    def predict(self, X):
        X = np.hstack((np.ones((len(X), 1)), X))
        return np.dot(X, self.theta_)

    @staticmethod
    def cost_value(X, y, theta):
        hx = np.dot(X, theta)
        return np.dot((hx - y).T, hx - y) / (2 * np.shape(X)[0])
