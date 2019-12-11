import numpy as np


class Logistic_Regression:
    def __init__(self):
        self.X_ = None
        self.y_ = None
        self.w_ = None
        self.cost_ = None

    def fit(self, X, y, epsilon=1e-6):
        m, n = np.shape(X)
        cost = 0
        self.X_ = X
        self.y_ = y
        self.w_ = np.zeros((n, 1))
        self.cost_ = []
        while True:
            H = np.zeros((n, n))
            for i in range(m):
                h = Logistic_Regression.sigmoid(np.dot(self.X_[i], self.w_))
                H += (h * (1 - h)) * np.dot(self.X_[i].reshape(n, -1), self.X_[i].reshape(-1, n)) 
            err = self.y_ - Logistic_Regression.sigmoid(np.dot(self.X_, self.w_))
            self.w_ += np.dot(np.dot(np.linalg.inv(H), self.X_.T), err)
            pre_cost, cost = cost, Logistic_Regression.cost_value(self.X_, self.y_, self.w_)
            self.cost_.append(cost)
            if abs(cost - pre_cost) < epsilon:
                break

    def predict(self, X):
        return Logistic_Regression.sigmoid(np.dot(X, self.w_))

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def cost_value(X, y, weights):
        ret = 0
        m = np.shape(X)[0]
        for i in range(m):
            h = Logistic_Regression.sigmoid(np.dot(X[i], weights))
            ret += -y[i] * np.log(h) - (1 - y[i]) * np.log(1 - h)
        return ret[0] / m
