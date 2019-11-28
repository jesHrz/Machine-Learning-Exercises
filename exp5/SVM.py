import numpy as np
import cvxopt


class SVM:
    @staticmethod
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    @staticmethod
    def RBF_kernel(x1, x2, gamma):
        return np.exp(-gamma*np.dot(x1-x2, x1-x2))

    def __init__(self, kernel=None, C=None, **kargcs):
        self.kernel = SVM.linear_kernel if kernel is None else kernel
        self.C = C if C is None else float(C)
        self.kargcs = kargcs

    def fit(self, X, y):
        m, n = X.shape
        print("overall %d training datas" % m)

        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = self.kernel(X[i], X[j], **self.kargcs)

        P = cvxopt.matrix(np.outer(y, y) * K)
        # P = cvxopt.matrix(np.dot(np.dot(y, y.transpose()), K))
        q = cvxopt.matrix(np.ones(m) * -1)
        # sigma(a*y)=0
        A = cvxopt.matrix(y, (1, m))
        b = cvxopt.matrix(0.0)

        if self.C is None or self.C == 0:  # hard-margin
            # a >= 0
            G = cvxopt.matrix(np.eye(m) * -1)
            h = cvxopt.matrix(np.zeros(m))
        else:   # soft-margin
            # a >= 0 && a <= c
            p1 = np.eye(m) * -1
            p2 = np.eye(m)
            G = cvxopt.matrix(np.vstack((p1, p2)))
            p1 = np.zeros(m)
            p2 = np.ones(m) * self.C
            h = cvxopt.matrix(np.hstack((p1, p2)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])
        # 非0的a对应支持向量
        sv = a > 0
        # 支持向量对应下标
        self.support_ = np.arange(len(a))[sv]
        self.a = a[sv]
        self.support_vectors_ = X[sv]
        self.support_vectors_y = y[sv]
        print("%d support vectors out of %d points." % (len(self.a), m))

        if self.kernel == SVM.linear_kernel:    # 线性可分
            self.w = np.zeros(n)
            for i in range(len(self.a)):
                self.w += self.a[i] * self.support_vectors_y[i] * self.support_vectors_[i]
            self.b = 0
            for i in range(len(self.a)):
                self.b += self.support_vectors_y[i] - np.dot(self.w, self.support_vectors_[i])
            self.b /= len(self.a)
        else:   # 非线性
            self.w = None
            self.b = 0
            for i in range(len(self.a)):
                self.b += self.support_vectors_y[i]
                self.b -= np.sum(self.a * self.support_vectors_y * K[self.support_[i], sv])
            self.b /= len(self.a)

    def decision_function(self, X):
        if self.w is not None:  # 线性可分
            return np.dot(X, self.w) + self.b

        y_decision = np.zeros(len(X))
        for j in range(len(X)):
            for i in range(len(self.a)):
                y_decision[j] += self.a[i] * self.support_vectors_y[i] * self.kernel(X[j], self.support_vectors_[i], **self.kargcs)
        return y_decision + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))
