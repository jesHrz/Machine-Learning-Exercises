import numpy as np
import cvxopt


class SVM:
    @staticmethod
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    @staticmethod
    def RBF_kernel(x1, x2, gama):
        return np.exp(-gama*np.dot(x1-x2, x1-x2))

    def __init__(self, kernel=None, C=None, **kargcs):
        self.kernel = SVM.linear_kernel if kernel is None else kernel
        self.C = C if C is None else float(C)
        self.kargcs = kargcs

    def fit(self, X, y):
        m, n = X.shape

        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = self.kernel(X[i], X[j], **self.kargcs)

        P = cvxopt.matrix(np.outer(y, y) * K)
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
        sv = a > 1e-5
        # 支持向量对应下标
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), m))

        if self.kernel == SVM.linear_kernel:    # 线性可分
            self.w = np.zeros(n)
            for i in range(len(self.a)):
                self.w += self.a[i] * self.sv_y[i] * self.sv[i]
            self.b = 0
            for i in range(len(self.a)):
                self.b += self.sv_y[i] - np.dot(self.w, self.sv[i])
            self.b /= len(self.a)
        else:   # 非线性
            self.w = None
            self.b = 0
            for i in range(len(self.a)):
                self.b += self.sv_y[i]
                self.b -= np.sum(self.a * self.sv_y * K[ind[i], sv])
            self.b /= len(self.a)

    def project(self, X):
        if self.w is not None:  # 线性可分
            return np.dot(X, self.w) + self.b

        y_predict = np.zeros(len(X))
        for j in range(len(X)):
            for i in range(len(self.a)):
                y_predict[j] += self.a[i] * self.sv_y[i] * self.kernel(X[j], self.sv[i], **self.kargcs)
        return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))
