import numpy as np
import cvxopt
import matplotlib.pyplot as plt


def load_data(file):
    X = []
    y = []
    with open(file) as f:
        for each_line in f.readlines():
            data = list(map(float, each_line.strip().split()))
            X.append(data[0:-1])
            y.append(data[-1])
    return np.array(X), np.array(y)


class SVM:
    @staticmethod
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    def __init__(self, kernel=None, C=None):
        self.kernel = SVM.linear_kernel if kernel is None else kernel
        self.C = C if C is None else float(C)

    def fit(self, X, y):
        m, n = X.shape

        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(m) * -1)
        # sigma(a*y)=0
        A = cvxopt.matrix(y, (1, m))
        b = cvxopt.matrix(0.0)

        if self.C is None:  # hard-margin
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
                y_predict[j] += self.a[i] * self.sv_y[i] * self.kernel(X[j], self.sv[i])
        return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


def plot_margin(X, y, clf):
    def f(w, x, b, c=0):    # w'x+b=c
        return (-w[0] * x - b + c) / w[1]
    X1 = X[y == 1]
    X2 = X[y == -1]
    plt.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="r")
    plt.plot(X1[:, 0], X1[:, 1], "+", label="Positive")
    plt.plot(X2[:, 0], X2[:, 1], "o", label="Negative")

    a0 = np.min(X)
    b0 = np.max(X)

    # w'x+b=0
    a1 = f(clf.w, a0, clf.b)
    b1 = f(clf.w, b0, clf.b)
    plt.plot([a0, b0], [a1, b1], "k", label="Hyperplane")

    # w'x+b=1
    a1 = f(clf.w, a0, clf.b, 1)
    b1 = f(clf.w, b0, clf.b, 1)
    plt.plot([a0, b0], [a1, b1], "k--")

    # w'x+b=-1
    a1 = f(clf.w, a0, clf.b, -1)
    b1 = f(clf.w, b0, clf.b, -1)
    plt.plot([a0, b0], [a1, b1], "k--")

    plt.title("SVM")
    plt.axis("tight")
    plt.legend()
    plt.show()


def SVM_test(X_train, y_train, X_test=None, y_test=None, kernel=None, C=None):
    clf = SVM(kernel=kernel, C=C)
    clf.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        print("Test data found.")
        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct(%.2f%%)." % (correct, len(y_test), correct / len(y_test) * 100))

    return clf


if __name__ == "__main__":
    X_train, y_train = load_data("exp5/data/training_2.txt")
    X_test, y_test = load_data("exp5/data/test_1.txt")
    clf = SVM_test(X_train, y_train)
    plot_margin(X_train, y_train, clf)
