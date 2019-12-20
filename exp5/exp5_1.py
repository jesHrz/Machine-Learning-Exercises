import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM


def load_data(file):
    X = []
    y = []
    with open(file) as f:
        for each_line in f.readlines():
            data = list(map(float, each_line.strip().split()))
            X.append(data[0:-1])
            y.append(data[-1])
    return np.array(X), np.array(y)


def plot_margin(X, y, clf):
    def f(w, x, b, c=0):    # w'x+b=c
        return (-w[0] * x - b + c) / w[1]
    X1 = X[y == 1]
    X2 = X[y == -1]
    # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c="r", label="Support vector")
    plt.plot(X1[:, 0], X1[:, 1], "+", label="Positive")
    plt.plot(X2[:, 0], X2[:, 1], "x", label="Negative")

    a0 = np.min(X)
    b0 = np.max(X)

    plt.plot([a0, b0], [f(clf.w, a0, clf.b), f(clf.w, b0, clf.b)], "k")             # w'x+b=0
    # plt.plot([a0, b0], [f(clf.w, a0, clf.b, 1), f(clf.w, b0, clf.b, 1)], "k--")     # w'x+b=1
    # plt.plot([a0, b0], [f(clf.w, a0, clf.b, -1), f(clf.w, b0, clf.b, -1)], "k--")   # w'x+b=-1

    plt.title("SVM - Linear")
    plt.axis("tight")
    plt.legend()
    plt.show()


def plot_contour(X, y, clf):
    X1 = X[y == 1]
    X2 = X[y == -1]
    # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c="r", label="Support vector")
    plt.plot(X1[:, 0], X1[:, 1], "+", label="Positive")
    plt.plot(X2[:, 0], X2[:, 1], "x", label="Negative")

    a0 = np.min(X)
    b0 = np.max(X)

    x1, x2 = np.meshgrid(np.linspace(a0, b0, 50), np.linspace(a0, b0, 50))
    x = np.array([[p1, p2] for p1, p2 in zip(np.ravel(x1), np.ravel(x2))])
    z = clf.decision_function(x).reshape(x1.shape)

    plt.contour(x1, x2, z, [0.0], colors='k', linewidths=1, origin='lower')
    # plt.contour(x1, x2, z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    # plt.contour(x1, x2, z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    plt.title("SVM - Nonlinear")
    plt.axis("tight")
    plt.legend()
    plt.show()


def linear_test():
    X_train, y_train = load_data("exp5/data/training_1.txt")
    X_test, y_test = load_data("exp5/data/test_1.txt")

    clf = SVM(C=1e-9)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct(%.2f%%)." % (correct, len(y_test), correct / len(y_test) * 100))

    plot_margin(X_train, y_train, clf)
    # plot_margin(X_test, y_test, clf)


def nonlinear_test():
    X_train, y_train = load_data("exp5/data/training_3.txt")

    clf = SVM(kernel=SVM.RBF_kernel, gamma=10000)
    clf.fit(X_train, y_train)
    plot_contour(X_train, y_train, clf)


if __name__ == "__main__":
    linear_test()
    nonlinear_test()
