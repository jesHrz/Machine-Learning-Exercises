import numpy as np
import matplotlib.pyplot as plt

def sig(x): return 1. / (1. + np.exp(-x))


def load_data():
    feature = []
    with open("exp3/data/ex3Logx.dat") as f:
        for each_line in f.readlines():
            feature_tmp = [1]
            for data in each_line.strip().split(','):
                feature_tmp.append(float(data))
            feature.append(feature_tmp)
    label = []
    with open("exp3/data/ex3Logy.dat") as f:
        for each_line in f.readlines():
            label_tmp = []
            for data in each_line.strip().split():
                label_tmp.append(float(data))
            label.append(label_tmp)
    return np.mat(feature), np.mat(label)


def map_feature(feature1, feature2):
    x = []
    for i in range(7):
        for j in range(i + 1):
            x.append((feature1 ** (i - j)) * (feature2 ** j))
    return x


def cost(feature, label, theta, lamb):
    ret = 0
    m, n = np.shape(feature)
    for i in range(m):
        h = sig(feature[i] * theta)
        ret += -label[i, 0] * np.log(h) - (1 - label[i, 0]) * np.log(1 - h)
    for i in range(1, n):
        ret += lamb / 2 * (theta[i, 0] ** 2)
    return ret / m


def gradient(feature, label, theta, lamb):
    m, n = np.shape(feature)
    err = feature.T * (sig(feature * theta) - label)
    for j in range(1, n):
        err[j, 0] += lamb * theta[j, 0]
    return err / m


def hessian(feature, label, theta, lamb):
    m, n = np.shape(feature)
    H = np.mat(np.zeros((n, n)))
    for i in range(m):
        h = sig(feature[i] * theta)[0, 0]
        H += (h * (1 - h)) * feature[i].T * feature[i]
    E = np.eye(n)
    E[0, 0] = 0
    H += lamb * E
    return H / m


def newton(feature, label, lamb, epsilon=1e-6):
    val = 0
    n = np.shape(feature)[1]
    w = np.mat(np.zeros((n, 1)))
    while True:
        H = hessian(feature, label, w, lamb)
        dJ = gradient(feature, label, w, lamb)
        w = w - H.I * dJ
        last, val = val, cost(feature, label, w, lamb)
        print(val)
        if abs(last - val) <= epsilon:
            break
    return w


def plt_logistic(feature, label, theta):
    # divide pos and neg
    x_pos = []
    x_neg = []
    y_pos = []
    y_neg = []
    for (x, y) in zip(feature, label):
        if y[0] == 0:
            x_neg.append(x[0, 1])
            y_neg.append(x[0, 2])
        else:
            x_pos.append(x[0, 1])
            y_pos.append(x[0, 2])
    plt.scatter(x_pos, y_pos, marker='+')
    plt.scatter(x_neg, y_neg, marker='o')
    # plot contour
    u = np.linspace(-1, 1.5, 200)
    v = np.linspace(-1, 1.5, 200)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[j, i] = map_feature(u[i], v[j]) * theta
    plt.contour(u, v, z, [0], colors='g')


if __name__ == '__main__':
    feature, label = load_data()
    # all monomials
    _feature = []
    for i in range(len(feature)):
        _feature.append(map_feature(feature[i, 1], feature[i, 2]))
    _feature = np.mat(_feature)

    plt.figure()
    # we calculate theta for each lambda
    lamb = [0, 1, 10]
    for i in range(len(lamb)):
        theta = newton(_feature, label, lamb[i])
        print(theta.T)
        ax = plt.subplot(1, len(lamb), i + 1)
        ax.set_title("%.3f" % lamb[i])
        plt_logistic(feature, label, theta)
    plt.show()
