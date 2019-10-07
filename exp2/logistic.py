import numpy as np


def sig(x): return 1 / (1 + np.exp(-x))


def J(feature, label, weights):
    ret = 0
    m = np.shape(feature)[0]
    for i in range(m):
        h = sig(feature[i] * weights)
        if h <= 0:
            h = 0.0000001
        elif h >= 1:
            h = 0.9999999
        ret += -label[i, 0] * np.log(h) - (1 - label[i, 0]) * np.log(1 - h)
    return ret / m


def load_data():
    feature = []
    label = []
    with open('exp2/data/ex2x.dat') as f:
        for each_line in f.readlines():
            feature_temp = []
            feature_temp.append(1)
            for data in each_line.strip().split():
                feature_temp.append(float(data))
            feature.append(feature_temp)
    with open('exp2/data/ex2y.dat') as f:
        for each_line in f.readlines():
            label_temp = []
            for data in each_line.strip().split():
                label_temp.append(float(data))
            label.append(label_temp)
    return np.mat(feature), np.mat(label)


def load_result(filename):
    w = np.mat(np.zeros((3, 1)))
    cost = []
    with open(filename) as f:
        alpha, w[0, 0], w[1, 0], w[2, 0], _ = f.readline().split()
        for each_line in f.readlines():
            cost.append(each_line.strip())
    return alpha, w, np.mat(cost).T


# 梯度下降 太慢
def fit(feature, label, alpha, epsilon=1e-7):
    m, n = np.shape(feature)
    # w = np.mat([-4.96253179, 0.07103084, 0.03521583]).T
    w = np.mat(np.zeros((n, 1)))
    val = 0
    iteration = 0
    cost = []
    while True:
        iteration += 1
        h = sig(feature * w)
        err = label - h
        w = w + alpha * feature.T * err / m
        last = val
        val = J(feature, label, w)
        cost.append(val)
        if abs(val - last) < epsilon:
            break
    return w, np.mat(cost).T, iteration


def plot_fit(feature, label, **kwrags):
    x_pos = []
    y_pos = []
    x_neg = []
    y_neg = []
    for (x, y) in zip(feature, label):
        if y[0] == 1:
            x_pos.append(x[0, 1])
            y_pos.append(x[0, 2])
        else:
            x_neg.append(x[0, 1])
            y_neg.append(x[0, 2])
    file = kwrags.get('filename', None)
    weights = kwrags.get('weights', [])
    cost = kwrags.get('cost', [])
    alpha = kwrags.get('alpha', -1)
    if file:
        alpha, weights, cost = load_result(file)
    print(weights.T)
    print(1 - predict(np.mat([1, 20, 80]), weights))  # probability of [20, 80] will not be admiited
    import matplotlib.pyplot as plt
    plt.figure(str(alpha) if alpha != -1 else "Newton")
    plt.subplot(1, 2, 1)
    plt.scatter(x_pos, y_pos, marker='+', label='Admitted')
    plt.scatter(x_neg, y_neg, marker='o', label='Not Admitted')
    x = np.arange(min(feature[:, 1]), max(feature[:, 1]), 0.1)
    y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]
    plt.plot(x, y, 'y-', label='Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, len(cost), len(cost)), cost)
    plt.show()


def predict(data, w):
    h = sig(data * w)
    # m = np.shape(h)[0]
    # for i in range(m):
    # h[i, 0] = 0.0 if h[i, 0] < 0.5 else 1.0
    return h


def newton(feature, label, epsilon=1e-6):
    m, n = np.shape(feature)
    w = np.mat(np.zeros((n, 1)))
    val = 0
    iteration = 0
    cost = []
    while True:
        iteration += 1
        H = np.mat(np.zeros((n, n)))
        for i in range(m):
            h = sig(feature[i] * w)
            H += (h * (1 - h))[0, 0] * feature[i].T * feature[i]
        err = label - sig(feature * w)
        w = w + H.I * feature.T * err / m
        last = val
        val = J(feature, label, w)
        cost.append(val[0, 0])
        if abs(val - last) < epsilon:
            break
    return w, cost, iteration


if __name__ == '__main__':
    feature, label = load_data()
    '''
    exe1 梯度下降法 C++计算结果，Python可视化
    '''
    # # w, theta, iteration = fit(feature, label, 0.0012)
    # result_file = ["0.000900.txt", "0.001200.txt", "0.001500.txt", "0.001800.txt", "0.001900.txt", "0.002000.txt", "0.002500.txt"]
    # for file in result_file:
    #     plot_fit(feature, label, filename="exp2/data/"+file)

    '''
    exe2 牛顿迭代
    '''
    w, cost, iteration = newton(feature, label)
    plot_fit(feature, label, weights=w, cost=cost)
