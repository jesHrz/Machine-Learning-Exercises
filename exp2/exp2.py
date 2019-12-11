import numpy as np
from logistic_regression import Logistic_Regression


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
    return np.array(feature), np.array(label)


def load_result(filename):
    w = np.mat(np.zeros((3, 1)))
    cost = []
    with open(filename) as f:
        alpha, w[0, 0], w[1, 0], w[2, 0], _ = f.readline().split()
        for each_line in f.readlines():
            cost.append(each_line.strip())
    return alpha, w, np.mat(cost).T


# 梯度下降法 c++计算结果 py可视化
def GD(X, y):
    def plot_GD(X, y, filename):
        x_pos = []
        y_pos = []
        x_neg = []
        y_neg = []
        for (x, y) in zip(X, y):
            if y[0] == 1:
                x_pos.append(x[1])
                y_pos.append(x[2])
            else:
                x_neg.append(x[1])
                y_neg.append(x[2])
        alpha, weights, cost = load_result(filename)
        print(weights.T)
        print(1 - Logistic_Regression.sigmoid(np.mat([1, 20, 80]) * weights))  # probability of [20, 80] will not be admiited

        import matplotlib.pyplot as plt
        plt.figure(str(alpha))
        plt.subplot(1, 2, 1)
        plt.scatter(x_pos, y_pos, marker='+', label='Admitted')
        plt.scatter(x_neg, y_neg, marker='o', label='Not Admitted')
        x = np.arange(min(X[:, 1]), max(X[:, 1]), 0.1)
        y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]
        plt.plot(x, y, 'y-', label='Decision Boundary')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(0, len(cost), len(cost)), cost)
        plt.show()

    result_file = ["0.000900.txt", "0.001200.txt", "0.001500.txt", "0.001800.txt", "0.001900.txt", "0.002000.txt", "0.002500.txt"]
    for file in result_file:
        plot_GD(X, y, filename="exp2/data/"+file)


def Newton(X, y):
    def plot_Newton(X, y, lg):
        x_pos = []
        y_pos = []
        x_neg = []
        y_neg = []
        for (x, y) in zip(X, y):
            if y[0] == 1:
                x_pos.append(x[1])
                y_pos.append(x[2])
            else:
                x_neg.append(x[1])
                y_neg.append(x[2])
        print(lg.w_.T)
        print(1 - lg.predict(np.array([1, 20, 80])))
        import matplotlib.pyplot as plt
        plt.figure('Newton')
        plt.subplot(1, 2, 1)
        plt.scatter(x_pos, y_pos, marker='+', label='Admitted')
        plt.scatter(x_neg, y_neg, marker='o', label='Not Admitted')
        x = np.arange(min(X[:, 1]), max(X[:, 1]), 0.1)
        y = (-lg.w_[0] - lg.w_[1] * x) / lg.w_[2]
        plt.plot(x, y, 'y-', label='Decision Boundary')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(0, len(lg.cost_), len(lg.cost_)), lg.cost_)
        plt.show()

    lg = Logistic_Regression()
    lg.fit(X, y)
    plot_Newton(X, y, lg=lg)


if __name__ == '__main__':
    X, y = load_data()

    GD(X, y)
    Newton(X, y)
