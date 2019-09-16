import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 标准化dt
def scale(dt):
    mu = np.mean(dt, axis=0)
    sigma = np.std(dt, axis=0)
    return (dt - mu) / sigma


# 计算损失函数
def j_val(X, Y, theta):
    hx = np.dot(X, theta)
    return np.dot((hx - Y).transpose(), hx - Y) / (2 * X.shape[0])


# 梯度下降
def gradient_descent(X, Y, theta, learning_rate=0.07, max_iteration=1500):
    # g = ((XO - Y)' * (X))'
    cost = np.zeros((max_iteration, 1))
    for i in range(max_iteration):
        cost[i] = j_val(X, Y, theta)
        hx = np.dot(X, theta)
        theta = theta - (learning_rate / X.shape[0]) * (np.dot((hx - Y).transpose(), X)).transpose()
    return theta, cost


def task5():
    x = np.loadtxt('exp1/data/ex1_2x.dat')
    y = np.loadtxt('exp1/data/ex1_2y.dat').reshape(-1, 1)
    m = x.shape[0]
    # 将x标准化
    x = scale(x)
    x = np.hstack((np.ones((m, 1)), x))
    m, n = x.shape
    Theta, cost = gradient_descent(x, y, np.zeros((n, 1)), learning_rate=0.15, max_iteration=50)
    print(Theta, cost)
    # plot3
    plt.figure('Iteration')
    plt.plot(np.linspace(0, 50, 50), cost)
    plt.xlabel('number of iterations')
    plt.ylabel('Cost J')
    plt.show()


if __name__ == "__main__":
    x = np.loadtxt('exp1/data/ex1_1x.dat').reshape(-1, 1)
    y = np.loadtxt('exp1/data/ex1_1y.dat').reshape(-1, 1)
    m = x.shape[0]
    x = np.hstack((np.ones((m, 1)), x))
    m, n = x.shape
    Theta, cost = gradient_descent(x, y, np.zeros((n, 1)))
    y1 = np.dot(np.array([1, 3.5]), Theta)
    y2 = np.dot(np.array([1, 7]), Theta)
    print("X=3.5\tY=%f" % y1)
    print("X=7.0\tY=%f" % y2)
    learn_y = np.dot(x, Theta)
    # plot1
    plt.figure('Linear Regression')
    plt.plot(x[:, 1], y, '.')
    plt.plot(x[:, 1], learn_y)
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.show()

    # plot2
    fig2 = plt.figure('J value')
    theta_0 = np.linspace(-3, 3, 100)
    theta_1 = np.linspace(-1, 1, 100)
    j_vals = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            j_vals[i][j] = j_val(x, y, np.array([theta_0[i], theta_1[j]]).reshape(-1, 1))
    theta_0, theta_1 = np.meshgrid(theta_0, theta_1)
    # surf
    # ax = Axes3D(fig2)
    # ax.plot_surface(theta_0, theta_1, j_vals)
    plt.contour(theta_0, theta_1, j_vals)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.show()

    task5()

