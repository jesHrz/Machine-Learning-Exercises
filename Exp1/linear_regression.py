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
    _theta = np.zeros((max_iteration, theta.shape[0]))
    for i in range(max_iteration):
        cost[i] = j_val(X, Y, theta)
        _theta[i] = theta.transpose()
        hx = np.dot(X, theta)
        theta = theta - (learning_rate / X.shape[0]) * (np.dot((hx - Y).transpose(), X)).transpose()
    return theta, cost, _theta


def task5():
    x = np.loadtxt('exp1/data/ex1_2x.dat')
    y = np.loadtxt('exp1/data/ex1_2y.dat').reshape(-1, 1)

    # 将x标准化
    x = np.vstack((x, [[1650, 3]]))
    x = scale(x)
    x = np.hstack((np.ones((x.shape[0], 1)), x))

    n = x.shape[1]
    max_iteration = 50
    Theta, cost, _theta = gradient_descent(x[0: x.shape[0] - 1], y, np.zeros((n, 1)),
                                           learning_rate=0.15, max_iteration=max_iteration)
    print("Theta =", Theta.transpose())

    print("prediction[1650, 3] =", np.dot(Theta.transpose(), x[x.shape[0] - 1].transpose()))

    # 每次迭代损失函数的值
    plt.figure('cost')
    plt.plot(np.linspace(0, max_iteration, max_iteration), cost)
    plt.xlabel('number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # 每次迭代的theta
    plt.figure('theta')
    plt.plot(np.linspace(0, max_iteration, max_iteration), _theta[:, 0])    # theta_0
    plt.plot(np.linspace(0, max_iteration, max_iteration), _theta[:, 1])    # theta_1
    plt.plot(np.linspace(0, max_iteration, max_iteration), _theta[:, 2])    # theta_2
    plt.show()


if __name__ == "__main__":
    x = np.loadtxt('exp1/data/ex1_1x.dat').reshape(-1, 1)
    y = np.loadtxt('exp1/data/ex1_1y.dat').reshape(-1, 1)

    x = np.hstack((np.ones((x.shape[0], 1)), x))

    m, n = x.shape
    Theta, cost, _theta = gradient_descent(x, y, np.zeros((n, 1)))
    print("Theta =", Theta.transpose())

    # 对x=3.5和x=7做预测
    y1 = np.dot([1, 3.5], Theta)
    y2 = np.dot([1, 7], Theta)
    print("prediction[3.5] =", y1)
    print("prediction[7.0] =", y2)

    # 散点图与拟合直线
    plt.figure('Linear Regression')
    plt.plot(x[:, 1], y, '.')   # 散点
    plt.plot(x[:, 1], np.dot(x, Theta))  # 直线
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.show()

    # 损失函数可视化
    fig2 = plt.figure('J value')
    theta_0 = np.linspace(-3, 3, 100)
    theta_1 = np.linspace(-1, 1, 100)
    j_vals = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            j_vals[i][j] = j_val(x, y, np.array([theta_0[i], theta_1[j]]).reshape(-1, 1))
    theta_0, theta_1 = np.meshgrid(theta_0, theta_1)

    # 三维图
    Axes3D(fig2).plot_surface(theta_0, theta_1, j_vals)
    # 等高线
    # plt.contour(theta_0, theta_1, j_vals)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.show()

    task5()
