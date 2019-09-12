import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(x, y, theta, learning_rate=0.07, max_iteration=1500):
    # g = ((XO - Y)' * (X))'
    for i in range(max_iteration):
        hx = np.dot(x, theta)
        theta = theta - (learning_rate / x.shape[0]) * (np.dot((hx - y).transpose(), x)).transpose()
    return theta


if __name__ == "__main__":
    x = np.loadtxt('data/ex1_1x.dat').reshape(-1, 1)
    y = np.loadtxt('data/ex1_1y.dat').reshape(-1, 1)
    m = x.shape[0]
    x = np.hstack((np.ones((m, 1)), x))
    m, n = x.shape
    Theta = gradient_descent(x, y, np.zeros((n, 1)))
    y1 = np.dot(np.array([1, 3.5]), Theta)
    y2 = np.dot(np.array([1, 7]), Theta)
    print("X=3.5\tY=%f" % y1)
    print("X=7.0\tY=%f" % y2)
    Y = np.dot(x, Theta)
    # plot
    plt.figure('Linear Regression')
    plt.plot(x[:, 1], y, '.')
    plt.plot(x[:, 1], Y)
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.show()