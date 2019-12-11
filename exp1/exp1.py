import numpy as np
from linear_regression import Linear_Regression


def load_data(file1, file2):
    X = []
    y = []
    with open(file1) as f:
        for each_line in f.readlines():
            X.append(list(map(float, each_line.strip().split())))
    with open(file2) as f:
        for each_line in f.readlines():
            y.append(list(map(float, each_line.strip().split())))
    return np.array(X), np.array(y)


def plot_regression(lr):
    import matplotlib.pyplot as plt
    plt.figure("Linear Regression")
    x0 = np.min(lr.X_)
    x1 = np.max(lr.X_)
    y0, y1 = lr.predict([[x0], [x1]])
    plt.plot(lr.X_[:, 1], lr.y_, '.', label="origin")
    plt.plot([x0, x1], [y0, y1], label='regression')
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    plt.legend()
    plt.show()


def plot_cost(X, y):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    figure = plt.figure("cost value")
    w_0 = np.linspace(-3, 3, 100)
    w_1 = np.linspace(-1, 1, 100)
    cost_value = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            cost_value[i][j] = Linear_Regression.cost_value(X, y, np.array([w_0[i], w_1[j]]).reshape(-1, 1))
    w_0, w_1 = np.meshgrid(w_0, w_1)
    # 三维图
    Axes3D(figure).plot_surface(w_0, w_1, cost_value)
    # 等高线
    # plt.contourf(w_0, w_1, cost_value, 40, alpha=0.8)
    # cb = plt.colorbar()
    # cb.set_label('cost')
    plt.xlabel('w_0')
    plt.ylabel('w_1')
    plt.show()


def task5():
    import matplotlib.pyplot as plt

    # 标准化dt
    def scale(dt):
        mu = np.mean(dt, axis=0)
        sigma = np.std(dt, axis=0)
        return (dt - mu) / sigma
    X, y = load_data("exp1/data/ex1_2x.dat", "exp1/data/ex1_2y.dat")
    X = scale(np.vstack((X, [[1650, 3]])))

    max_iteration = 50
    learning_rate = [0.05, 0.15, 0.45, 1.2, 0.005]
    lr = [Linear_Regression() for i in range(len(learning_rate))]
    for i in range(len(learning_rate)):
        lr[i].fit(X[:-1], y, learning_rate=learning_rate[i], max_iteration=max_iteration)
    print("w=", lr[1].w_.T)
    print("prediction[1650, 3]=", lr[1].predict([X[-1]]))

    # 每次迭代损失函数的值
    plt.figure('cost')
    for i in range(len(learning_rate)):
        plt.plot(np.linspace(0, max_iteration, max_iteration), lr[i].cost_, label='%f' % learning_rate[i])
    plt.xlabel('number of iterations')
    plt.ylabel('cost value')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X, y=load_data("exp1/data/ex1_1x.dat", "exp1/data/ex1_1y.dat")
    lr=Linear_Regression()
    lr.fit(X, y)
    y_predict=lr.predict([[3.5], [7]])
    print("w=", lr.w_.T)
    print("predction[3.5]=", y_predict[0])
    print("predction[7.0]=", y_predict[1])

    # plot_regression(lr)
    # plot_cost(lr.X_, lr.y_)

    task5()
