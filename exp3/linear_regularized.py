import numpy as np

def load_data():
    feature = []
    with open("exp3/data/ex3Linx.dat") as f:
        for each_line in f.readlines():
            feature_tmp = []
            for data in each_line.strip().split():
                for i in range(6):
                    feature_tmp.append(float(data) ** i)
            feature.append(feature_tmp)
    label = []
    with open("exp3/data/ex3Liny.dat") as f:
        for each_line in f.readlines():
            label_tmp = []
            for data in each_line.strip().split():
                label_tmp.append(float(data))
            label.append(label_tmp)
    return np.mat(feature), np.mat(label)

def plt_linear(feature, label, theta):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(feature[:, 1].tolist(), label.tolist(), marker='o')
    x = np.arange(min(feature[:, 1]), max(feature[:, 1]), 0.01)
    y = []
    for i in x:
        now = 0
        for j in range(6):
           now += theta[j, 0] * (i ** j)
        y.append(now)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    feature, label = load_data()
    m, n = np.shape(feature)
    lamd = [0, 1, 10]
    for lam in lamd:
        E = np.eye(n)
        E[0, 0] = 0
        theta = (feature.T * feature + lam * E).I * feature.T * label
        print("lambda=", lam)
        print(theta.T)
        plt_linear(feature, label, theta)