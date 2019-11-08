import numpy as np
import matplotlib.pyplot as plt
laplace = [3, 5, 4, 4, 3, 2, 3, 3]


def load_data(file):
    feature = []
    label = []
    with open(file) as f:
        for each_line in f.readlines():
            data = list(map(int, each_line.strip().split()))
            feature.append(data[0:-1])
            label.append([data[-1]])
    return np.mat(feature), np.mat(label)


def fit(feature, label):
    n, m = np.shape(feature)
    max_x = np.max(feature)
    max_y = np.max(label)
    p = [[[0 for i in range(max_y + 1)] for i in range(max_x + 1)] for i in range(m)]
    y = [0 for i in range(max_y + 1)]
    for j in range(n):
        y[label[j, 0]] += 1
        for i in range(m):
            p[i][feature[j, i]][label[j, 0]] += 1
    for i in range(m):
        for j in range(max_x + 1):
            for k in range(max_y + 1):
                p[i][j][k] /= (y[k] + laplace[i])
    for i in range(max_y + 1):
        y[i] /= n
    return p, y


def classify(feature, prob, y):
    label = 0
    max_p = 0
    m = np.shape(feature)[1]
    for i in range(5):
        p = y[i]
        for j in range(m):
            p *= prob[j][feature[0, j]][i]
        if p > max_p:
            max_p = p
            label = i
    return label


def random_test(feature, label, num):
    import random
    vis = dict()
    n = np.shape(feature)[0]
    feature_select = []
    label_select = []
    for _ in range(num):
        pos = random.randint(0, n - 1)
        while pos in vis:
            pos = random.randint(0, n - 1)
        vis[pos] = 1
        feature_select.append(feature[pos].tolist()[0])
        label_select.append(label[pos].tolist()[0])
    return np.mat(feature_select), np.mat(label_select)


if __name__ == "__main__":
    feature, label = load_data("exp4/data/training_data.txt")
    feature_test, label_test = load_data("exp4/data/test_data.txt")

    tot = [i * 100 for i in range(1, 101)]
    rate = []
    for cnt in tot:
        f, l = random_test(feature, label, cnt)
        p, y = fit(f, l)
        correct = 0
        for (f_test, l_test) in zip(feature_test, label_test):
            if classify(f_test, p, y) == l_test:
                correct += 1
        rate.append(correct / len(feature_test) * 100)
    plt.plot(tot, rate)
    plt.show()
