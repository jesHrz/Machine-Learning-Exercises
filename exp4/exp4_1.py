import numpy as np


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
                p[i][j][k] /= y[k]
    for i in range(max_y + 1):
        y[i] /= n
    return p, y


def classify(feature, prob, y):
    ans = 0
    max_p = 0
    m = np.shape(feature)[1]
    for i in range(len(y)):
        p = y[i]
        for j in range(m):
            p *= prob[j][feature[0, j]][i]
        if p > max_p:
            max_p = p
            ans = i
    return ans


if __name__ == "__main__":
    feature, label = load_data("exp4/data/training_data.txt")
    feature_test, label_test = load_data("exp4/data/test_data.txt")
    p, y = fit(feature, label)
    correct = 0
    for (f_test, l_test) in zip(feature_test, label_test):
        if classify(f_test, p, y) == l_test:
            correct += 1
    print("accuracy=%.2f%%" % (correct / len(feature_test) * 100))
