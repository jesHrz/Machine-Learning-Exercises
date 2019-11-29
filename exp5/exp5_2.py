import numpy as np
import random
from SVM import SVM


def load_data(file):
    X = []
    y = []
    with open(file) as f:
        for each_line in f.readlines():
            data = each_line.strip().split()
            y.append(float(data[0]))
            x = [0 for i in range(784)]
            for s in data[1:-1]:
                ind, color = map(int, s.split(":"))
                x[ind - 1] = color * 100 / 255
            X.append(x)
    return np.array(X), np.array(y)


def random_sample(X, y, k):
    ind = random.sample([i for i in range(len(X))], k)
    X_random = []
    y_random = []
    for i in ind:
        X_random.append(X[i])
        y_random.append(y[i])
    return np.array(X_random), np.array(y_random), ind


if __name__ == "__main__":
    X_train, y_train = load_data("exp5/data/train-01-images.svm")
    X_test, y_test = load_data("exp5/data/test-01-images.svm")
    X_train, y_train, ind = random_sample(X_train, y_train, 3000)

    clf = SVM(C=1e-10)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_train)
    correct = np.sum(y_predict == y_train)
    print("%d out of %d training examples correct(%.2f%%)." % (correct, len(y_train), correct/len(y_train)*100))

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct(%.2f%%)." % (correct, len(y_test), correct/len(y_test)*100))
