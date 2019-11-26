import numpy as np
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
                x[ind - 1] = float(color)
            X.append(x)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X_train, y_train = load_data("exp5/data/train-01-images.svm")
    X_test, y_test = load_data("exp5/data/test-01-images.svm")

    clf = SVM()
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct(%.2f%%)." % (correct, len(y_test), correct/len(y_test) * 100))
