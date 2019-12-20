#  F. Samaria and A. Harter
#   "Parameterisation of a stochastic model for human face identification"
#   2nd IEEE Workshop on Applications of Computer Vision
#   December 1994, Sarasota (Florida).

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from PCA import PCA
from SVM import SVM
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC


def load_data(file):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(1, 41):
        for j in range(1, 11):
            face = matplotlib.image.imread(file + "/s%d/%d.pgm" % (i, j))
            m, n = np.shape(face)
            face = np.array(face, dtype=np.float64) / 255
            face = np.reshape(face, m * n)
            if j <= 5:
                X_train.append(face)
                y_train.append(i)
            else:
                X_test.append(face)
                y_test.append(i)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data("exp7/orl_faces")

    pca = PCA(n_components=5)
    # pca.fit_save(np.vstack((X_train, X_test)), "exp7/eig_val.txt", "exp7/eig_vec.txt")
    pca.fit_load("exp7/eig_val.txt", "exp7/eig_vec.txt")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # pca.fit(np.vstack((X_train, X_test)))
    # X_train_pca = pca.transform(X_train)
    # X_test_pca = pca.transform(X_test)

    svm_all = []
    for i in range(1, 41):
        for j in range(i + 1, 41):
            X1 = X_train_pca[y_train == i]
            X2 = X_train_pca[y_train == j]
            y1 = np.array([1 for k in range(len(X1))])
            y2 = np.array([-1 for k in range(len(X2))])
            svm = SVM()
            svm.fit(np.vstack((X1, X2)), np.hstack((y1, y2)))
            svm_all.append(svm)

    correct = 0
    for k in range(len(X_test_pca)):
        score = [0 for i in range(41)]
        cnt = 0
        for i in range(1, 41):
            for j in range(i + 1, 41):
                if svm_all[cnt].predict(X_test_pca[k]) == 1:
                    score[i] += 1
                else:
                    score[j] += 1
                cnt += 1
        max_score = max(score)
        each = []
        for i in range(1, 41):
            if score[i] == max_score:
                each.append(i)
                if i == y_test[k]:
                    correct += 1
        print(each)
    print("%d out of %d predictions correct(%.2f%%)" % (correct, len(X_test_pca), correct / len(X_test_pca) * 100))