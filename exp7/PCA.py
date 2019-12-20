import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components_ = n_components
        self.U_ = None

    def fit_save(self, X, eig_val_path, eig_vec_path):
        m, n = np.shape(X)
        X = X - np.mean(X, axis=0)
        S = np.dot(X.T, X) / m  # (n, n)
        eig_val, eig_vec = np.linalg.eig(S)  # vec (n, 1)
        eig_val = eig_val.real
        eig_vec = eig_vec.real
        np.savetxt(eig_val_path, eig_val)
        np.savetxt(eig_vec_path, eig_vec)
        eig_pairs = [(eig_val[i], eig_vec[:, i]) for i in range(n)]
        eig_pairs.sort(key=lambda pair: -pair[0])
        self.U_ = np.array([pair[1] for pair in eig_pairs[:self.n_components_]])

    def fit_load(self, eig_val_path, eig_vec_path):
        eig_val = np.loadtxt(eig_val_path)
        eig_vec = np.loadtxt(eig_vec_path)
        eig_pairs = [(eig_val[i], eig_vec[:, i]) for i in range(len(eig_val))]
        eig_pairs.sort(key=lambda pair: -pair[0])
        self.U_ = np.array([pair[1] for pair in eig_pairs[:self.n_components_]])

    def fit_transform_save(self, X, eig_val_path, eig_vec_path):
        self.fit_save(X, eig_val_path, eig_vec_path)
        return self.transform(X)

    def fit_transform_load(self, X, eig_val_path, eig_vec_path):
        self.fit_load(eig_val_path, eig_vec_path)
        return self.transform(X)

    def transform(self, X):
        assert self.U_ is not None
        return np.dot(X - np.mean(X, axis=0), self.U_.T)
