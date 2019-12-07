import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
# from sklearn.cluster import KMeans


class KMeans:
    def __init__(self, n_clusters, max_iter=200):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        m, n = np.shape(X)
        assert self.n_clusters <= m
        self.__random_cluster_centers_(X)
        self.labels_ = [0 for i in range(m)]
        for iter in range(self.max_iter):
            for i in range(m):
                k = 0
                for j in range(1, self.n_clusters):
                    if np.dot(X[i] - self.cluster_centers_[j], X[i] - self.cluster_centers_[j]) < np.dot(X[i] - self.cluster_centers_[k], X[i] - self.cluster_centers_[k]):
                        k = j
                self.labels_[i] = k
            for j in range(self.n_clusters):
                sum_x = np.zeros(n)
                sum_cnt = 0
                for i in range(m):
                    if self.labels_[i] == j:
                        sum_x += X[i]
                        sum_cnt += 1
                if sum_cnt > 0:
                    self.cluster_centers_[j] = sum_x / sum_cnt

    def __random_cluster_centers_(self, X):
        import random
        ind = random.sample([i for i in range(len(X))], self.n_clusters)
        self.cluster_centers_ = []
        for i in ind:
            self.cluster_centers_.append(X[i])
        self.cluster_centers_ = np.array(self.cluster_centers_)


if __name__ == "__main__":
    def image_reconstruction(cluster_centers_, labels_, w, h):
        channel = np.shape(cluster_centers_)[1]
        image = np.zeros((w, h, channel))
        ind = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = cluster_centers_[labels_[ind]]
                ind += 1
        return image

    bird = matplotlib.image.imread("exp6/data/bird_small.tiff")
    bird = np.array(bird, dtype=np.uint)
    m, n, channel = np.shape(bird)

    kmeans = KMeans(n_clusters=16, max_iter=50)
    kmeans.fit(np.reshape(bird, (m * n, channel)))

    bird_compressed = image_reconstruction(kmeans.cluster_centers_, kmeans.labels_, m, n)
    matplotlib.image.imsave("exp6/data/bird_compressed.tiff", bird_compressed)

    plt.figure(1)
    plt.title("Origin image")
    plt.axis("off")
    plt.imshow(bird)
    plt.figure(2)
    plt.title("Compressed image (16 colors)")
    plt.axis("off")
    plt.imshow(bird_compressed)
    plt.show()