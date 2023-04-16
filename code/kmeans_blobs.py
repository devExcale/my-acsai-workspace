import matplotlib.pyplot as plt
import numpy as np


def gen_blobs(n_blobs: int, blob_points: int, blob_size: int, blob_dev: int) -> np.ndarray:
    # array of n_blobs tuples with 2 random numbers between 0 and blob_size
    means = np.random.randint(0, blob_size, (n_blobs, 2))
    # create n_blobs clusters of normal groups of points, blob_points each,
    # with random means and standard deviations
    points = np.concatenate([
        np.random.normal(means[i], blob_dev, (blob_points, 2))
        for i in range(n_blobs)
    ])
    return points


def plot_points(points: np.ndarray, color: str = 'k'):
    plt.scatter(points[:, 0], points[:, 1], c=color)


def init_centroids(points: np.ndarray, k: int) -> np.ndarray:
    # get first centroid randomly
    print("Choosing centroid 0")
    centroids = [points[np.random.choice(points.shape[0], 1)]]

    # get other centroids using kmeans++
    for i in range(1, k):
        print("Choosing centroid " + str(i))

        # create a list to store the distances
        distances = []

        # for each point
        for point in points:
            # compute the distance from the closest centroid
            d = np.min(np.sqrt(np.sum((point - centroids) ** 2, axis=1)))
            distances.append(d)

        weights = distances / np.sum(distances)
        # square and normalize the weights
        weights = weights ** 2
        weights = weights / np.sum(weights)

        # choose the next centroid using the distances
        centroid = points[np.random.choice(points.shape[0], 1, p=weights)]
        centroids.append(centroid)

    # convert the list of centroids to a numpy array
    return np.array(centroids).squeeze()


def kmeanspp(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    k = centroids.shape[0]

    # create a copy of the centroids
    centroids_old = np.copy(centroids)
    # while the centroids have changed
    while True:
        # create a list to store the points for each cluster
        clusters = [[] for _ in range(k)]
        # for each point
        for point in points:
            # calculate the distance between the point and each centroid
            distances = np.sqrt(np.sum((point - centroids) ** 2, axis=1))
            # get the index of the closest centroid
            cluster = np.argmin(distances, axis=0)
            # add the point to the cluster
            clusters[cluster].append(point)
        # for each cluster
        for i in range(k):
            # calculate the mean of the cluster
            centroids[i] = np.mean(clusters[i], axis=0)
        # if the centroids have not changed
        if np.array_equal(centroids, centroids_old):
            # return the centroids
            return centroids
        # update the old centroids
        centroids_old = np.copy(centroids)


def get_class(point: np.ndarray, centroids: np.ndarray) -> int:
    # calculate the distance between the point and each centroid
    distances = np.sqrt(np.sum((point - centroids) ** 2, axis=1))
    # get the index of the closest centroid
    cluster = np.argmin(distances)
    return cluster


def get_clusters(points: np.ndarray, centroids: np.ndarray) -> list:
    # create a list to store the points for each cluster
    clusters = [[] for _ in range(centroids.shape[0])]
    # for each point
    for point in points:
        # get the index of the closest centroid
        cluster = get_class(point, centroids)
        # add the point to the cluster
        clusters[cluster].append(point)
    # convert the clusters to numpy arrays
    clusters = [np.array(cluster) for cluster in clusters]
    # return the clusters
    return clusters


def plot_clusters(clusters: list, colors: list):
    for cluster, color in zip(clusters, colors):
        plot_points(cluster, color)


def main():
    dataset = gen_blobs(5, 100, 400, 20)
    centroids = init_centroids(dataset, 5)

    plot_points(dataset)
    plot_points(centroids, 'y')
    plt.show()

    kmeanspp(dataset, centroids)
    clusters = get_clusters(dataset, centroids)
    plot_clusters(clusters, ['r', 'g', 'b', 'c', 'm'])
    plot_points(centroids, 'y')
    plt.show()


if __name__ == '__main__':
    main()
