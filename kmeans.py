import matplotlib.pyplot as plt
import numpy as np

# open plot in new window
plt.figure()
# enable interactive mode
plt.ion()

# array of 5 tuples with 2 random numbers between 0 and 200
means = np.random.randint(0, 400, (5, 2))
dev = np.random.randint(10, 20, 5)

# create 5 clusters of normal groups of points, 100 points each,
# with random means and standard deviations
points = np.concatenate((
    np.random.normal(means[0], dev[0], (100, 2)),
    np.random.normal(means[1], dev[1], (100, 2)),
    np.random.normal(means[2], dev[2], (100, 2)),
    np.random.normal(means[3], dev[3], (100, 2)),
    np.random.normal(means[4], dev[4], (100, 2))
))

# plot the points black
plt.scatter(points[:, 0], points[:, 1], c='k')

# show the plot
plt.show()

# choose 5 random points from the points as the initial centroids
centroids = points[np.random.choice(points.shape[0], 5, replace=False)]

# assign a color to each centroid
colors = ['r', 'g', 'b', 'y', 'c']

# plot the points
plt.scatter(points[:, 0], points[:, 1], c='k')
# plot the centroids with different colors, double the size
for i in range(5):
    plt.scatter(centroids[i, 0], centroids[i, 1], c=colors[i], s=150)

# show the plot
plt.show()

# create a copy of the centroids
centroids_old = np.copy(centroids)

# while the centroids have changed
while True:
    # create a list to store the points for each cluster
    clusters = [[] for _ in range(5)]
    # for each point
    for point in points:
        # calculate the distance between the point and each centroid
        distances = np.sqrt(np.sum((point - centroids) ** 2, axis=1))
        # get the index of the closest centroid
        cluster = np.argmin(distances)
        # add the point to the cluster
        clusters[cluster].append(point)

    # for each cluster
    for i in range(5):
        # get the points in the cluster
        cluster = np.array(clusters[i])
        # if there are points in the cluster
        if cluster.any():
            # calculate the new centroid
            centroids[i] = np.mean(cluster, axis=0)

    # if the centroids have not changed, stop the loop
    if np.array_equal(centroids, centroids_old):
        break
    # otherwise, set the old centroids to the new centroids
    centroids_old = np.copy(centroids)

    # plot the clusters with different colors, centroids black double the size
    for i in range(5):
        cluster = np.array(clusters[i])
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i])
        plt.scatter(centroids[i, 0], centroids[i, 1], c='k', s=150)

    # show the plot
    plt.show()
