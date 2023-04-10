import sklearn.cluster as skc

from my_libs.images import *

dims = (28, 28, 1)
n_clusters = 3


def init_centroids(dataset: np.ndarray, k: int) -> np.ndarray:
	# get first centroid randomly
	print("Choosing centroid 0")
	# ndarray(1, D)
	centroids = dataset[np.random.choice(dataset.shape[0], 1)]

	# get other centroids using kmeans++
	for i in range(1, k):
		print("Choosing centroid " + str(i))

		# create a list to store the distances
		distances = []

		# for each point
		for point in dataset:
			# compute the distance from the closest centroid
			d_vec = point - centroids
			d = np.min(np.sqrt(np.sum(d_vec ** 2, axis=1)))
			distances.append(d)

		weights = distances / np.sum(distances)
		# square and normalize the weights
		weights = weights ** 2
		weights = weights / np.sum(weights)

		# choose the next centroid using the distances
		new_centroid = dataset[np.random.choice(dataset.shape[0], 1, p=weights)]
		centroids = np.concatenate((centroids, new_centroid), axis=0)

	# convert the list of centroids to a numpy array
	return np.asarray(centroids)


def kmeanspp(dataset: np.ndarray, centroids: np.ndarray) -> None:
	# Number of centroids
	k = centroids.shape[0]
	# create a copy of the centroids
	centroids_old = np.copy(centroids)

	n_iter = 1
	# while the centroids have changed
	while True:
		print("Iteration " + str(n_iter))
		n_iter += 1

		# create a list to store the points for each cluster
		clusters = [[] for _ in range(k)]
		labels = [0 for _ in range(dataset.shape[0])]

		# assignment step
		for i, point in enumerate(dataset):
			# calculate the distance between the point and each centroid
			distances = point - centroids
			distances = np.sqrt(np.sum(distances ** 2, axis=1))
			# get the index of the closest centroid
			label = np.argmin(distances, axis=0)
			labels[i] = int(label)
			# add the point to the cluster
			clusters[label].append(point)

		# check for empty clusters
		for i, empty_cluster in enumerate(clusters):
			if len(empty_cluster) != 0:
				continue

			# compute distances from the centroids
			distances = []
			for j, point in enumerate(dataset):
				# distance between the point and the assigned centroid
				d = np.sqrt(np.sum((point - centroids[labels[j]]) ** 2))
				distances.append(d)

			# choose the point with the largest distance
			far_point = np.argmax(distances)
			i_prev_cluster = labels[far_point]
			# assign the point to the empty cluster
			clusters[i].append(dataset[far_point])
			labels[far_point] = i
			# remove the point from the cluster it was previously assigned to
			clusters[i_prev_cluster] = [p for p in clusters[i_prev_cluster] if list_equiv(p, dataset[far_point])]

		# update step
		for i in range(k):
			# calculate the mean of the cluster
			cluster = clusters[i]
			mu = np.mean(cluster, axis=0)
			centroids[i] = mu

		# if the centroids have not changed
		if np.array_equal(centroids, centroids_old):
			# stop the loop
			break
		# otherwise
		else:
			# update the old centroids
			centroids_old = np.copy(centroids)

	return


def main():
	# read images
	circles = read_images("assets/shapes/circles")
	squares = read_images("assets/shapes/squares")
	triangles = read_images("assets/shapes/triangles")
	images = np.concatenate((circles, squares, triangles))

	print(f"Dataset size: {images.shape}")

	# initialize centroids
	centroids = init_centroids(images, n_clusters)

	# run kmeans++
	kmeanspp(images, centroids)

	print(f"Final centroids: {centroids.shape}")

	# show centroids
	for i, img in enumerate(centroids):
		cv.imshow("Centroid " + str(i), img.reshape(dims))

	cv.waitKey(0)
	cv.destroyAllWindows()


def main_scikit():
	# read images
	circles = read_images("assets/shapes/circles")
	squares = read_images("assets/shapes/squares")
	triangles = read_images("assets/shapes/triangles")
	images = np.concatenate((circles, squares, triangles))

	images = np.asarray([img_invert(img) for img in images])

	print(f"Dataset size: {images.shape}")

	# run kmeans using scikit-learn
	kmeans = skc.KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, max_iter=1000).fit(images)
	print(f"Iterations: {kmeans.n_iter_}")

	clusters = [[] for _ in range(n_clusters)]
	for i, label in enumerate(kmeans.labels_):
		clusters[label].append(images[i])

	# show centroids
	for i, img in enumerate(kmeans.cluster_centers_):
		# turn vector back into an image
		img = img.reshape(dims)
		# rescale 10x the image, no interpolation
		img = cv.resize(img, (0, 0), fx=10, fy=10)
		# show the image
		cv.imshow("Centroid " + str(i), img)
	cv.waitKey(0)
	cv.destroyAllWindows()

	# show first image of each cluster
	for i, cluster in enumerate(clusters):
		# turn vector back into an image
		img = cluster[0].reshape(dims)
		# rescale 10x the image
		img = cv.resize(img, (0, 0), fx=10, fy=10)
		# show the image
		cv.imshow("Cluster " + str(i), img)
	cv.waitKey(0)
	cv.destroyAllWindows()


def main_center():

	# read images
	path_shapes = 'assets/shapes/'
	circles = read_images(path_shapes + 'circles')
	squares = read_images(path_shapes + 'squares')
	triangles = read_images(path_shapes + 'triangles')

	# get two random images from each class
	circle = circles[np.random.randint(0, circles.shape[0], 2)]
	square = squares[np.random.randint(0, squares.shape[0], 2)]
	triangle = triangles[np.random.randint(0, triangles.shape[0], 2)]
	imgs = np.concatenate((circle, square, triangle))

	# process images
	imgs = np.asarray([img_invert(img) for img in imgs])
	imgs = np.asarray([img_threshold(img, 100, binary=False) for img in imgs])

	# for each image
	# compute the center of mass
	# and display the image with the center of mass
	for i, img in enumerate(imgs):
		# turn vector back into an image
		img = img.reshape(28, 28)
		# paint a pixel at the center of mass
		center = image_center(img)
		img[center] = abs(img[center] - 255)
		# show the image
		cv.imshow("Image " + str(i), img_scale(img, 10))

	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == "__main__":

	options = {
		0: main,
		1: main_scikit,
		2: main_center
	}

	options[2]()
