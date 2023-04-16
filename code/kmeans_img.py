import cv2 as cv
import sklearn.cluster as skc

from my_libs.images import *

dims = (28, 28, 1)
glob_n_clusters = 3


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


def main_first():
	# read images
	circles = read_images("../assets/shapes/circles")
	squares = read_images("../assets/shapes/squares")
	triangles = read_images("../assets/shapes/triangles")
	images = np.concatenate((circles, squares, triangles))

	print(f"Dataset size: {images.shape}")

	# initialize centroids
	centroids = init_centroids(images, 3)

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
	circles = read_images("../assets/shapes/circles")
	squares = read_images("../assets/shapes/squares")
	triangles = read_images("../assets/shapes/triangles")

	images = np.asarray([img_invert(img) for img in images])

	print(f"Dataset size: {images.shape}")

	# run kmeans using scikit-learn
	kmeans = skc.KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=1000).fit(images)
	print(f"Iterations: {kmeans.n_iter_}")

	clusters = [[] for _ in range(3)]
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


def main_cv(images: np.ndarray, n_clusters: int = 3) -> None:
	# convert to np.float32
	images_float = np.float32(images)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
	ret, label, center = cv.kmeans(images_float, n_clusters, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)

	# convert back to np.uint8
	center = np.uint8(center)

	# show centroids
	for i, img in enumerate(center):
		# turn vector back into an image
		img = img.reshape((28, 28))
		# show the image
		cv.imshow(f'Centroid {i}', img_scale(img, 10))

	# Group images by cluster
	clusters = [[] for _ in range(n_clusters)]
	for i, label in enumerate(label):
		clusters[label[0]].append(images[i])

	# get max cluster size
	max_cluster_size = max([len(cluster) for cluster in clusters])
	for i in range(max_cluster_size):
		for j, cluster in enumerate(clusters):
			if i < len(cluster):
				cv.imshow(f'Cluster {j}', img_scale(cluster[i].reshape((28, 28)), 10))
		cv.waitKey(0)

	cv.destroyAllWindows()


def main_workbench(images: np.ndarray) -> None:

	for img in images:
		# Reshape image
		img = img.reshape((28, 28))
		img_cropped = img_crop_content(img)

		# Show original image
		cv.imshow("Original", img_scale(img, 10))
		# Show centered image
		cv.imshow("Cropped", img_scale(img_cropped, 10))
		# Show resized cropped image
		cv.imshow("Resized", img_scale(cv.resize(img_cropped, (28, 28)), 10))

		cv.waitKey(0)


def main() -> None:
	options = {
		0: main_first,
		1: main_scikit,
		2: main_cv,
		3: main_workbench
	}

	# read images
	path_shapes = '../assets/shapes/'
	circles = read_images(path_shapes + 'circles')
	squares = read_images(path_shapes + 'squares')
	triangles = read_images(path_shapes + 'triangles')
	images = np.concatenate((circles, squares, triangles))

	# process images
	images = np.asarray([preprocess_image(img) for img in images])
	# shuffle images
	np.random.shuffle(images)

	# run main
	options[2](images)


if __name__ == "__main__":
	main()
