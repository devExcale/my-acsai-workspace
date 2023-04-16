# Gaussian Mixture Model

from sklearn.mixture import GaussianMixture
from dotenv import load_dotenv
from my_libs.images import *


def main_gmm(images: np.ndarray) -> None:
	gmm: GaussianMixture

	# Get all variables from environment
	env = os.environ
	n_clusters = int(env.get('CLUSTERING_N', 3))
	gmm_cov = env.get('GMM_COV', 'full')
	option = env.get('OPTION_ASSIGNMENT', 'hard')

	# Fit the GMM to the data
	gmm = GaussianMixture(
		n_components=n_clusters,
		covariance_type=gmm_cov,
		init_params='k-means++',
		n_init=10,
	)
	gmm.fit(images)

	# Print whether the model converged
	print(f'Converged: {gmm.converged_}')

	options = {
		'soft': show_soft_assign,
		'hard': show_hard_assign,
	}

	options[option](gmm, images)


def show_soft_assign(gmm: GaussianMixture, images: np.ndarray):
	# Get the cluster means
	centroids = gmm.means_
	# Convert the centroids to uint8
	centroids = centroids.astype(np.uint8)

	n_clusters = len(centroids)

	# Show the centroids
	for i, img in enumerate(centroids):
		img = img.reshape((28, 28))
		cv.imshow(f'Centroid {i}', img_scale(img, 10))

	# Get the cluster probabilities
	probs = gmm.predict_proba(images)

	for i, img in enumerate(images):
		# Get the probability of the image being in each cluster
		cluster_probs = probs[i]

		# Scale the image
		img = img_scale(img.reshape((28, 28)), 10)

		# Print index and probabilities on the image
		cv.putText(img, f'#{i}', (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		for j, prob in enumerate(cluster_probs):
			cv.putText(img, f'[{j}] {prob:.2f}', (0, 15 * (j + 2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)

		# Show the image
		cv.imshow('Dataset', img)
		cv.waitKey(0)

	cv.destroyAllWindows()


def show_hard_assign(gmm: GaussianMixture, images: np.ndarray):
	# Get the cluster means
	centroids = gmm.means_
	# Convert the centroids to uint8
	centroids = centroids.astype(np.uint8)

	n_clusters = len(centroids)

	# Show the centroids
	for i, img in enumerate(centroids):
		img = img.reshape((28, 28))
		cv.imshow(f'Centroid {i}', img_scale(img, 10))

	# Group images by cluster
	clusters = [[] for _ in range(n_clusters)]
	for i, label in enumerate(gmm.predict(images)):
		clusters[label].append(images[i])

	# Show all images in each cluster
	max_cluster_size = max([len(cluster) for cluster in clusters])
	for i in range(max_cluster_size):
		for j, cluster in enumerate(clusters):
			if i < len(cluster):
				img = cluster[i].reshape((28, 28))
				cv.imshow(f'Cluster {j}', img_scale(img, 10))
		cv.waitKey(0)

	cv.destroyAllWindows()


def main() -> None:
	options = {
		0: main_gmm,
	}

	# get absolute path
	shapes = os.path.abspath('../../assets/shapes')
	path = os.path.join

	# read images
	circles = read_images(path(shapes, 'circles'))
	squares = read_images(path(shapes, 'squares'))
	triangles = read_images(path(shapes, 'triangles'))
	images = np.concatenate((circles, squares, triangles))

	# process images
	images = np.asarray([preprocess_image(img) for img in images])
	# shuffle images
	np.random.shuffle(images)

	# run main
	options[0](images)


if __name__ == "__main__":
	load_dotenv()
	main()
