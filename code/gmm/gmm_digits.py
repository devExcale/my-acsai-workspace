from typing import Any, Callable

from dotenv import load_dotenv
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from my_libs.images import *

load_dotenv()


def get_env(key: str, default: Any = None, cast: Callable[[str], Any] = None) -> Any:
	var = os.environ.get(key)

	if var is None:
		return default

	if cast is not None:
		return cast(var)

	return var


def cast_bool(x: str) -> bool:
	return x.lower() in ['true', 't', '1', 'yes', 'y']


# TODO: move environment init to a class
my_env = {

	# A multiplier on the showed size of the images
	'IMG_SCALE': get_env('GMM_DIGITS.IMG_SCALE', default=2, cast=int),

	# Whether to show the centroids
	'SHOW_CENTROIDS': get_env('GMM_DIGITS.SHOW_CENTROIDS', default=True, cast=cast_bool),

	# Wheter to test the resulting model (centroids will be always shown to map the labels)
	'DO_TEST': get_env('GMM_DIGITS.DO_TEST', default=False, cast=cast_bool),

	# Dataset parameters
	'SAMPLE': {

		# How many images to use for training (<= 0 means all)
		'LIMIT_TRAIN': get_env('GMM_DIGITS.SAMPLE.LIMIT_TRAIN', default=0, cast=int),

		# How many images to use for testing (<= 0 means all)
		'LIMIT_TEST': get_env('GMM_DIGITS.SAMPLE.LIMIT_TEST', default=0, cast=int),

		# The path to the dataset
		'DATA_PATH': get_env('GMM_DIGITS.SAMPLE.DATA_PATH', default='../../assets/digits'),

	},

	# Gaussian Mixture parameters
	'GMM': {

		# The number of components in the Gaussian Mixture
		'N_COMPONENTS': get_env('GMM_DIGITS.GMM.N_COMPONENTS', default=10, cast=int),

		# How many times the Gaussian Mixture should be initialized
		'N_INIT': get_env('GMM_DIGITS.GMM.N_INIT', default=5, cast=int),

		# The initialization method for the Gaussian Mixture
		'INIT_PARAMS': get_env('GMM_DIGITS.GMM.INIT_PARAMS', default='k-means++'),

		# The covariance type for the Gaussian Mixture
		'COVAR_TYPE': get_env('GMM_DIGITS.GMM.COVAR_TYPE', default='full'),

	},

	# Image processing parameters
	'IMG': {

		# Whether to invert the images
		'INVERT': get_env('GMM_DIGITS.IMG.INVERT', default=False, cast=cast_bool),

		# Whether to crop the images to the content
		'CROP_CONTENT': get_env('GMM_DIGITS.IMG.CROP_CONTENT', default=False, cast=cast_bool),

		# The size to resize the images to
		'SIZE': get_env('GMM_DIGITS.IMG.SIZE', cast=lambda x: tuple(map(int, x.split(',')))),

	},

}


def process_images(images: np.ndarray) -> np.ndarray:
	# slice the image to 100x100
	# img = img[16:112, 16:112]

	og_size = images[0].shape

	# invert images
	if invert := my_env['IMG']['INVERT']:
		print('Inverting images...')
		images = 255 - images

	# crop image
	if invert and my_env['IMG']['CROP_CONTENT']:
		print('Cropping content...')
		images = (img_crop_content(img) for img in images)

	# resize image
	if size := my_env['IMG']['SIZE']:
		print(f'Resizing images to {size}...')
		images = (cv.resize(img, size) for img in images)
	else:
		my_env['IMG']['SIZE'] = og_size

	# flatten image
	images = (img.flatten() for img in images)

	return np.asarray(list(images))


def get_train_images() -> ndarray:
	path = os.path.abspath(my_env['SAMPLE']['DATA_PATH'])
	paths = [os.path.join(path, 'train', str(i)) for i in range(10)]

	# Get train images
	return gather_images(
		paths,
		limit=my_env['SAMPLE']['LIMIT_TRAIN'],
		shuffle=False,
		process_all=process_images,
	)


def get_test_images() -> ndarray:
	path = os.path.abspath(my_env['SAMPLE']['DATA_PATH'])
	paths = [os.path.join(path, 'test', str(i)) for i in range(10)]

	# Get test images
	return gather_images(
		paths,
		limit=my_env['SAMPLE']['LIMIT_TEST'],
		shuffle=False,
		process_all=process_images,
	)


def do_training(reduced_images: np.ndarray) -> GaussianMixture:
	gmm: GaussianMixture

	print('Training...')

	# Fit the GMM to the data
	gmm = GaussianMixture(
		n_components=my_env['GMM']['N_COMPONENTS'],
		covariance_type=my_env['GMM']['COVAR_TYPE'],
		init_params=my_env['GMM']['INIT_PARAMS'],
		n_init=my_env['GMM']['N_INIT'],
		verbose=2,
	)
	gmm.fit(reduced_images)

	print('Done training.')

	return gmm


def do_testing(model: GaussianMixture, centroids_assignments: tuple[list[int]], reduced_images: np.ndarray) -> None:
	print('Testing...')

	# Predict the labels of the test images
	labels = model.predict(reduced_images)

	correct = [0 for _ in range(10)]

	# Assume the test images are ordered by digit
	for i, label in enumerate(labels):
		digit = i // (len(reduced_images) // 10)
		if label in centroids_assignments[digit]:
			correct[digit] += 1

	print(f'Accuracy: {sum(correct) / len(reduced_images):.2f}%')


def do_testing_visual(model: GaussianMixture, reduced_images: np.ndarray, images: np.ndarray) -> None:
	print('Testing...')

	# Predict the labels of the test images
	probs = model.predict_proba(reduced_images)

	size = my_env['IMG']['SIZE']

	for i, img in enumerate(images):
		# Get the probability of the image being in each cluster
		cluster_probs = probs[i]

		# Scale the image
		img = img_scale(img.reshape(size), my_env['IMG_SCALE'])

		# Print index and probabilities on the image
		cv.putText(img, f'#{i}', (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		for j, prob in enumerate(cluster_probs):
			cv.putText(img, f'[{j}] {prob:.2f}', (0, 15 * (j + 2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)

		# Show the image
		cv.imshow('Dataset', img)
		cv.waitKey(0)

	cv.destroyAllWindows()


def assign_centroids(gmm: GaussianMixture) -> tuple[list[int]]:
	# Get the cluster means
	centroids = gmm.means_

	assignments = tuple([] for _ in range(10))
	for label, centroid in enumerate(centroids):
		# Ask the user to assign the label to a digit
		assignments[int(input(f'Assign centroid {label} to digit: '))].append(label)

	# noinspection PyTypeChecker
	return assignments


def show_centroids(
		pca: PCA,
		gmm: GaussianMixture,
		scaler: StandardScaler = None,
		size: tuple[int, int] = None,
		wait_key: bool = True,
		destroy: bool = False
) -> None:
	# Get the cluster means
	centroids = gmm.means_
	# Project the centroids to the original space
	centroids = pca.inverse_transform(centroids)

	if scaler:
		centroids = scaler.inverse_transform(centroids)

	# Convert the centroids to uint8
	centroids = centroids.astype(np.uint8)

	# Parse image size
	if not size:
		side = int(np.sqrt(len(centroids[0])))
		if side * side == len(centroids[0]):
			size = (side, side)
		else:
			raise ValueError('Cannot infer image size')

	# Show the centroids
	windows = []
	for i, img in enumerate(centroids):
		window = f'Centroid {i}'
		windows.append(window)
		img = img.reshape(size)
		cv.imshow(window, img_scale(img, my_env['IMG_SCALE']))

	if wait_key:
		cv.waitKey(0)
	if destroy:
		for window in windows:
			cv.destroyWindow(window)


def gather_images(
		paths: str | Iterable[str],
		limit: int = 0,
		process_all: Callable[[np.ndarray], np.ndarray] = None,
		shuffle: bool = False,
) -> np.ndarray:
	print('Gathering images...')

	if isinstance(paths, str):
		paths = [os.path.abspath(paths)]
	else:
		paths = [os.path.abspath(path) for path in paths]

	print(f'Limiting to {limit} images x {len(paths)} folders -> {limit * len(paths)} images.')

	img_subsets = []
	for path in paths:
		imgs = read_images(
			path,
			limit=limit,
			flatten=False,
		)
		img_subsets.append(imgs)

	# unroll the subsets
	images = np.concatenate(img_subsets)

	# shuffle images
	if shuffle:
		print('Shuffling images...')
		np.random.shuffle(images)

	# apply preprocessing
	if process_all:
		print('Processing images...')
		images = process_all(images)

	print('Done.')

	return images


def pca_reduction(train_images: np.ndarray) -> tuple[np.ndarray, PCA, StandardScaler]:
	pca: PCA

	print('Performing Principal Component Analysis...')

	# Standardize the images
	scaler = StandardScaler()
	std_images = scaler.fit_transform(train_images)

	# noinspection PyTypeChecker
	pca = PCA().fit(std_images)
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('number of components')
	plt.ylabel('cumulative explained variance')

	# get the top component that explains 95% of the variance
	n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1

	print(f'95% variance: {n_components} components.')

	# noinspection PyTypeChecker
	pca = PCA(n_components=n_components)
	return pca.fit_transform(std_images), pca, scaler


def main() -> None:
	# Get the training images
	images = get_train_images()

	# Reduce the images using PCA
	reduced_images, pca, scaler = pca_reduction(images)

	# Train the model
	model = do_training(reduced_images)

	# Print whether the model converged
	print(f'Converged: {model.converged_}')

	if my_env['SHOW_CENTROIDS'] or my_env['DO_TEST']:
		# Show the centroids
		show_centroids(pca, model, scaler=scaler, wait_key=True, destroy=False)

	if my_env['DO_TEST']:
		# Get the test images
		test_images = get_test_images()

		# Reduce the test images using PCA
		reduced_test_images = pca.transform(test_images)

		# Test the model
		do_testing(model, assign_centroids(model), reduced_test_images)


if __name__ == "__main__":
	main()
