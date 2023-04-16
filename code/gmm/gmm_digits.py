from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture

from my_libs.images import *

load_dotenv()

my_env = {
	'SAMPLE_LIMIT': int(os.environ.get('GMM_DIGITS.SAMPLE_LIMIT', 0)),
	'TEST_SAMPLE_LIMIT': int(os.environ.get('GMM_DIGITS.TEST_SAMPLE_LIMIT', 0)),
	'SHOW_CENTROIDS': bool(os.environ.get('GMM_DIGITS.SHOW_CENTROIDS', False)),
	'IMG_SCALE': int(os.environ.get('GMM_DIGITS.IMG_SCALE', 2)),
	'GMM': {
		'N_COMPONENTS': int(os.environ.get('GMM_DIGITS.GMM.N_COMPONENTS', 10)),
		'N_INIT': int(os.environ.get('GMM_DIGITS.GMM.N_INIT', 10)),
		'COVAR_TYPE': os.environ.get('GMM_DIGITS.GMM.COVAR_TYPE', 'full'),
	},
}


def main_gmm(images: np.ndarray) -> None:
	gmm: GaussianMixture

	# Fit the GMM to the data
	gmm = GaussianMixture(
		n_components=my_env['GMM']['N_COMPONENTS'],
		covariance_type=my_env['GMM']['COVAR_TYPE'],
		init_params='k-means++',
		n_init=my_env['GMM']['N_INIT'],
		verbose=2,
	)
	gmm.fit(images)

	# Print whether the model converged
	print(f'Converged: {gmm.converged_}')

	if my_env['SHOW_CENTROIDS']:
		show_centroids(gmm)

	# Get test images
	test_images = get_test_images(
		invert=True,
		crop_content=True,
		size=(48, 48),
	)

	print('Calculating probabilities...')

	# Predict the labels of the test images
	probs = gmm.score(test_images)

	for i, img in enumerate(test_images):
		# Get the probability of the image being in each cluster
		cluster_probs = probs[i]

		# Scale the image
		img = img_scale(img.reshape((48, 48)), my_env['IMG_SCALE'])

		# Print index and probabilities on the image
		cv.putText(img, f'#{i}', (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		for j, prob in enumerate(cluster_probs):
			cv.putText(img, f'[{j}] {prob:.2f}', (0, 15 * (j + 2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)

		# Show the image
		cv.imshow('Dataset', img)
		cv.waitKey(0)

	cv.destroyAllWindows()


def show_centroids(
		gmm: GaussianMixture,
		size: tuple[int, int] = None,
		wait_key: bool = True,
		destroy: bool = False
) -> None:
	# Get the cluster means
	centroids = gmm.means_
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


def get_train_images(
		invert: bool = True,
		crop_content: bool = False,
		size: tuple[int, int] = None,
) -> np.ndarray:
	print('Gathering images...')

	# read images
	grouped_digits = []
	path_digits = os.path.abspath('../../assets/digits/train')
	for i in range(10):
		images = read_images(
			os.path.join(path_digits, str(i)),
			limit=my_env['SAMPLE_LIMIT'],
			flatten=False,
		)
		grouped_digits.append(images)

	# concatenate digits
	digits = np.concatenate(grouped_digits)
	# shuffle images
	np.random.shuffle(digits)

	print('Processing images...')

	# image processing pipeline
	processed_digits = []
	for i, img in enumerate(digits):
		# slice the image to 100x100
		img = img[16:112, 16:112]
		# invert image
		if invert:
			img = 255 - img
		# crop image
		if invert and crop_content:
			img = img_crop_content(img)
		# resize image
		if size:
			img = cv.resize(img, size)
		# flatten image
		img = img.flatten()
		# append image
		processed_digits.append(img)

	print('Gathered images.')

	return np.asarray(processed_digits)


def get_test_images(
		invert: bool = True,
		crop_content: bool = False,
		size: tuple[int, int] = None,
) -> np.ndarray:
	print('Gathering images...')

	# read images
	grouped_digits = []
	path_digits = os.path.abspath('../../assets/digits/test')
	for i in range(10):
		images = read_images(
			os.path.join(path_digits, str(i)),
			limit=my_env['TEST_SAMPLE_LIMIT'],
			flatten=False,
		)
		grouped_digits.append(images)

	# concatenate digits
	digits = np.concatenate(grouped_digits)
	# shuffle images
	np.random.shuffle(digits)

	print('Processing images...')

	# image processing pipeline
	processed_digits = []
	for i, img in enumerate(digits):
		# slice the image to 100x100
		img = img[16:112, 16:112]
		# invert image
		if invert:
			img = 255 - img
		# crop image
		if invert and crop_content:
			img = img_crop_content(img)
		# resize image
		if size:
			img = cv.resize(img, size)
		# flatten image
		img = img.flatten()
		# append image
		processed_digits.append(img)

	print('Gathered images.')

	return np.asarray(processed_digits)


def main() -> None:
	digits = get_train_images(
		invert=True,
		crop_content=True,
		size=(48, 48),
	)
	main_gmm(digits)


if __name__ == "__main__":
	main()
