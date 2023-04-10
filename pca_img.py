from typing import Generator

from my_libs.images import *


scale_factor = 10
n_components = 6


# noinspection PyPep8Naming
def pca_compress(img: np.ndarray, limit: int = 5) -> np.ndarray:
	# convert the array to float
	img = img.astype(np.float32)

	# compute the mean of the image
	mean = np.mean(img, axis=0)
	# subtract the mean from the image
	img -= mean

	# covariance matrix
	cov = np.cov(img, rowvar=True)
	# compute eigendecomposition on the image
	eig_val, eig_vec = np.linalg.eig(cov)
	# sort the eigenvectors and take the principal 'n=limit' ones
	eig_vec = eig_vec[:, eig_val.argsort()[::-1][:max(1, limit)]]

	# project the image onto the eigenvectors
	proj = np.dot(eig_vec.T, img)
	# reconstruct the image
	rec = np.dot(eig_vec, proj) + mean

	# return the reconstructed image
	return rec.astype(np.uint8)


# noinspection PyPep8Naming
def pca_components(img: np.ndarray, limit: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
	# convert the array to float
	img = img.astype(np.float32)

	# compute the mean of the image
	mean = np.mean(img, axis=0)
	# subtract the mean from the image
	img -= mean

	# covariance matrix
	cov = np.cov(img, rowvar=True)
	# compute eigendecomposition on the image
	eig_val, eig_vec = np.linalg.eig(cov)
	# sort descending the eigenvectors
	idx = eig_val.argsort()[::-1]
	eig_val = eig_val[idx]
	eig_vec = eig_vec[:, idx]

	components = []
	for i in range(max(1, limit)):
		# get the eigenvector
		eig = eig_vec[:, i:i+1]
		# project the image onto the eigenvector
		proj = np.dot(eig.T, img)
		# reconstruct the image
		rec = np.dot(eig, proj) + mean

		# return the reconstructed image
		components.append((eig_val[i], rec))

	return components


def main_compress(img: np.ndarray) -> None:
	# do pca
	compressed = pca_compress(img, n_components)

	# show the image
	cv.imshow('Original', img_scale(img, scale_factor))
	# show the compressed image
	cv.imshow('Compressed', img_scale(compressed, scale_factor))

	cv.waitKey(0)
	cv.destroyAllWindows()


def main_components(img: np.ndarray) -> None:
	# do pca
	components = pca_components(img, n_components)

	# show the image
	cv.imshow('Original', img_scale(img, scale_factor))

	for i, (value, component) in enumerate(components):
		# show the component image
		int_img = component.astype(np.uint8)
		cv.imshow(f'Component {i}', img_scale(int_img, scale_factor))

	# weighted sum of the components
	weighted = np.zeros_like(img, dtype=components[0][1].dtype)
	for i, (value, component) in enumerate(components):
		weighted += component * value
	weighted /= np.sum([value for value, _ in components])
	weighted = weighted.astype(np.uint8)

	# show the weighted sum
	cv.imshow('Weighted Sum', img_scale(weighted, scale_factor))

	cv.waitKey(0)
	cv.destroyAllWindows()


def main() -> None:
	option = {
		0: main_compress,
		1: main_components
	}

	# read an image
	img_path = 'assets/shapes/triangles/drawing(11).png'
	img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
	# invert image
	img = img_invert(img)

	# run application
	option[1](img)


if __name__ == '__main__':
	main()
