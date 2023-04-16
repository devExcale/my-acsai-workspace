import my_libs.utils as utils
from my_libs.images import *


def reduce_color(img: np.ndarray, n_colors: int = 5) -> np.ndarray:
	# get image shape
	h, w, c = img.shape
	# convert the image to float
	img = img.astype(np.float32)
	# reshape the image
	img = img.reshape((-1, 3))
	# convert to np.float32
	img = np.float32(img)
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
	ssd, labels, centers = cv.kmeans(img, n_colors, None, criteria, 10, cv.KMEANS_PP_CENTERS)
	# convert back to uint8
	center = np.uint8(centers)
	# map the labels
	res = center[labels.flatten()]
	# reshape the image
	return res.reshape(h, w, c)


def blur_image(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
	# kernel matrix
	kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

	# convolve the image manually
	for i in range(img.shape[0] - kernel_size):
		for j in range(img.shape[1] - kernel_size):
			for c in range(img.shape[2]):
				img[i, j, c] = np.sum(img[i:i + kernel_size, j:j + kernel_size, c] * kernel)

	return img


def main(img_path: str, img_height: int = 0, colors: int = 5, iterate: bool = False) -> None:
	# Read an image
	img_og = cv.imread(img_path)
	if img_og is None:
		print('Could not open or find the image')
		return

	print(f'Image size: {img_og.shape}')

	tick = utils.clock()

	if img_height > 0:
		# Resize the image
		h = img_height
		w = int(h / img_og.shape[0] * img_og.shape[1])

		print(f'Resizing to {w}x{h} [{next(tick)}s]')
		img_og = cv.resize(img_og, (w, h))

		print(f'Done resizing [{next(tick)}s]')

	print(f'Reducing to {colors} colors [{next(tick)}s]')

	# Reduce the colors of the image
	img_reduced = []
	if iterate:
		for i in range(2, colors + 1):
			img_reduced.append(reduce_color(img_og, i))
	else:
		img_reduced.append(reduce_color(img_og, colors))

	print(f'Showing images [{next(tick)}s]')

	# Show the original image
	cv.imshow('Original', img_og)
	# Show the reduced image
	last_reduced = None
	for i, reduced in enumerate(img_reduced):
		cv.imshow(f'Reduced to {i + (2 if iterate else colors)} colors', reduced)
		last_reduced = reduced

	print(f'Finished [{next(tick)}s]')

	# show blurred image
	# cv.imshow('Blurred', blur_image(last_reduced))

	# Wait for a key press
	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	main(
		'C:\\users\\escac\\Desktop\\drip_mimmo.jpg',
		# img_height=640,
		colors=7,
		iterate=True
	)
