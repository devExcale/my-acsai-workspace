import os

import cv2 as cv
import numpy as np


def read_images(dir_path: str) -> np.ndarray:
	images = []

	for file in os.listdir(dir_path):
		# read image
		img = cv.imread(os.path.join(dir_path, file), cv.IMREAD_GRAYSCALE)
		if img is None:
			continue

		# turn image into a vector
		img = img.flatten()

		images.append(img)

	return np.asarray(images)


def img_invert(img: np.ndarray) -> np.ndarray:
	return 255 - img


def img_threshold(img: np.ndarray, threshold: int, binary: bool = False) -> np.ndarray:
	for i, pixel in enumerate(img):
		if pixel < threshold:
			img[i] = 0
		elif binary:
			img[i] = 255
	return img


def img_scale(img: np.ndarray, scale: int) -> np.ndarray:
	return cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_NEAREST_EXACT)


def image_center(img: np.ndarray) -> tuple:
	# remove the color dimension
	img = img.squeeze()
	# get the dimensions of the image
	_dims = img.shape[:2]
	# get the joint probability distribution of the image
	distr = img / np.sum(img, axis=(0, 1))
	# get the probability distribution of each axis
	dx = np.sum(distr, axis=0)
	dy = np.sum(distr, axis=1)
	# compute the expected value of each axis
	ax = dx * np.arange(_dims[1])
	ay = dy * np.arange(_dims[0])
	cx = int(np.sum(ax))
	cy = int(np.sum(ay))

	return cx, cy
