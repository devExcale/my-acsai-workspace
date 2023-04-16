import os
from typing import Iterable

import cv2 as cv
import numpy as np


def read_images(dir_path: str, limit: int = 0, flatten: bool = True) -> np.ndarray:
	images = []

	# loop through files
	for file in os.listdir(dir_path):
		# check limit
		if 0 < limit <= len(images):
			break

		# read image
		img = cv.imread(os.path.join(dir_path, file), cv.IMREAD_GRAYSCALE)
		if img is None:
			continue

		# turn image into a vector
		if flatten:
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


def img_scale(img: np.ndarray, scale: int | float) -> np.ndarray:
	return cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_NEAREST_EXACT)


def img_crop_content(img: np.ndarray) -> np.ndarray:
	# Get dimensions
	h, w = img.shape

	# Compute vertical and horizontal distributions
	dx = np.sum(img, axis=0)
	dy = np.sum(img, axis=1)

	# Get first and last non-zero pixels
	x_end = 0
	for i, x in enumerate(dx):
		if x > 0:
			x_end = i
	x_start = len(dx) - 1
	for i, x in enumerate(dx[::-1]):
		if x > 0:
			x_start = len(dx) - i - 1
	y_end = 0
	for i, y in enumerate(dy):
		if y > 0:
			y_end = i
	y_start = len(dy) - 1
	for i, y in enumerate(dy[::-1]):
		if y > 0:
			y_start = len(dy) - i - 1

	# Slice the image
	return img[y_start:y_end+1, x_start:x_end+1]


def img_center_image(img: np.ndarray, shape: tuple[int, int] = (28, 28)) -> np.ndarray:
	# Get cropped image
	img = img_crop_content(img)

	# Get dimensions
	h, w = img.shape

	# Compute the pad measures
	pad_left = (shape[0] - w) // 2
	pad_right = shape[0] - w - pad_left
	pad_top = (shape[1] - h) // 2
	pad_bottom = shape[1] - h - pad_top
	pad_thickness = ((pad_top, pad_bottom), (pad_left, pad_right))

	# Pad the image to center it
	return np.pad(img, pad_thickness, 'constant', constant_values=0)


def preprocess_image(img: np.ndarray, shape: tuple[int, int] = (28, 28)) -> np.ndarray:
	# Reshape image
	img = img.reshape(shape)
	# Invert image
	img = img_invert(img)
	# Crop image
	img = img_crop_content(img)
	# Resize image
	img = cv.resize(img, shape)
	# Flatten image
	img = img.flatten()

	return img
