import cv2
import numpy as np


# given an image, perform pca and return the top 3 eigenvectors
def do_pca(img, n_eig=5):

    # turn image into grayscale
    img_2d = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert image to float
    img_2d = img_2d.astype(np.float32)
    # get the image shape
    h, w = img_2d.shape
    # reshape the image to a 2D array
    img_2d = img_2d.reshape((h * w, 1))
    # compute the mean of the image
    mean = np.mean(img_2d, axis=0)
    # subtract the mean from the image
    img_2d -= mean
    # compute the covariance matrix of the image
    cov = np.cov(img_2d, rowvar=False)
    # compute the eigenvalues and eigenvectors of the covariance matrix
    eig_val, eig_vec = np.linalg.eig(cov)
    # sort the eigenvectors by the eigenvalues
    eig_vec = eig_vec[:, eig_val.argsort()[::-1]]
    # get the top 3 eigenvectors
    top_3 = eig_vec[:, :n_eig]
    # compute the new image
    new_img = np.dot(img_2d, top_3)
    # reshape the new image to the original shape
    new_img = new_img.reshape((h, w, 3))
    # add the mean to the new image
    new_img += mean
    # return the new image as uint8
    return new_img.astype(np.uint8)

def main():
    # Read an image
    img = cv2.imread('C:\\Users\\EMANUELE\\Desktop\\1920x1080-intellij-idea.png')
    # compute pca on image
    pca_img = do_pca(img, 5)
    # show the image
    cv2.imshow('image', pca_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()