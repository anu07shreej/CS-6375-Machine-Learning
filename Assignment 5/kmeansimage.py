import numpy as np
import cv2



def main():

	img = cv2.imread('img/Koala.jpg')
	# 3 features for RGB so matrix of 0,3
	Z = img.reshape((-1, 3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
	K = 3

	ret, label, center = cv2.kmeans(Z, K, None, criteria, 5,
	                                cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))

	cv2.imshow('res2', res2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
