import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
import ntpath

'''
Steps:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''

def camera_calibration():

	nx = 9 # of inner corners in x
	ny = 6 # of inner corners in y
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((ny*nx,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.

	# Make a list of calibration images
	images = glob.glob('camera_cal/*.jpg')

	# Step through the list and search for chessboard corners
	for idx, fname in enumerate(images):
		img = cv2.imread(fname)
		
		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

		# If found, add object points, image points
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)
			# Draw and display the corners
			cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
			directory = 'output_images/1_corners_found/'
			if not os.path.exists(directory):
				os.makedirs(directory)
			write_name = directory+'corners_found'+str(idx)+'.jpg'
			cv2.imwrite(write_name, img)
			#cv2.imshow('img', img)
			#cv2.waitKey(500)
	cv2.destroyAllWindows()
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
	return mtx, dist


def cal_undistort(img, mtx, dist):
	undist = cv2.undistort(img, mtx, dist, None, None)
	return undist

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	print(scaled_sobel)
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
	plt.imshow(binary_output, cmap='gray')
	print(binary_output)
	# Return the result
	return binary_output


# Step 1
# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
mtx, dist = camera_calibration()

test_images = glob.glob('test_images/*.jpg')
for idx, fname in enumerate(test_images):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#printing out some stats and plotting
	#print('This image is:', type(img), 'with dimesions:', img.shape)
	plt.imshow(img) 

	# Step 2: Apply the distortion correction to the raw image.  
	undist = cal_undistort(img,mtx,dist)
	plt.imshow(undist) 
	directory = 'output_images/2_distortion_corrected/'
	if not os.path.exists(directory):
		os.makedirs(directory)
	write_name = directory + ntpath.basename(fname) +'.jpg'
	cv2.imwrite(write_name, undist)

	# Step 3: Use color transforms, gradients, etc., to create a thresholded binary image.
	grad_binary = abs_sobel_thresh(undist, orient='x', thresh_min=10, thresh_max=100)
	directory = 'output_images/3_color_transforms/'
	if not os.path.exists(directory):
		os.makedirs(directory)
	write_name = directory + ntpath.basename(fname) +'.jpg'
	cv2.imwrite(write_name, grad_binary)

	# Step 4: Apply a perspective transform to rectify binary image ("birds-eye view"). 

