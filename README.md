# Advanced Lane Finding

## Udacity Self Driving Car Engineer Nanodegree - Project 4

![alt text][image0]

The goal of this project is to develop a pipeline to process a video stream from a forward-facing camera mounted on the front of a car, and output an annotated video which identifies:
- The positions of the lane lines 
- The location of the vehicle relative to the center of the lane
- The radius of curvature of the road

The pipeline created for this project processes images in the following steps:
- **Step 1**: Apply distortion correction using a calculated camera calibration matrix and distortion coefficients.
- **Step 2**: Apply color and gradient thresholds to create a binary image which isolates the pixels representing lane lines.
- **Step 3**: Apply a perspective transformation to warp the image into a bird's-eye view perspective of the lane lines.
- **Step 4**: Identify the lane line pixels and fit polynomials to the lane boundaries.
- **Step 5**: Determine curvature of the lane and vehicle position with respect to center.
- **Step 6**: Warp the detected lane boundaries back onto the original image.
- **Step 7**: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/output.gif "Project Output"
[image1]: ./output_images/1_corners_found/corners_found1.jpg "Chessboard Conors"
[image2]: ./output_images/2_distortion_corrected_chessboard/0.png "Undistorted Chessboard"
[image3]: ./output_images/2_distortion_corrected/0.png "Undistorted"
[image4]: ./output_images/3_color_gradient_transformd/0.png "Binary Example"
[image5]: ./output_images/4_birdsseye/0.png "Warp and Histogram Example"
[image6]: ./output_images/5_fitlines/0.png "Fitted Lines"
[image7]: ./output_images/CurvatureFormula.png ""
[image8]: ./output_images/7_visualization/0.png "Visualize Fitted Lines"

### Code:
This project requires python 3.5 and the following dependencies:
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [OpenCV](http://opencv.org/)
- [MoviePy](http://zulko.github.io/moviepy/)

All source code are located in the iPython notebook: p4_advanced_lane_finding.ipynb.

## Camera Calibration
In this step, I used the OpenCV functions `findChessboardCorners` and `drawChessboardCorners` to identify the locations of corners on a series of pictures of a chessboard taken from different angles.

![alt text][image1]

The OpenCV checkerboard calibration program is based on Professor Z. Zhang's paper "Z. Zhang. "A flexible new technique for camera calibration". 

Calibration process calculates intrinsics (camera 3D coordinates to image 2D coordinates using pinhole camera model) and extrinsics (Rotation R and Translation T from world 3D coordinates to camera 3D coordinates). [Mathoworks explanation page](https://www.mathworks.com/help/vision/ug/camera-calibration.html) on this. 

OpenCV implementation source code can be found [here](https://github.com/opencv/opencv/blob/master/modules/calib3d/src/calibration.cpp). The OpenCV implementation involves non-linear optimization, such as Levenberg-Marquardt method, as it is a non-linear model, such as the distortion models. 

Next, the locations of the chessboard corners were used as input to the OpenCV function `calibrateCamera` to compute the camera calibration matrix and distortion coefficients. 

#### Example of a distortion corrected calibration image.
![alt text][image2]


## Pipeline (single images)

### 1. Distortion Correction
The camera calibration matrix and distortion coefficients calculated in the previous step were used with the OpenCV function `undistort` to remove distortion from highway driving images.

![alt text][image3]

Note that if you compare the two images, especially around the edges, there are noticable differences between the original and undistorted image, indicating that distortion has been removed from the original image.

### 2: color tranform and gradient threshold
* Threshold x gradient (for grayscaled image)
* Threshold colour channel (S channel)
* Combine the two binary thresholds to generate a binary image.
* The parameters (e.g. thresholds) were determined via trial and error (see Discussion). 
* Improvement: determine the parameters in a more rigorous way.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at ln [30] in Color_and_Gradient_Thresh() function in `p4_advanced_lane_finding.ipynb`). 

#### Example of a thresholded binary image
![alt text][image4]

### Step 4: perspective transform
* Select the area of interest (the lower half of the image) using a binary mask.
* Transform the image from the car camera's perspective to a birds-eye-view perspective.
* Hard-code the source and destination polygon coordinates and obtain the matrix `M` that maps them onto each other using `cv2.getPerspective`.
* Warp the image to the new birds-eye-view perspective using `cv2.warpPerspective` and the perspective transform matrix `M` we just obtained.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

### 4: Detect Lane Lines
I started by taking a histogram of the warpped image. To reduce noise, only the bottom half of the image are used as the lane lines only appear in the bottom half. Then the peak of the left and right halves of the histogram were identified and used as the starting point for the left and right lines. Note that if the car drifted away from the center of the lane (e.g. change lanes), this approach would no longer work.

Theh next step is to apply sliding windows from the starting point (bottom of the image) up to the top of the image. The width of the window is 100 pixels. The threshold for the mininum number of pixels is 50 for the window to be qualified as containing lanes.

Once the left and right line pixel positions are extraced, a second order polynomail fit is applied using np.polyfit. Below is an example of the fitted lines.

![alt text][image6]

### 5: Curvature and Lane Position
The detected lane is fit to a 2nd order polynomial curve $$f(y)=Ay^2+By+C$$. The radius of curvature is calculated using the fomular below in ln [24]:

```python
def curvature(left_fit, right_fit, binary_warped):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty)
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    center = (((left_fit[0]*720**2+left_fit[1]*720+left_fit[2]) +(right_fit[0]*720**2+right_fit[1]*720+right_fit[2]) ) /2 - 640)*xm_per_pix
    
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    return left_curverad, right_curverad, center
```

![alt text][image7]

### Step 7: Visualize fitted line

I implemented this step in ln [27].  Here is an example of my result on a test image:

![alt text][image8]

---

## Pipeline (video)
I condensed the operations into a single function `process_image` in the ipynb. In the video pipeline, I have added fit_continuous() function to process the 2nd frame and beyond as we could simply fit the line from previously identified line positions and using a margin. I also added santify_check() function to perform santi check on each line detection based on the assumption that the lane line coeffcients should not change beyond a certain threashold compare to the previous frame. If the detection failed, the previouly fitted lines will be used instead.

[![Video output](https://youtu.be/mudLDc6hJsM/0.jpg)](https://youtu.be/mudLDc6hJsM/0.jpg)
---

## Discussion/ToDos
* 1: Noise interfering with detection of lane lines, resulting in lines with higher curvature being drawn
    Solution: increase the minimum threshold for the x gradient from 20 to 40 to filter out noise. (Increasing it to 50 left out parts of the lane.)

* 2: Sanity check algorithm failed when lane line curvature changed dramatically. 
    The santiy check was performed by calculating the tangent between left and right in two points, and check if it is in a reasonable threshold. This approach is too fragile when the lane lines change direction frequenly and the threshold needs to be manully tuned per video. I have implemented a reset feautre to start from the scratch usign histograms and sliding windows to detect the lanes, which helped with the challenge video. 
    Futurue improvement: Checking that newly fitted lines have similar curvature indead of checking the slope.
* 3: Several vertical edges sit close to each other (such as in challenge_video.mp4), which easily confuses the searching window. This has to be resolved using better binary filter technique, probably adaptive ones. 
* 4: Brightness and light reflection sensitivity: In harder_challenge_video.mp4, there are cases input video image suddenly becomes much brighter that filtering binary, with fixed threshold, cannot differentiate features. Again, adaptive filtering threshold and/or input image histogram equalization should both help.

    
