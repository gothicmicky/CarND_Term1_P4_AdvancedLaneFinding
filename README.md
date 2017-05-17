# Advanced Lane Finding

## Udacity Self Driving Car Engineer Nanodegree - Project 4

![Final Result Gif](./images/project_vid.gif)

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

[image1]: ./output_images/1_corners_found/corners_found1.jpg "Chessboard Conors"
[image2]: ./output_images/2_distortion_corrected_chessboard/0.png "Undistorted Chessboard"
[image3]: ./output_images/2_distortion_corrected/0.png "Undistorted"
[image4]: ./output_images/3_color_gradient_transformd/0.png "Binary Example"
[image5]: ./output_images/4_birdsseye/0.png "Warp and Histogram Example"
[image6]: ./output_images/5_fitlines/0.png "Fitted Lines"
[image7]: ./output_images/CurvatureFormula.png ""
[image8]: ./output_images/7_visualization/0.png "Visualize Fitted Lines"

[video1]: ./project_video.mp4 "Video"

### Code:
This project requires python 3.5 and the following dependencies:
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [OpenCV](http://opencv.org/)
- [MoviePy](http://zulko.github.io/moviepy/)

### Step 1: Camera Calibration
In this step, I used the OpenCV functions `findChessboardCorners` and `drawChessboardCorners` to identify the locations of corners on a series of pictures of a chessboard taken from different angles.

![alt text][image1]

Next, the locations of the chessboard corners were used as input to the OpenCV function `calibrateCamera` to compute the camera calibration matrix and distortion coefficients. 

#### Example of a distortion corrected calibration image.
![alt text][image2]


## Pipeline (single images)

### Step 2: Distortion Correction
The camera calibration matrix and distortion coefficients calculated in the previous step were used with the OpenCV function `undistort` to remove distortion from highway driving images.

![alt text][image3]

Note that if you compare the two images, especially around the edges, there are noticable differences between the original and undistorted image, indicating that distortion has been removed from the original image.

### Step 3: color tranform and gradient threshold
* Threshold x gradient (for grayscaled image)
* Threshold colour channel (S channel)
* Combine the two binary thresholds to generate a binary image.
* The parameters (e.g. thresholds) were determined via trial and error (see Discussion). 
* Improvement: determine the parameters in a more rigorous way.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at ln [30] in `p4_advanced_lane_finding.ipynb`). 

#### Example of a thresholded binary image
![alt text][image4]

### Step 4: perspective transform
* Select only a hard-coded region (lower half of the image) of interest using a binary mask.
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

### Step 5: Detect Lane Lines
I started by taking a histogram of the warpped image. To reduce noise, only the bottom half of the image are used as the lane lines only appear in the bottom half. Then the peak of the left and right halves of the histogram were identified and used as the starting point for the left and right lines. Note that if the car drifted away from the center of the lane (e.g. change lanes), this approach would no longer work.

Theh next step is to apply sliding windows from the starting point (bottom of the image) up to the top of the image. The width of the window is 100 pixels. The threshold for the mininum number of pixels is 50 for the window to be qualified as containing lanes.

Once the left and right line pixel positions are extraced, a second order polynomail fit is applied using np.polyfit. Below is an example of the fitted lines.

![alt text][image6]

### Step 6: Curvature and Lane Position
The radius of curvature and lane position were calculated using the fomular below in ln [24]

![alt text][image7]

### Step 7: Visualize fitted line

I implemented this step in ln [27].  Here is an example of my result on a test image:

![alt text][image8]

---

## Pipeline (video)
I condensed the operations into a single function `process_image` in the ipynb. In the video pipeline, I have added fit_continuous() function to process the 2nd frame and beyond as we could simply fit the line from previously identified line positions and using a margin. I also added santify_check() function to perform santi check on each line detection based on the assumption that the lane line coeffcients should not change beyond a certain threashold compare to the previous frame. If the detection failed, the previouly fitted lines will be used instead.

[![Video output](https://img.youtube.com/vi/TnCGr3EAxP0/0.jpg)](https://youtu.be/TnCGr3EAxP0 "Video output")
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

