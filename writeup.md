# **Advanced Lane Finding**

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/distorted_undistorted.png "Distorted and undistorted chessboard"
[image2]: ./examples/pipeline_undistortion.png "Distorted and undistorted test image"
[image3]: ./examples/pipeline_thresholding.png "Thresholded binary image"
[image4]: ./examples/destination_corners.png "Source corners for perspective transform"
[image5]: ./examples/pipeline_warped.png "Warped image with destination corners"
[image6]: ./examples/test_image.png "Test image for lane-line pixel identification from scratch"
[image7]: ./examples/histogram.png "Histogram for lane-line pixel identification from scratch"
[image8]: ./examples/sliding_windows.png "Sliding window approach for lane-line pixel identification from scratch"
[image9]: ./examples/lane-line_pixels_from_scratch.png "Lane-line pixel identification from scratch"
[image10]: ./examples/test_image_with_previous_fit.png "Test image for lane-line pixel identification from previous fit"
[image11]: ./examples/lane-line_pixels_from_previous_fit.png "Lane-line pixel identification from previous fit"
[image12]: ./examples/end_result.png "Output"
[video1]: ./project_video.mp4 "Video"

In this writeup I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! My project code can be found [here](https://github.com/ninemillionbicycles/CarND-AdvancedLaneLines-P2/blob/master/P2.ipynb).

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

`OpenCV` provides the `calibrateCamera()` function that can be used to compute camera matrix and distortion coefficients. The function requires the following inputs: 

* An array `object_points` containing 3D points corresponding to the location of the chessboard corners in real world coordinates.
* An array `image_points` containing 2D points corresponding to the location of these chessboard corners in image space.

The function `calc_calibration_points(images, n_x, n_y)` for obtaining `object_points` and `image_points` from a series of chessboard images is located in cell 2 of the Jupyter Notebook called `P2.ipynb`. Besides the series of chessboard images, it also takes the number of chessboard corners in x and y direction as input. These have to be counted manually according to the following definition: Chessboard corners are the points on the chessboard where two black and two white squares intersect.

In `calc_calibration_points(images, n_x, n_y)`, the following operations are performed:
* I start by preparing `object_points_current`, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z = 0, such that the object points are the same for each calibration image.  Thus, `object_points_current` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
* `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then can use the output `object_points` and `image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

When applying the distortion correction to a test image using the `cv2.undistort()` function, I obtained the following result: 

![alt text][image1]

### Pipeline (single images)

In the following, I will describe the pipeline I built for processing single images (cells 6-21 of `P2.ipynb`).

#### 1. Provide an example of a distortion-corrected image.

First of all, I applied the same `cv2.undistort()` function to the image:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Im implemented 3 different gradient thresholding techniques (cell 7 of `P2.ipynb`):
* The function `abs_threshold()` takes a single channel image (e.g. grayscale), applies the Sobel operator in `orient='x'` or `orient='y'` orientation with the the specified kernel, then takes the absolute value, scales the image back to `(0,255)` and applies the specified threshold.
* The function `mag_threshold()` applies the Sobel operator in x and y orientation separately with the the specified kernel, then calculates the magnitude using `np.sqrt()`, scales the image back to (0,255) and applies the specified threshold.
* The function `dir_threshold()` applies the Sobel operator in x and y orientation separately with the the specified kernel, then calculates the direction of the gradient using `np.arctan2()` and applies the specified threshold.

After experimenting with the different gradient thresholding techniques, I found that combining the `abs_threshold()` technique with color thresholds on the S and L channel of an image in HLS color space yielded the best results. The complete thresholding pipeline is implemented in the function `apply_thresholds(image, S_thresh, L_thresh, sobel_x_L_thresh, sobel_kernel)`.

Here is the result I get when calling this function with parameters `S_thresh=(170, 255)`, `L_thresh=(20, 255)`, `sobel_x_L_thresh=(20, 100)` and `sobel_kernel = 5`:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for obtaining the perspective transform matrix `M` is located inside the function `calc_transform_matrix(image_binary, n_x, n_y, mtx, dist, src, offset)` (cell 10 of `P2.ipynb`). The function takes the undistorted color image for which the perspective transform matrix is to be calculated. The source corners `src` must be determined manually and passed to the function alongside the camera's distortion coefficients `dist` and the target offset of the destination corners from the image edges in the topview image.

I used an interactive `matplotlib` window to determine the polygon corners for a test image:

```python
src = np.float32[(190, y_size), 
                 (1120, y_size),
                 (585, 455), 
                 (695, 455)]
```

`y_size` denotes the size of the input image in y direction. Here is how the chosen source corners look like when drawn on top of a test image:

![alt text][image4]

I specified the destination corners as follows:

```python
dst = np.float32([(src[0][0]+offset, src[0][1]), 
                  (src[1][0]-offset, src[1][1]), 
                  (src[0][0]+offset, 0), 
                  (src[1][0]-offset, 0)])
```

I verified that my perspective transform was working as expected by applying the `cv2.warpPerspective()` function to the same test image used above and drawing the polygon spanned by the destination corners on top of it. If the perspective transform was performed as intended, the lane lines must appear parallel in the warped image:

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the pipeline I built for processing a video, I used the following general strategy to identify lane-line pixels:

If there are no lines in the lines history, identify lane-line pixels from scratch using a histogram and a sliding windows approach. This occurs when the image that is being processed is the very first image of the video or whenever the last couple of lines were bad. The function `find_lane_pixels()` (cells 12 and 14 of `P2.ipynb`) calculates the lane-line pixels based on this approach. 

Here is a binary test image, the corresponding histogram obtained by summing vertically across the image pixels, a visualization of the sliding windows approach based on the obtained histogram and the identified left (red) and right (blue) lane-line pixels:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

If there are some lines in the lines history, identify lane-line pixels by searching around the polynomial calculated for the previous image. The function `search_around_polynomial()` (cell 15 of `P2.ipynb`) calculates the lane-line pixels based on this approach.

Here is a visualization of the result from `search_around_polynomial()` when applied to the same test image and a fit from an (obviously quite different) fictional previous image:

![alt text][image10]

Here are the identified left (red) and right (blue) lane-line pixels for this approach:

![alt text][image11]

The calculation of the corresponding polynomial is located inside the `fit_polynomial()` function (cells 13 of `P2.ipynb`). Based off the identified lane-line pixels the function uses the Numpy `np.polyfit()` function to calculate the polynomial.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function `measure_curvature_real()` (cell 25 of `P2.ipynb`) calculates the radius of curvature for a given polynomial fit at the point `y_eval` according to the following equation:

\begin{equation}
R_{curve} = \frac{(1 + (2Ay + B)^2)^{3/2}}{\mid 2A \mid}
\end{equation}

The function `calculate_line_base_position()` (cell 29 of `P2.ipynb`) calculates the distance of the vehicle from a given line. This function is called to obtain the distances of the vehicle from both left and right lane line in the `process_image()` function (cell 30 of `P2.ipynb`). The position of the vehicle with respect to center is then calculated as the absolute of the difference of `line_left.line_base_position` and `line_right.line_base_position`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `to_original_image()` (cell 21 of `P2.ipynb`). Here is an example of my result:

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I implemented the pipeline for a single image step by step as described above. I then implemented the function `process_image()` that performs the following steps:

1. Undistort the image using the known camera calibration parameters
2. Apply some thresholds
3. Warp the image to top view using the perspective transform matrix
4. Detect the lane-line pixels and fit them to a polynomial to find the corresponding lane boundaries
5. Calculate the radius of curvature of both lane boundaries
6. Calculate the offset of the vehicle with respect to center
7. Do the following sanity checks on the detected lane boundaries and add them to the history if the sanity checks pass:
    * Check if the distance of both lane boundaries is plausible
    * Check if both lane boundaries are roughly parallel
8. Average the detected lane boundaries over last `n` frames in the history
9. Draw the detected lane boundaries and the corresponding lane onto the image
10. Warp the image back to the original view using the inverse of the perspective transform matrix

After having implemented the complete pipeline, I spend most of my time optimizing some hyperparameters that are required for steps 2, 7 and 8:

`n = 10` specifies that the averaging of the detected lane boundaries will be computed by using the last 10 images in the history. A large `n` value will lead to very smooth lines across multiple frames but also results in longer response times of the pipeline. While a larger `n` value might be preferable for highway scenarios, a smaller `n` value will yield better results for curvy roads. It might therefore be necessary to dynamically change this parameter depending on the type of road the vehicle is driving on. The corresponding data could be extracted e.g. from a map.

`sobel_kernel = 5`, `S_thresh=(170, 255)`, `L_thresh=(20, 255)` and `sobel_x_L_thresh=(20, 100)` are now highly optimized to detect the lane boundaries of the project video, where the pipeline I implemented then performed reasonably well. My pipeline did not perform very well on the two challenge videos. It regularly got stuck other nearly vertical structures, e.g. on tar joints or on the physical divider between the own and the oncoming lanes. I suspect that both the image pre-processing part as well as the sanity checks would have to be optimized even more to work with a variety of road surface and light conditions.