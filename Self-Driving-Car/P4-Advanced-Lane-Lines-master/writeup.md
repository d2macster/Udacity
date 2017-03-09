**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

All the code is located in Processor folder.

###Video convertion.

In this project I decided to save all intermediate images, which was a hlp with debugging process. The first step for this project is to convert input video in a sequence of images, store those. And at the end assemble processed images in a video stream. Helper fuctions to work with video are in `Processor/video_converter.py`. I use `moviepy.editor` to help with video parsing.

###Camera Calibration

After we successfully converted the video into a  sequence of images, we need to remove distortion which was introduced by the camera. Helper functions are located in `Processor/camera_calibration.py`. Udacity provided a set of camera calibration images:  photos ofchess boards taken in different alngles. I use `cv2.findChessboardCorners` to find the location of the chess board corners, combine them with `objpoints` - expected coordinates of those points in a undistorted image, and generate callibration matrix with the help of `cv2.calibrateCamera`. I pass the obtained matrix into `cv2.undistort` function and undistort video images. Here is an example of how removing distortion works: we take one of the calibration images ( which is distorted), and apply undistirtion procedure. The result of this process is shown here.
<img src="examples/calibration1.jpg" width="200">
<img src="examples/calibration1_u.jpg" width="200">

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The same undistortion procedure we apply to every image we extract from the video stream. Here is an example of undistort operation applied to a video image.
<img src="examples/undistort_0180.jpeg" width="200">


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform functions are located in `Processor/warper.py`. I use `cv2.getPerspectiveTransform` function to compute transformation matrix and `cv2.warpPerspective` to apply this transformation to every image. To construct a transformation matrix, I chose a video image where lanes visually parallel, e.g. Thus after perspective transform lanes should be (almost) parallel. 
<img src="examples/undistort_0180.jpeg" width="200">.
<img src="examples/warped_0180.jpeg" width="200">
One important detail: perspective transform should not cut off lanes when the road is turning. After trial and error, i cam eup with the transform which preserves mostly lanes ( no extra objects), and preserves parallel lanes. Here I visualise the source mask.
<img src="examples/perspective_0180.jpeg" width="200">

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result]([video1])

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

