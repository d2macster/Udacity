# Finding Lane Lines on the Road

## Udacity Self Driving Car Engineer: Finding Lane Lines on the Road project: overview
The lines on the road show us where the lanes are. They act as our 
 constant reference for where to steer the vehicle.  One 
 of the first things we would like to do in developing a self-driving car 
 is to automatically detect lane lines using an algorithm.

In this project we are using opencv + python to perform the following steps:
* convert the original image into its grey-scale representation
* apply canny edge detection to the obtained grey-scale image
* define area of interest on the image to filter out areas where 
we expect to detect highway lanes. we assume that images are obtained from a 
front-facing camera located in the middle of the dash board. The car will be
 driving approximately in the middle of the lane
* apply Hough Line Transform to the masked area of interest , and detect lanes
* finally draw lines on top of highway lanes, and apply the algorithm to 

### Results
At first we will demonstrate how algorithm performs on static images.
This is a necessary tuning step before we apply this transform to a video.

Original image             |  Lane detection
:-------------------------:|:-------------------------:
![whiteCurve](test_images/solidWhiteCurve.jpg)  |  ![whiteCurve](test_images_output/lanes_solidWhiteCurve.jpg)
![whiteRight](test_images/solidWhiteRight.jpg)  | ![whiteCurve](test_images_output/lanes_solidWhiteCurve.jpg)
![yellowCurve](test_images/solidYellowCurve.jpg) | ![yellowCurve](test_images_output/lanes_solidYellowCurve.jpg)
![yellowCurve2](test_images/solidYellowCurve2.jpg) | ![yellowCurve2](test_images_output/lanes_solidYellowCurve2.jpg)
![yellowLeft](test_images/solidYellowLeft.jpg) | ![yellowLeft](test_images_output/lanes_solidYellowLeft.jpg)
![whiteSwitch](test_images/whiteCarLaneSwitch.jpg) | ![whiteSwitch](test_images_output/lanes_whiteCarLaneSwitch.jpg)


Original video             |  Lane detection video
:-------------------------:|:-------------------------:
[![yellowLane](https://img.youtube.com/vi/YpwzumuZIQ4/0.jpg)](https://youtu.be/YpwzumuZIQ4) | [![yellowLane](https://img.youtube.com/vi/yBHEQAlq6eE/0.jpg)](hhttps://youtu.be/yBHEQAlq6eE)
[![whiteLane](https://img.youtube.com/vi/gKVAncmPWnA/0.jpg)](https://youtu.be/gKVAncmPWnA)  | [![whiteLane](https://img.youtube.com/vi/G3M5t6sOvzc/0.jpg)](https://youtu.be/G3M5t6sOvzc)