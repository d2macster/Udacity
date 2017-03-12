**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

For this project I downloaded a set of `vehicle` images from https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip and `non-vehicle` images from https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip.

I started by reading in all the `vehicle` and `non-vehicle` images. The code is located in `Processor/main.py`, lines 50-51. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
<img src="examples/car.png" width="64">
<img src="examples/notcar.png" width="64">

This project requires tuning many parameters, e.g. color space, HOG parameters, sliding window parameters, carrying heat map between video images. Before going crazy into optimization loop, I started with using the default settings for HOG: `RGB` color space, `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`,  `hog channel = 0`, and performed training / validation process.
The code is located in `Processor/scale_train.py`. I discovered that my validation accuracy was ~ 92% on the provided labeled data sets. When I changed color space to `YUV` color space, the accuracy jumped to ~ 94%. When I switched to `YCrCb` color space, the accuracy increased to ~ 98%, and finally when used `hog channel = ALL` the accuracy became 100%. At this point I did not want to spend any more time optimizing the result without seeing how good / bad car detection will work in practice. 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for channel 0:
for car
<img src="examples/car.png" width="64">
<img src="examples/car_hog_ch0.png" width="128">
and not car 
<img src="examples/notcar.png" width="64">
<img src="examples/notcar_hog_ch0.png" width="128">

####2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features.

The code is located in `/Processor/scale_train.py`. For both `car` and `not car` data sets we extract HOG features, notmalize them using `X_scaler = StandardScaler().fit(X)` and `scaled_X = X_scaler.transform(X)`. Then we split inpiut dta aset into train and test subsets using `sklearn.model_selection.train_test_split`, and finally train linear svm classifier using `svc = sklearn.svm.LinearSVC()`. Finally we save both `X_scaler` and `svc` using `sklearn.externals.joblib`. We need these objects to do pattern matching in the actial video images later. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search routine is located in `/Processor/detection.py` and is called `def find_cars():`. Sliding window is a computationally expensive operation. Our training data set with `cars` and `notcars` contains images on `64x64` pixels, thus we need to convert our windows to that standard size. Instead of taking small image patches and resize them to standard `64z64`, i was resizing the whole image, compute HOG for it, anf then slide cell by cell , where `cell = pixels_per_cell` parameter for HOG computation. I explored video images to understand what car size are we supposed to detect, and set scale parameter to be `scale_list = [0.8, 1, 1.2, 1.5, 1.7, 2, 2.3, 2.5, 2.7, 3]` in `/Processor/main.py`. Thus effectively we can search for cars of size `52x52` to `192x192` pixels.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I decided to use `YCrCb` color space and HOG as features. I did not use any color features : as I showed earlier, even with HOG featutes i got 100% accuracy on test data set. 
<img src="examples/sliding_window_search.png" width="620">
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

!video[ project_video_output.mp4 ]( ./project_video_output.mp4 )



####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

When we apply HOG to a single image , we may received faulse positive signals. Let us closely examinene a sequence of 6 video images.

### Here are six frames

<img src="examples/img0981.jpg" width="192">
<img src="examples/img0982.jpg" width="192">
<img src="examples/img0983.jpg" width="192">
<img src="examples/img0984.jpg" width="192">
<img src="examples/img0985.jpg" width="192">
<img src="examples/img0986.jpg" width="192">

Now let us plot their corresponding heat maps: sum of all positive detections, as described in `Processor/detection.py` , `def find_cars` routine.

### Here are their corresponding heatmaps:

<img src="examples/heat_981.png" width="192">
<img src="examples/heat_982.png" width="192">
<img src="examples/heat_983.png" width="192">
<img src="examples/heat_984.png" width="192">
<img src="examples/heat_985.png" width="192">
<img src="examples/heat_986.png" width="192">

We observe some phantom detections, in addition to true detection of those two cars.
To mitigate the problem, i incorporate historic lookup: 6 frames back. For each individual heat map i filter out values which are bellow a threshould, and then 

### Here is the processed cumulative heat map

<img src="examples/cumulative_heat.png" width="192">

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project required a lot of parameter tuning. I had to narrow down my search space to what features to use, how to scale sliding windows, how to implement false positive rejection. Some sort of kalman filter would help more to identify which detection is a vehicle and which one is not, but I didnt have time to do this. 

Another dimention for improment is classifier itself. I used linear svm, however by now we learned a very powerful tool like neural networks. Using Udacity car dataset in combination with a NN should be the next step to improve the project.
