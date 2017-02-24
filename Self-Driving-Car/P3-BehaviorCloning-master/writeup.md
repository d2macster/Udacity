# **Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior / use provided data set
* Visualize obtained data, discuss potential pitfalls of applying learning using that data 
* Augment collected data
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/train_cv.png "Train and CV error"
[image3]: ./images/center.jpg "Center Image"
[image4]: ./images/left.jpg "Left Image"
[image5]: ./images/right.jpg "Right Image"
[image6]: ./images/augment_0.jpg "Augmented Image 1"
[image7]: ./images/augment_6.jpg "Augmented Image 2"
[image8]: ./images/augment_9.jpg "Augmented Image 3"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* track1.mp4 and track2.mp4 - recordings of autonomous car driving on both tracks

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. The code uses batch generator for training the model, as well as a generator for validation data set. Thus we can use big data sets which do not fit in memory for training and validation.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed
I tried to copy as close as possible NVIDIA model, with a couple deviations.
Instead of YUV transformation i used original image colors, and instead of subsampling I used pooling layer. After searching the publications on this topic, I found that pooling outperforms subsampling, and thus chose pooling.

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 141 - 180).

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Also I use dropout regularlisation to prevent overfitting.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Also I use continuous data augmentation , like changing image brightness ( line 60), random image translations (line 70) and flip images (line 81). We obtain pretty big data set of not repeating images which minimizes the chance of overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on both tracks.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 240).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left and right cameras with augmented steering angles, and artificial shifts to the data ( with corresponding streeting angle adjastements).

###Model Architecture and Training Strategy

####1. Solution Design Approach

For my experiments I started with a data set provided by Udacity. I also collected data from 4 lapses of driving on track # 2 (lines 200 - 217). As it turned out later, the first data set was totally sufficient.

The first step I tried to recreate NVIDIA network. This project had many degrees of freedom, and i wanted to start with a working network. My major effort was in image processing / image augmentation.

I started with the exploration of the udacity data set and understood that most data points have steering angle equal 0. This is totally normal because we want to stay close as possible to the middle of the lane. My first model trained only on center images tried to maintain steering angle close to 0 and went off track. The model was trying to minimize RMSE, and that would happen when the model would output 0 most of the time. 

The situation started to improve when I added left and right camera images and adjasted values for steering angle, which was 
``` python
angle = k * steering + delta
``` 

(for left image) and 

``` python
angle = k * steering - delta
```

(for right image).


At the end of the process, the vehicle is able to drive autonomously around both tracks autonomously.

####2. Final Model Architecture

The final model architecture (model.py lines 141-180) consisted of a convolution neural network with the following layers and layer sizes:
1. Normalisation layer: 
2. Conv layer with kernel 5x5, output shape 31x98x24, followed by max pooling, elu, and dropout
3. Conv layer with kernel 5x5, output shape 14x47x36, followed by max pooling, elu, and dropout
4. Conv layer with kernel 5x5, output shape 5x22x48, followed by max pooling, elu, and dropout
5. Conv layer with kernel 3x3, output shape 3x20x64, followed by elu, and dropout
6. Conv layer with kernel 3x3, output shape 1x18x64, followed by elu, and dropout
7. fully connected with 100 outputs and dropout
8. fully connected with 50 outputs and dropout
9. fully connected with 10 outputs and dropout
dropout value was set to 0.25

Here is a visualization of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]
![alt text][image8]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
