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
[image2]: ./images/train_cv_error.png "Train and CV error"
[image3]: ./images/center.jpg "Center Image"
[image4]: ./images/left.jpg "Left Image"
[image5]: ./images/right.jpg "Right Image"
[image6]: ./images/augment_0.jpg "Augmented Image 1"
[image7]: ./images/augmented_6.jpg "Augmented Image 2"
[image8]: ./images/augmented_7.jpg "Augmented Image 3"

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

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. The code uses batch generator for training the model, as well as a generator for validation data set. Thus we can use big data sets which do not fit in memory for training 

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed
I tried to copy as close as possible NVIDIA model, with a couple deviations.
Instead of YUV transformation i used original image colors, and instead of subsampling I used pooling layer. After searching the publications on this topic, i found that pooling outperforms subsampling, and thus chose pooling.

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 141 - 180).

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Also I use dropout regulsrlisation to prevent overfitting 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Also I use continuous data augmentation , like changing image brightness ( line 60), do random image translations (line 70) and flip images (line 81). Thus we obtain pretty big data set of not repeating data points which minimizes the chance of overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 240).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left and right cameras with augmented steering angles, and artificial shifts to the data ( with corresponding streeting angle adjastements).

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

For my experiments I used data set provided by udacity, as well as augmented it with extra lapses of driving data collected myself on track # 2 (lines 200 - 217). As it turned out later, the first data set was totally sufficient.

The first step I tried to recreate NVIDIA network. I forgot pooling layers and ended up with a network which didt fit on GPU instance. This lead to debuging, network visualisation, fixing errors. This project had many degrees of freedom, and i wanted to start with a workable network.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

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

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
