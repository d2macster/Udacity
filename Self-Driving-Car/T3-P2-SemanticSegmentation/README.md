# Semantic Segmentation
### Udacity Self Driving Car Engineer Semantic Segmentation project: overview
The goals for this project are:
 * to build fully convolutional neural network in Tensorflow using VGG16 network pre-trained on ImageNet as an encoder
 * train this network using labeled data coming from a front facing camera on a car; data 
 will come in pairs : street view and labels marking road / non-road pixels
 * apply trained model to detect and mark pixels corresponding to road on 
 previously unseen images
 #### Results
 KITTI road segmentation results, split into two distinct movie stories
 ![um.gif animation](images/um.gif)
 ![umm.gif animation](images/umm.gif)
### Setup

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
