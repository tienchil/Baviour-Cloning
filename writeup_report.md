# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data/center1.jpg "Centre view"
[image2]: ./data/recovery1.jpg "Recovery From the Left"
[image3]: ./data/recovery2.jpg "Recovery From the Left"
[image4]: ./data/recovery3.jpg "Recovery From the Left"
[image5]: ./data/flipped1.jpg "Before Flipping"
[image6]: ./data/flipped2.jpg "After Flipping"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with several concolutional layers. There are 3 convolutional layers with 5x5 filters and depth of 24, 36, and 48, and 2 convolutional layers with 3x3 filters and depth of 64.  

All convolutional layers with 5x5 filters have RELU activations, and those with 3x3 filters have dropout layers after each one. The data is normalized in the model using a Keras lambda layer

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 59, 61). 

The model was trained and validated on different data sets to ensure that the model was not overfitting, with 70-30 split on the data for the training set and the validation set (model.py line 19). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 66).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use nVidia model as template and then modified from there.

The first step was to crop and resize the data to reduce the number of computations. Since only the lane curvature matters, I resized all data to 64x64x3. The data was then fed into my model.

The model, as described above, had 5 convolutional layers and 3 fully-connected layers. The last layer only had one node because the car only needed to know how much angle to turn. To combat the overfitting, I added a couple of Dropout layers after convolutional layers with drop rate 0.1. This prevented model from overfitting and gave a better result than using RELU.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set, with ratio 70-30. The final step was to run the simulator to see how well the car was driving around track one. 

There were a few spots where the vehicle fell off the track. For example, the bridge and the first curve after the bridge were particularlly hard for the vehicle. To improve the driving behavior in these cases, I used the strategy of recovery laps and wiggling. By recording several recovery laps, the network would learn how to get back on track. Wiggling while recording also helped the network to learn to stay off the sides.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 53-70) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 | Description                         |
|:---------------------:|:------------------------------------|
| Input                 | 64x64x3 RGB images                  |
| Convolution 5x5       | 2x2 stride, valid padding, depth 24 |
| ReLu                  |                                     |
| Convolution 5x5       | 2x2 stride, valid padding, depth 36 |
| ReLu                  |                                     |
| Convolution 5x5       | 2x2 stride, valid padding, depth 48 |
| ReLu                  |                                     |
| Convolution 3x3       | 2x2 stride, valid padding, depth 64 |
| Dropout               | Drop Rate 0.1                       |
| Convolution 3x3       | 2x2 stride, valid padding, depth 64 |
| Dropout               | Drop Rate 0.1                       |
| Fully Connected       | Output 100                          |
| Fully Connected       | Output 50                           |
| Fully Connected       | Output 1                            |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help the model generalize. For example, here is an image that has then been flipped:

Before:
![alt text][image5]

After:
![alt text][image6]

After the collection process, I had 33249 data points. I then preprocessed this data by cropping top 70 pixels and bottom 25 pixels, since these were not affecting how the vehicle moved. Also, I resized the data to reduce computation costs because only the lane curvature mattered in driving.


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as the validation loss continued to drop during training. I used an adam optimizer so that manually training the learning rate wasn't necessary.
