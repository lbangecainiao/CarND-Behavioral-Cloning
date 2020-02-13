# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2020_02_08_13_09_42_157.jpg "Center lane driving"
[image6]: ./examples/center_2020_02_08_13_10_22_249.jpg "Original image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model(The model.py file is under the folder of CarND-Behavioral-Cloning-P3)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 recording the driving images for the final test in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Generally the NIVIDIA network is adopted in this project. 

First, a Lambda layer is created to normalize the input data. (model.py lines 61) 

Second a Cropping layer is implemented to cut the useless information, focusing only on the road. (model.py lines 62) 

Next, three convolution neural network with 5x5 filter sizes and depths between 24 and 64. Each convolution layer is followed by a RELU layer to increase the nonlinearity. (model.py lines 63-65)

Then, two convolution layters with 3x3 filter sizes and 64 depths are created. Each convolution layer is followed by a RELU layer to increase the nonlinearity. (model.py lines 66-67)

Finally, a Flatten layer and four fully connected layers are created before the output. The dimensions of the fully connected layers vary from 100 to 1. 

#### 2. Attempts to reduce overfitting in the model

The model contains L2 regularizer layer in order to reduce overfitting (model.py lines 72). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 74). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. In totaly two lap of data are adopted to train the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The transfer learning technique is implemented to define the architecture of the model in this project.

My first step was to verity the image could be read and processed by the neural network, and we could train the neural network and save the model. Thus only a flatten layer is implemented to connect a single output layer. I finally succeeded in running the model.h5 in the simulator even the car drove terribly.

The next step was to create the normalizing layer to preprocess the image. 

In order to gauge how well the was working, I split my image and steering angle data into a training and validation set to detect the overfitting.

Then the LeNet network was implemented. However, after several experiments with different hyperparameters(correction factor, batch size, Epochs) it still wasn't able to achieve a satisfactory results.

Then the NIVIDIA network was used to replace the LeNet network.After several tests,I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I added a L2 regularizer to reduce the overfitting. Finally it achieved 0.0031 training loss and 0.0044 validation loss.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I tried to fine tuning the hyper parameters(Epochs, steering angle correction factors, batch size). And tested the model on the simulator again and again.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 61-72) consisted of a Lambda layer, a Cropping layer, followed by 5 convolution layers with a RELU activation layer after each layer. A Flatten layer and 4 fully connected layers.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would increase the generalization ability of the model. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 18888 number of data points. I then preprocessed this data by applying the normalization.

I finally randomly shuffled the data set and put 25% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
