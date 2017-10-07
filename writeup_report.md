# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[centre]: ./examples/centre.jpg "Centre"
[recov1]: ./examples/recov1.jpg "Recovery Image"
[recov2]: ./examples/recov2.jpg "Recovery Image"
[recov3]: ./examples/recov3.jpg "Recovery Image"
[recov4]: ./examples/recov4.jpg "Recovery Image"
[recov5]: ./examples/recov5.jpg "Recovery Image"
[recov6]: ./examples/recov6.jpg "Recovery Image"

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 video of simulator running in autonomous mode

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Note that drive.py has been modified to work better on my machine by tuning down the PD controller.

The model.py file contains the code for training and saving the convolution neural network. This file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### Overview

My model consists of three stages: input preprocessing (code lines 77-79), three sets of convolutional - max pooling - activation layers (code lines 81-89), and three fully connected layers (code lines 91-100). The final fully connected layer is the regression output.

The model includes RELU layers to introduce nonlinearity (eg. code line 83), and the data is cropped and normalized in the model using Keras cropping and lambda layers (input preprocessing). Input preprocessing is performed in the model because the output of the simulator will need these operations performed as well.

#### Attempts to reduce overfitting in the model

The model contains a 50% dropout layer before the first fully connected layer in order to reduce overfitting (code line 93). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. It was also tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 108).

#### Appropriate training data

I found very quickly that the ability of the model is entirely dependent on the quality, quantity, and type of training data provided. Under recommendation of the course material and forums, and through my own experimentation, I collected data of the following types for training:
* forwards laps (normal)
* backwards laps (normal)
* bridge (normal and recovery)
* sharp corners
* road recoveries
* provided example data

#### Solution Design Approach

The overall strategy for deriving a model architecture was to add layers or increase parameters when the model seemed to require more complexity, and to use the LeNet 5 and NVidia end to end learning CNN for inspiration.

My first step was to follow through sections 7-18 of the behavioral cloning project lesson. I set up a network with a single layer initially to test everything, and found as expected that is was not complex enough for this problem. I added preprocessing to perform normalisation so it would optimise quicker, and cropping so the network would not use irrelevant information.

I built up convolutional layers, flattening, dropout and fully connected layers with activations inbetween inspired by the more complex networks mentioned above. I experimented with adding layers, increasing and decreasing neuron counts in the layers and increasing and decreasing kernel size of convolutional and pooling layers until my model was able to reach a low validation cost (approx. 0.01). My validation cost and learning cost were never too different, I think because the dropout layer prevented overfitting. At this point I was confident that the model was able to learn well, but was still not performing well at driving around the track. I had initially been using only normal forward driving data, and it was clear that I needed to focus on data collection. I suspected that the model was overlearning on the most prominent data, i.e.:
* left turning
* low turning angles
* regular road surface

Because it was ok in these situation but performed poorly on:
* sharp steering angles
* the bridge
* recovering from the road edge
* irregular road markings

I trained the model on a combination of all these types of examples, saved the model, tested it in the simulator, and if it was lacking in a particular driving skill, trained the saved model further on data of that type only.

Validation was performed automatically using the keras validation_split, with the data being shuffled beforehand to prevent ordering affecting learning.
To combat the overfitting, I included dropout in the model and implemented early stopping in the fitting. If the accuracy was no longer increasing by at least 0.01, the training was stopped. This prevents overfitting to the training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160,320,3 RGB image   							| 
| Cropping         		| 50 rows from the tpo and 20 rows from the bottom of the image | 
| Normalisation 		| zero mean etc. |
| Convolution  	| 16 filters, 10x10 kernel, 1x1 stride, valid padding |
| Max pooling	   | 2x2 kernel, valid padding	|
| RELU					|												|
| Convolution  	| 32 filters, 10x10 kernel, 1x1 stride, valid padding |
| Max pooling	   | 2x2 kernel, valid padding	|
| RELU					|												|
| Convolution  	| 64 filters, 10x10 kernel, 1x1 stride, valid padding |
| Max pooling	   | 2x2 kernel, valid padding	|
| RELU					|												|
| Flattening	    |    								|
| Dropout					|					50%							|
| Fully connected		| 64 neurons  									|
| RELU					|												|
| Fully connected		| 32 neurons	|
| RELU					|												|
| Fully connected		| 1 neuron, output of regression		|


#### Creation of the Training Set & Training Process

At first I only captured two regular driving laps and used those to create a model of sufficient complexity to do the task. When I was satistied that the model was sufficiently complex and learning well, I worked on data collection for training the model well because it was not performing well training on just the two regular laps.

It is important for the model to experience (train on) all the different driving modes it might have to perform, and not to overlearn certain types of driving at the expense of others. Therefore, I started by collecting separate data sets that covered all possible driving domains I could think of, listed above under "Appropriate training data". I compiled all these images into a data set of about 25,000 images and trained the model on this. The driving performance of the model was significantly improved, but it still struggled a bit on sharp corners and recoveries, so I trained the saved model again on the recovery and sharp corner data sets only, and this resulted in a good model that could drive the whole track.

Example of regular driving:

![alt text][centre]

The collection of normal laps, sharp corners and the provided data shouldn't need any further explanation, so I will detail the collection of recovery data.

Recovery data was collected by positioning the car to one side of the track, then turning towards the centre, then starting recording while driving back to the centre of the road. I did this on the left side and right sides of the road back to center so that the vehicle would learn to recovery both ways, and on the bridge, around corners, reverse direction etc. so it could generalise. See below for an example:

![alt text][recov1] ![alt text][recov2] ![alt text][recov3]
![alt text][recov4] ![alt text][recov5] ![alt text][recov6]

To augment the data sat, I also flipped images and angles. This, along with other measures, was designed to provide the model with diversity in training data, allowing it to generalise and learn to turn in both directions.

Data was randomly shuffled and 20% set aside for a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Only around 5 epochs was required for initial training, and only 2 for most additional training to fix specific driving behaviours.
