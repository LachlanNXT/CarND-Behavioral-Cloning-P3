# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

My model consists of three stages: input preparation (code lines 77-79), three sets of convolutional - max pooling - activation layers (code lines 81-89), and three fully connected layers (code lines 91-100). The final fully connected layer is the regression output.

The model includes RELU layers to introduce nonlinearity (eg. code line 83), and the data is cropped and normalized in the model using Keras cropping and lambda layers (input preparation). Input preparation is performed in the model because the output of the simulator will need these operations performed as well.

#### Attempts to reduce overfitting in the model

The model contains a 50% dropout layer before the first fully connected layer in order to reduce overfitting (code line 93). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. I was also tested by running it through the simulator and ensuring that the vehicle could stay on the track.

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
To combat the overfitting, included dropout in the model and implemented early stopping in the fitting. If the accuracy was no longer increasing by at least 0.01, the training was stopped.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles. This, along with other measures, was designed to provide the model with diversity in training data, allowing it to generalise and learn to turn in both directions.

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
