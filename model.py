import csv
import ntpath
import os
import numpy as np
import matplotlib.image as mpimg

lines = []
# Change PATH to current dataset
PATH = '../P3_data_sharp_corners/'

# csv file contains image filenames
with open(PATH + 'driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

# This is inefficient and should probably use a
# generator, but it worked fine without
for i,line in enumerate(lines):
	# Get centre, left and right images
	source_path_c = line[0]
	source_path_l = line[1]
	source_path_r = line[2]
	for j,k in enumerate([source_path_c, source_path_l, source_path_r]):
		# simulator run on windows so use ntpath to get filename
		filename = ntpath.basename(k)
		current_path = os.path.join(PATH + 'IMG/', filename)
		# Try - Except for image read issues
		try:
			image = mpimg.imread(current_path)
		except:
			print('imread failed')
			break
		measurement = float(line[3])
		images.append(image)
		# if centre image
		if j==0:
			measurements.append(measurement)
		correction = .2
		# if left image, turn right more
		if j ==1:
			measurements.append(measurement+correction)
		# if right image turn left more
		if j ==2:
			measurements.append(measurement-correction)
		# flip image and measurement for diversity
		image_flipped = np.fliplr(image)
		measurement_flipped = -measurement
		images.append(image_flipped)
		measurements.append(measurement_flipped)

print("read {} images".format(i*4))

# generator was not necessary using aws
X_train = np.array(images)
y_train = np.array(measurements)
# check training examples
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
assert(len(X_train)==len(y_train))

from sklearn.utils import shuffle

# shuffle training data
X_train, y_train = shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, \
Dropout, Convolution2D, MaxPooling2D, Lambda, Cropping2D

# Final Neural Network using Keras
model = Sequential()
# Crop image to useful region
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# Normalise, 0 mean
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# Three convolutional-max pooling-activation layers
model.add(Convolution2D(16, 10, 10))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 10, 10))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 10, 10))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
# Flatten for fully connected layers
model.add(Flatten())
# Implement drop-out on fully connected layers to prevent over-fitting
model.add(Dropout(0.5))
# Three fully connected layers
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
# Final layer is regression only, no activation for classification
model.add(Dense(1))

# Load model from last time if it was OK, and improve it
from keras.models import load_model
model = load_model('model.h5')

from keras.callbacks import EarlyStopping
# Compile model
model.compile(loss='mse', optimizer='adam')
# Stop learning if not improving much
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, \
			       patience = 0)
# Train model
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, \
	  callbacks=[early_stopping])
# Save model
model.save('model.h5')
