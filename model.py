import csv
import ntpath
import os
import numpy as np
import matplotlib.image as mpimg

lines = []
with open('../P3_data_sharp_corners/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for i,line in enumerate(lines):
	source_path_c = line[0]
	source_path_l = line[1]
	source_path_r = line[2]
	for j,k in enumerate([source_path_c, source_path_l, source_path_r]):
		filename = ntpath.basename(k)
		current_path = os.path.join('../P3_data_sharp_corners/IMG/', filename)
		# Try - Except for when the images are being uploaded
		try:
			image = mpimg.imread(current_path)
		except:
			print('imread failed')
			break
		measurement = float(line[3])
		if j==0:
			images.append(image)
			measurements.append(measurement)
		correction = .2
		if j ==1:
			measurement = (measurement+correction)
		if j ==2:
			measurement = (measurement-correction)
		image_flipped = np.fliplr(image)
		measurement_flipped = -measurement
		images.append(image_flipped)
		measurements.append(measurement_flipped)

print("read {} images".format(i))

X_train = np.array(images)
y_train = np.array(measurements)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
assert(len(X_train)==len(y_train))

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, \
Dropout, Convolution2D, MaxPooling2D, Lambda, Cropping2D

# TODO: Build the Final Neural Network in Keras Here
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(16, 10, 10))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 10, 10))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 10, 10))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))

from keras.models import load_model
model = load_model('model.h5')
from keras.callbacks import EarlyStopping
model.compile(loss='mse', optimizer='adam')
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, \
			       patience = 0)
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, \
	  callbacks=[early_stopping])

model.save('model.h5')
