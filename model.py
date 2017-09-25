import csv
import ntpath
import os
import numpy as np
import matplotlib.image as mpimg

lines = []
with open('../P3_Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images = []
measurements = []

for i,line in enumerate(lines):
	source_path = line[0]
	filename = ntpath.basename(source_path)
	current_path = os.path.join('../P3_Data/IMG/', filename)
	try:
		image = mpimg.imread(current_path)
		#print("Reading Image: ",i)
	except:
		break
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
print("read {} images".format(i))

X_train = np.array(images)
y_train = np.array(measurements)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
assert(len(X_train)==len(y_train))

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape = (160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=7)

model.save('model.h5')
