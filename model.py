import csv
import os
import cv2

lines = []
with open('../P3_Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images = []
measurements = []

for line in lines:
	source_path = line[0]
	filename = os.path.basename(source_path)
	current_path = os.path.join('../P3_Data/IMG/', filename)
	try:
		image = cv2.imread(current_path)
	except:
		break
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape = (160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, \
		  shuffle = True, nb_epoch=7)

model.save('model.h5')